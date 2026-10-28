"""
adapters/frameworks/autogen.py

AutoGen framework adapter for CASPIAN.

Implements MASFrameworkAdapter with ExecutionMode.LIVE.
Streams messages from BaseGroupChat.run_stream() and converts them
into CASPIAN ChannelEvents turn by turn.

Supports:
  RoundRobinGroupChat  (mas_type="group_chat")
  Swarm                (mas_type="group_chat")
  MagenticOneGroupChat (mas_type="magentic_one")

Native ACIArena AutoGen backend (ACIArenaNativeRunSpec):
  Uses aciarena.mas.autogen.autogen_mas.AutoGen directly.
  Logs via DebateTraceCollector.log_message(sender, receiver, message).
  Agent topology: user <-> assistant (bidirectional).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import numpy as np

from core.events import ChannelEvent, MASTopology
from adapters.benchmarks.base import Scenario
from adapters.frameworks.base import MASFrameworkAdapter, ExecutionMode
from adapters.frameworks.encoder import EmbeddingEncoder
from adapters.frameworks.autogen_runtime import build_team

try:
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.messages import (
        HandoffMessage,
        MemoryQueryEvent,
        SelectSpeakerEvent,
        TextMessage,
        ThoughtEvent,
        ToolCallExecutionEvent,
        ToolCallRequestEvent,
        ToolCallSummaryMessage,
    )
    from autogen_agentchat.teams import MagenticOneGroupChat
    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False


# ---------------------------------------------------------------------------
# ACI AutoGen agent names (what the native backend logs as sender/receiver)
# ---------------------------------------------------------------------------

# Native ACI AutoGen agents: UserProxyAgent logs as "user", AssistantAgent as "assistant"
# We normalise "user" -> "user_proxy" in _map() so topology uses canonical names
_ACI_AUTOGEN_AGENTS = ["user_proxy", "assistant"]
_ACI_AUTOGEN_EDGES  = [("user_proxy", "assistant"), ("assistant", "user_proxy")]


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def _as_datetime(x: Any) -> datetime:
    if isinstance(x, datetime):
        dt = x
    elif isinstance(x, str):
        try:
            dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# AutoGen framework adapter
# ---------------------------------------------------------------------------

class AutoGenAdapter(MASFrameworkAdapter):

    @property
    def name(self) -> str:
        return "AutoGen"

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.LIVE

    def build_topology(
        self,
        scenario:     Scenario,
        config:       str,
        runtime_spec: Any | None = None,
    ) -> MASTopology:
        """
        Build topology. For ACIArenaNativeRunSpec, use the agent names
        the native ACI backend actually logs (user <-> assistant).
        For TAMAS, use task_data["agents"].
        """
        # ACIArenaNativeRunSpec — use actual ACI AutoGen agent names
        try:
            from adapters.frameworks.specs import ACIArenaNativeRunSpec as _AciSpec
            if isinstance(runtime_spec, _AciSpec) and runtime_spec.framework == "AutoGen":
                return MASTopology(
                    framework=self.name,
                    mas_type="aci_autogen",
                    agents=_ACI_AUTOGEN_AGENTS,
                    edges=_ACI_AUTOGEN_EDGES,
                )
        except ImportError:
            pass

        # TAMAS / AutoGenRunSpec path
        raw_agents: list[str] = []
        try:
            from adapters.frameworks.specs import AutoGenRunSpec as _AGS
            if isinstance(runtime_spec, _AGS) and runtime_spec.agents:
                raw_agents = [a.role for a in runtime_spec.agents]
        except ImportError:
            pass

        if not raw_agents:
            raw_agents = [
                a["agent_name"]
                for a in scenario.task_data.get("agents", [])
                if isinstance(a.get("agent_name"), str) and a["agent_name"].strip()
            ]

        agent_names = [n.lower().replace(" ", "_") for n in raw_agents]
        cfg = config.strip().lower()

        if cfg == "magentic_one":
            mas_type     = "magentic_one"
            orchestrator = agent_names[0]
            workers      = agent_names[1:]
            edges = (
                [(orchestrator, w) for w in workers]
                + [(w, orchestrator) for w in workers]
            )
        else:
            mas_type = "group_chat"
            edges    = [(a, b) for a in agent_names for b in agent_names if a != b]

        return MASTopology(
            framework=self.name,
            mas_type=mas_type,
            agents=agent_names,
            edges=edges,
        )

    async def run_live(
        self,
        scenario:     Scenario,
        topology:     MASTopology,
        config:       str,
        model:        str,
        timeout:      int,
        runtime_spec: Any | None = None,
    ) -> AsyncIterator[list[ChannelEvent]]:

        import asyncio
        import time as _time

        # ── ACIArena native AutoGen backend ──────────────────────────────
        try:
            from adapters.frameworks.specs import ACIArenaNativeRunSpec as _AciSpec
            if isinstance(runtime_spec, _AciSpec):
                from adapters.frameworks.llm_debate import DebateTraceCollector

                def _run_native(spec, col):
                    mas    = spec.backend_factory(col)
                    attack = spec.attack_factory(mas) if spec.attack_factory else None
                    if attack is not None and hasattr(attack, "run"):
                        attack.run(mas)
                    col.set_stage("bootstrap", 0)
                    if all(hasattr(mas, x) for x in ["bootstrap", "step", "conclude"]):
                        args, done = mas.bootstrap(spec.user_query)
                        turn = 0
                        while not done and turn < spec.max_turn:
                            turn += 1
                            if attack is not None and hasattr(attack, "set_turn"):
                                attack.set_turn(turn)
                            col.set_stage("step", turn)
                            args, done = mas.step(args)
                        col.set_stage("conclude", turn + 1)
                        result = mas.conclude(args)
                    elif hasattr(mas, "run"):
                        col.set_stage("run", 0)
                        result = mas.run(spec.user_query)
                    else:
                        raise RuntimeError(f"Unsupported ACI backend: {type(mas)}")
                    if attack is not None and hasattr(attack, "set_answer"):
                        try: attack.set_answer(result)
                        except Exception: pass

                collector = DebateTraceCollector()
                await asyncio.wait_for(
                    asyncio.to_thread(_run_native, runtime_spec, collector),
                    timeout=timeout,
                )

                print(
                    f"[AUTOGEN TRACE] scenario={scenario.scenario_id} "
                    f"n_events={len(collector.events)} "
                    f"senders={dict(Counter(e.get('sender') for e in collector.events))}",
                    flush=True,
                )

                encoder  = EmbeddingEncoder()
                agent_set = set(topology.agents)
                edge_set  = set(topology.edges)

                def _map(raw: Any) -> str | None:
                    s = str(raw or "").strip()
                    # ACI AutoGen UserProxyAgent logs as "user" — map to "user_proxy"
                    if s == "user" and "user_proxy" in agent_set:
                        return "user_proxy"
                    if s in agent_set:
                        return s
                    sl = s.lower()
                    for a in agent_set:
                        if a.lower() == sl:
                            return a
                    return None

                # Group by (turn, stage). Stage is set by set_stage calls.
                # bootstrap → all bootstrap messages together
                # step_N    → all step N messages together
                # conclude  → all conclude messages together
                def _stage_order(key: tuple) -> tuple:
                    turn, stage = key
                    order = {"bootstrap": 0, "run": 0, "step": 1,
                             "conclude": 2, "unknown": 3}
                    return (int(turn), order.get(str(stage), 9), str(stage))

                grouped: dict[tuple, list] = defaultdict(list)
                for ev in collector.events:
                    key = (int(ev.get("turn", 0) or 0),
                           str(ev.get("stage", "unknown")))
                    grouped[key].append(ev)

                dropped = []
                converted = 0

                for caspian_idx, key in enumerate(
                    sorted(grouped.keys(), key=_stage_order), 1
                ):
                    batch: list[ChannelEvent] = []
                    for ev in grouped[key]:
                        src = _map(ev.get("sender") or ev.get("source"))
                        tgt = _map(ev.get("receiver") or ev.get("target"))
                        if not src or not tgt or (src, tgt) not in edge_set:
                            dropped.append({
                                "raw_sender":   ev.get("sender") or ev.get("source"),
                                "raw_receiver": ev.get("receiver") or ev.get("target"),
                                "mapped_src":   src,
                                "mapped_tgt":   tgt,
                                "agents":       sorted(agent_set),
                            })
                            continue

                        msg     = str(ev.get("message") or ev.get("content") or "")
                        tok     = max(1, len(msg.split()))
                        content = f"[{key[1]};t={key[0]}] {msg}"

                        vec_comm = encoder.comm_vector(content, tok)
                        batch.append(ChannelEvent(
                            turn=caspian_idx, source=src, target=tgt,
                            channel="comm",
                            payload={"kind": "aci_autogen", "content": content,
                                     "stage": key[1], "aci_turn": key[0],
                                     "u_vector": vec_comm.tolist(),
                                     "v_vector": vec_comm.tolist()},
                            vector=vec_comm,
                            timestamp=float(ev.get("ts", _time.time())),
                        ))

                        vec_exec = EmbeddingEncoder.exec_vector(
                            prompt_tokens=tok, completion_tokens=tok, latency_ms=0.0
                        )
                        batch.append(ChannelEvent(
                            turn=caspian_idx, source=src, target=tgt,
                            channel="exec",
                            payload={"kind": "aci_autogen_exec", "token_count": tok,
                                     "stage": key[1], "aci_turn": key[0],
                                     "u_vector": vec_exec.tolist(),
                                     "v_vector": vec_exec.tolist()},
                            vector=vec_exec,
                            timestamp=float(ev.get("ts", _time.time())),
                        ))

                    if batch:
                        converted += len(batch)
                        yield batch

                if collector.events and converted == 0:
                    print(
                        f"[AUTOGEN DROP] all {len(collector.events)} events dropped. "
                        f"sample={dropped[:3]}",
                        flush=True,
                    )
                return
        except ImportError:
            pass

        # ── Standard autogen-agentchat path (TAMAS) ──────────────────────
        if not _AUTOGEN_AVAILABLE:
            raise RuntimeError("autogen-agentchat is not installed.")

        team  = build_team(scenario.task_data, model_name=model, config=config)
        state = _AutoGenStreamState(topology)
        query = "Task: " + scenario.user_query()
        turn  = 0

        async def _stream():
            nonlocal turn
            async for message in team.run_stream(task=query):
                events = state.process(message, turn)
                if events:
                    turn += 1
                    yield events

        try:
            async with asyncio.timeout(timeout):
                async for event_list in _stream():
                    yield event_list
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return
        except Exception as e:
            if "task_done" not in str(e) and "CancelledError" not in str(e):
                raise
        finally:
            try:
                await team.reset()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Per-scenario stream state (TAMAS autogen-agentchat path)
# ---------------------------------------------------------------------------

class _AutoGenStreamState:

    def __init__(self, topology: MASTopology) -> None:
        self._topology       = topology
        self._agent_set      = set(topology.agents)
        self._encoder        = EmbeddingEncoder()
        self._last_ts:       dict[str, datetime]        = {}
        self._pending_calls: dict[str, tuple[str, str]] = {}

    def process(self, message: Any, turn: int) -> list[ChannelEvent]:
        if isinstance(message, TaskResult):
            return []
        if isinstance(message, (SelectSpeakerEvent, ThoughtEvent)):
            return []

        source = getattr(message, "source", None)
        if source is None or source not in self._agent_set:
            return []

        ts      = _as_datetime(getattr(message, "created_at", None))
        latency = self._estimate_latency(source, ts)
        usage   = getattr(message, "models_usage", None)
        p_tok   = usage.prompt_tokens     if usage else 0
        c_tok   = usage.completion_tokens if usage else 0
        events: list[ChannelEvent] = []
        targets = self._infer_targets(source)

        if isinstance(message, TextMessage):
            vec = self._encoder.comm_vector(message.content, p_tok + c_tok)
            for t in targets:
                events.append(self._comm_event(turn, source, t, message.content,
                                               p_tok + c_tok, vec, ts))
            if usage:
                for t in targets:
                    events.extend(self._exec_events(turn, source, t,
                                                    p_tok, c_tok, latency, ts))

        elif isinstance(message, HandoffMessage):
            tgts = self._infer_targets(source, explicit=message.target)
            vec  = self._encoder.comm_vector(message.content, p_tok + c_tok,
                                             is_handoff=True)
            for t in tgts:
                events.append(ChannelEvent(
                    turn=turn, source=source, target=t, channel="comm",
                    payload={"content": message.content,
                             "token_count": p_tok + c_tok, "handoff": True,
                             "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                    vector=vec, timestamp=ts.timestamp(),
                ))

        elif isinstance(message, ToolCallRequestEvent):
            for call in message.content:
                self._pending_calls[call.id] = (call.name, call.arguments)
                vec = self._encoder.tool_vector(call.name, call.arguments, "")
                for t in targets:
                    events.append(ChannelEvent(
                        turn=turn, source=source, target=t, channel="tool",
                        payload={"tool_name": call.name, "call_id": call.id,
                                 "arguments": call.arguments, "output": "",
                                 "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                        vector=vec, timestamp=ts.timestamp(),
                    ))

        elif isinstance(message, ToolCallExecutionEvent):
            for result in message.content:
                call_id = getattr(result, "call_id", None)
                if call_id and call_id in self._pending_calls:
                    tool_name, arguments = self._pending_calls.pop(call_id)
                else:
                    tool_name, arguments = "unknown", ""
                output = getattr(result, "data", str(result))
                vec    = self._encoder.tool_vector(tool_name, arguments, output)
                for t in targets:
                    events.append(ChannelEvent(
                        turn=turn, source=source, target=t, channel="tool",
                        payload={"tool_name": tool_name, "call_id": call_id or "",
                                 "arguments": arguments, "output": output,
                                 "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                        vector=vec, timestamp=ts.timestamp(),
                    ))

        elif isinstance(message, ToolCallSummaryMessage):
            vec = self._encoder.comm_vector(message.content, p_tok + c_tok)
            for t in targets:
                events.append(self._comm_event(turn, source, t, message.content,
                                               p_tok + c_tok, vec, ts))
            if usage:
                for t in targets:
                    events.extend(self._exec_events(turn, source, t,
                                                    p_tok, c_tok, latency, ts))

        elif isinstance(message, MemoryQueryEvent):
            for mem_item in message.content:
                cs  = str(getattr(mem_item, "content", mem_item))
                vec = self._encoder.mem_vector(cs, "read")
                for t in targets:
                    events.append(ChannelEvent(
                        turn=turn, source=source, target=t, channel="mem",
                        payload={"content": cs, "op": "read",
                                 "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                        vector=vec, timestamp=ts.timestamp(),
                    ))

        self._last_ts[source] = ts
        return events

    def _infer_targets(self, source: str, explicit: str | None = None) -> list[str]:
        if explicit is not None:
            if explicit != source and self._topology.has_edge(source, explicit):
                return [explicit]
            return []
        return [tgt for src, tgt in self._topology.edges
                if src == source and tgt != source]

    def _estimate_latency(self, source: str, ts: datetime) -> float:
        prev = self._last_ts.get(source)
        if prev is None:
            return 0.0
        return max(0.0, (ts - prev).total_seconds() * 1000.0)

    def _comm_event(self, turn, source, target, content, token_count, vec, ts):
        return ChannelEvent(
            turn=turn, source=source, target=target, channel="comm",
            payload={"content": content, "token_count": token_count,
                     "u_vector": vec.tolist(), "v_vector": vec.tolist()},
            vector=vec, timestamp=ts.timestamp(),
        )

    def _exec_events(self, turn, source, target, p_tok, c_tok, latency_ms, ts):
        vec = EmbeddingEncoder.exec_vector(p_tok, c_tok, latency_ms)
        return [ChannelEvent(
            turn=turn, source=source, target=target, channel="exec",
            payload={"prompt_tokens": p_tok, "completion_tokens": c_tok,
                     "latency_ms": latency_ms,
                     "u_vector": vec.tolist(), "v_vector": vec.tolist()},
            vector=vec, timestamp=ts.timestamp(),
        )]