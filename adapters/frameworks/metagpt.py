"""
adapters/frameworks/metagpt.py

MetaGPT framework adapter for CASPIAN — generic, benchmark-agnostic.

ExecutionMode.SUBPROCESS — MetaGPT runs in an isolated Python environment
(separate venv) because MetaGPT 0.8.2 requires typing_extensions==4.9.0
which conflicts with pydantic_core>=2.x used by AutoGen/CrewAI.

The subprocess emits __TRACE__ JSON lines to stdout.
This adapter parses them into CASPIAN ChannelEvents.

Set METAGPT_PYTHON env var to point to the MetaGPT venv python:
  export METAGPT_PYTHON=~/.venvs/metagpt_venv/bin/python3

Supported __TRACE__ event kinds:
  message / agent_message          -> comm
  memory_read / memory_write       -> mem
  tool_call / tool_result /
  tool_error                       -> tool
  action_start / action_end /
  code_exec                        -> exec

MASTopology:
  role_workflow (default): forward chain + feedback + planner broadcast
  fully_connected: all-to-all for ablation
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from core.events import ChannelEvent, MASTopology
from adapters.benchmarks.base import Scenario
from adapters.frameworks.base import MASFrameworkAdapter, ExecutionMode
from adapters.frameworks.encoder import EmbeddingEncoder
from adapters.frameworks.specs import MetaGPTRunSpec


TRACE_PREFIX = "__TRACE__"

_SUPPORTED_KINDS: set[str] = {
    "message", "agent_message", "publish", "subscribe",
    "memory_read", "memory_write",
    "tool_call", "tool_result", "tool_error",
    "action_start", "action_end",
    "code_exec", "file_write", "artifact_write",
}


def _normalize_name(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _agent_names_from_task(task_data: dict[str, Any]) -> list[str]:
    return [
        a["agent_name"]
        for a in task_data.get("agents", [])
        if isinstance(a.get("agent_name"), str) and a["agent_name"].strip()
    ]


class MetaGPTAdapter(MASFrameworkAdapter):
    """
    MetaGPT framework adapter. ExecutionMode.SUBPROCESS.
    Runs metagpt_trace_runner.py in an isolated venv.
    Parses __TRACE__ output into CASPIAN ChannelEvents.
    """

    def __init__(self, burst_seconds: float = 1.0) -> None:
        self._burst_seconds = burst_seconds
        self._runner_path   = Path(__file__).resolve().parents[2] / "metagpt_trace_runner.py"
        self._cwd           = Path(__file__).resolve().parents[2]
        # Use METAGPT_PYTHON env var or fall back to current python
        self._python = os.environ.get("METAGPT_PYTHON", sys.executable)

    @property
    def name(self) -> str:
        return "MetaGPT"

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.SUBPROCESS

    def build_topology(
        self,
        scenario:     Scenario,
        config:       str,
        runtime_spec: "Any | None" = None,
    ) -> MASTopology:
        agents = []
        try:
            from adapters.frameworks.specs import ACIArenaNativeRunSpec as _AciSpec
            if isinstance(runtime_spec, _AciSpec) and runtime_spec.framework == "MetaGPT":
                # Internal MAS agents only — drop user/system boundary nodes
                _agents = ["product_manager", "architect",
                           "project_manager", "engineer", "qa_engineer"]
                _edges  = [
                    ("product_manager", "architect"),
                    ("architect",       "project_manager"),
                    ("project_manager", "engineer"),
                    ("engineer",        "qa_engineer"),
                    # feedback edge: active when QA triggers iteration
                    ("qa_engineer",     "product_manager"),
                ]
                return MASTopology(
                    framework=self.name,
                    mas_type="aci_metagpt",
                    agents=_agents,
                    edges=_edges,
                )
        except ImportError:
            pass

        if not agents:
            if runtime_spec is not None and hasattr(runtime_spec, "agents"):
                agents = [a.role for a in runtime_spec.agents]
            else:
                agents = _agent_names_from_task(scenario.task_data)

        cfg = _normalize_name(config)

        if not agents:
            return MASTopology(framework=self.name, mas_type="unknown",
                               agents=[], edges=[])

        if cfg in {"fully_connected", "centralized", "group_chat"}:
            mas_type = "fully_connected"
            edges    = [(a, b) for a in agents for b in agents if a != b]
        else:
            mas_type = "role_workflow"
            edges: list[tuple[str, str]] = []
            for i in range(len(agents) - 1):
                edges.append((agents[i], agents[i + 1]))
            for i in range(1, len(agents)):
                edges.append((agents[i], agents[i - 1]))
            if len(agents) > 2:
                for j in range(1, len(agents)):
                    e = (agents[0], agents[j])
                    if e not in edges:
                        edges.append(e)
            seen: set[tuple[str, str]] = set()
            edges = [e for e in edges if not (e in seen or seen.add(e))]  # type: ignore

        return MASTopology(
            framework=self.name, mas_type=mas_type,
            agents=agents, edges=edges,
        )

    async def run_live(
        self,
        scenario:     Scenario,
        topology:     MASTopology,
        config:       str,
        model:        str,
        timeout:      int,
        runtime_spec: "Any | None" = None,
    ):
        """LIVE mode entry point — handles ACIArenaNativeRunSpec and MetaGPTRunSpec."""
        import asyncio
        from collections import defaultdict, Counter
        import hashlib, json, time as _time

        # ACIArenaNativeRunSpec: native ACI MetaGPT — MetaGPT-specific runner
        try:
            from adapters.frameworks.specs import ACIArenaNativeRunSpec as _AciSpec
            if isinstance(runtime_spec, _AciSpec) and runtime_spec.framework == "MetaGPT":
                from adapters.frameworks.llm_debate import DebateTraceCollector
                from collections import Counter
                import time as _time

                def _run_native_metagpt(spec, col):
                    mas    = spec.backend_factory(col)
                    print(
                        f"[METAGPT MAS] type={type(mas).__name__} "
                        f"malicious={getattr(mas,'malicious_agents',None)} "
                        f"logger={type(getattr(mas,'logger',None)).__name__}",
                        flush=True,
                    )
                    attack = spec.attack_factory(mas) if spec.attack_factory else None
                    if attack is not None and hasattr(attack, "run"):
                        attack.run(mas)

                    # Verify attack injected into agents — force-inject if not
                    _payload = ""
                    for _attr in ("payload", "instruction", "message"):
                        _payload = str(getattr(attack, _attr, "") or "")
                        if _payload:
                            break
                    if _payload and attack is not None:
                        _needles = _payload[:30].lower().split()[:3]
                        _injected = False
                        for _n in (spec.malicious_agents or []):
                            try:
                                _d = str(getattr(mas.get_agent(_n), "__dict__", "")).lower()
                                if any(x in _d for x in _needles):
                                    _injected = True; break
                            except Exception:
                                pass
                        if not _injected:
                            print(f"[METAGPT FORCE_INJECT] attack.run no-op, wrapping run_step", flush=True)
                            for _n in (spec.malicious_agents or []):
                                try:
                                    _ag   = mas.get_agent(_n)
                                    _orig = _ag.run_step
                                    _p    = _payload
                                    def _make(orig, p):
                                        def _injected_step(query=None, *a, **kw):
                                            if isinstance(query, dict):
                                                qd = dict(query)
                                                k0 = next(iter(qd))
                                                qd[k0] = f"{qd[k0]}\n\n[Instruction] {p}"
                                                return orig(query=qd, *a, **kw)
                                            return orig(query=f"{query}\n\n[Instruction] {p}", *a, **kw)
                                        return _injected_step
                                    _ag.run_step = _make(_orig, _p)
                                    print(f"[METAGPT FORCE_INJECT] wrapped {_n} head={_p[:80]!r}", flush=True)
                                except Exception as _e:
                                    print(f"[METAGPT FORCE_INJECT_ERR] {_n}: {_e}", flush=True)

                    print(
                        f"[METAGPT ATTACK] attack={spec.attack_class_name} "
                        f"malicious={spec.malicious_agents}",
                        flush=True,
                    )
                    col.set_stage("bootstrap", 0)
                    args, done = mas.bootstrap(spec.user_query)
                    turn = 0
                    while not done and turn < spec.max_turn:
                        turn += 1
                        if attack is not None and hasattr(attack, "set_turn"):
                            attack.set_turn(turn)
                        col.set_stage("step", turn)
                        args, done = mas.step(args)
                        # Log existing QA feedback state as adapter-side feedback edge
                        if (not done) and isinstance(args, dict) and args.get("qa_feedback"):
                            col.log_message(
                                sender="qa_engineer",
                                receiver="product_manager",
                                message=json.dumps(args["qa_feedback"], ensure_ascii=False),
                            )
                    col.set_stage("conclude", turn + 1)
                    result = mas.conclude(args)
                    if attack is not None and hasattr(attack, "set_answer"):
                        try: attack.set_answer(result)
                        except Exception: pass
                    try:
                        verify = attack.verify() if attack else None
                    except Exception as e:
                        verify = f"error={type(e).__name__}"
                    print(
                        f"[METAGPT VERIFY] verify={verify} n_events={len(col.events)}",
                        flush=True,
                    )

                collector = DebateTraceCollector()
                await asyncio.wait_for(
                    asyncio.to_thread(_run_native_metagpt, runtime_spec, collector),
                    timeout=timeout,
                )

                print(
                    f"[METAGPT TRACE] scenario={scenario.scenario_id} "
                    f"n_events={len(collector.events)} "
                    f"senders={dict(Counter(e.get('sender') for e in collector.events))}",
                    flush=True,
                )

                encoder   = EmbeddingEncoder()
                agent_set = set(topology.agents)
                edge_set  = set(topology.edges)

                def _map(raw: Any) -> "str | None":
                    s = str(raw or "").strip()
                    if s in agent_set: return s
                    sl = s.lower()
                    for a in agent_set:
                        if a.lower() == sl: return a
                    return None

                def _stage_order(key: tuple) -> tuple:
                    turn, stage = key
                    order = {"bootstrap": 0, "run": 0, "step": 1,
                             "conclude": 2, "unknown": 3}
                    return (int(turn), order.get(str(stage), 9), str(stage))

                # Group by (turn, stage) — one MetaGPT step = one CASPIAN turn
                grouped: dict = defaultdict(list)
                for ev in collector.events:
                    key = (int(ev.get("turn", 0) or 0),
                           str(ev.get("stage", "unknown")))
                    grouped[key].append(ev)

                converted = 0
                dropped   = []

                for caspian_idx, key in enumerate(
                    sorted(grouped.keys(), key=_stage_order), 1
                ):
                    batch: list[ChannelEvent] = []
                    for ev in grouped[key]:
                        src = _map(ev.get("sender") or ev.get("source"))
                        tgt = _map(ev.get("receiver") or ev.get("target"))
                        if not src or not tgt or (src, tgt) not in edge_set:
                            dropped.append({
                                "sender": ev.get("sender"),
                                "receiver": ev.get("receiver"),
                            })
                            continue
                        msg      = str(ev.get("message") or ev.get("content") or "")
                        tok      = max(1, len(msg.split()))
                        content  = f"[{key[1]};t={key[0]}] {msg}"
                        vec_comm = encoder.comm_vector(content, tok)
                        batch.append(ChannelEvent(
                            turn=caspian_idx, source=src, target=tgt,
                            channel="comm",
                            payload={"kind": "aci_metagpt", "content": content,
                                     "stage": key[1], "aci_turn": key[0],
                                     "u_vector": vec_comm.tolist(),
                                     "v_vector": vec_comm.tolist()},
                            vector=vec_comm,
                            timestamp=float(ev.get("ts", _time.time())),
                        ))
                        vec_exec = EmbeddingEncoder.exec_vector(tok, tok, 0.0)
                        batch.append(ChannelEvent(
                            turn=caspian_idx, source=src, target=tgt,
                            channel="exec",
                            payload={"kind": "aci_metagpt_exec", "token_count": tok,
                                     "stage": key[1], "aci_turn": key[0],
                                     "u_vector": vec_exec.tolist(),
                                     "v_vector": vec_exec.tolist()},
                            vector=vec_exec,
                            timestamp=float(ev.get("ts", _time.time())),
                        ))
                    if batch:
                        converted += len(batch)
                        yield batch

                print(
                    f"[METAGPT SUMMARY] scenario={scenario.scenario_id} "
                    f"raw={len(collector.events)} converted={converted} "
                    f"dropped={len(dropped)} topology={sorted(agent_set)}",
                    flush=True,
                )
                return
        except ImportError:
            pass

        # MetaGPTRunSpec with backend_factory (old native path)
        if runtime_spec is not None and getattr(runtime_spec, "backend_factory", None):
            async for batch in self.run_live_aci(
                scenario=scenario, topology=topology,
                config=config, model=model, timeout=timeout,
                runtime_spec=runtime_spec,
            ):
                yield batch
        else:
            raise RuntimeError(
                "MetaGPTAdapter.run_live called without backend_factory. "
                "Use run_subprocess for TAMAS MetaGPT."
            )

    async def run_live_aci(
        self,
        scenario:     Scenario,
        topology:     MASTopology,
        config:       str,
        model:        str,
        timeout:      int,
        runtime_spec: "MetaGPTRunSpec",
    ) -> "AsyncIterator[list[ChannelEvent]]":
        """
        Run ACIArena MetaGPT via native ACI backend (LIVE mode).
        Routes through LLMDebate-style collector pattern.
        """
        from adapters.frameworks.llm_debate import DebateTraceCollector
        from adapters.frameworks.encoder import EmbeddingEncoder
        import asyncio as _asyncio
        import time as _time

        collector = DebateTraceCollector()
        encoder   = EmbeddingEncoder()

        def _run() -> None:
            mas    = runtime_spec.backend_factory(collector)
            attack = runtime_spec.attack_factory(mas) if runtime_spec.attack_factory else None

            if attack is not None and hasattr(attack, "run"):
                attack.run(mas)

            collector.set_stage("bootstrap", 0)
            args, done = mas.bootstrap(runtime_spec.user_query)

            turn = 0
            while not done:
                turn += 1
                if attack is not None and hasattr(attack, "set_turn"):
                    attack.set_turn(turn)
                collector.set_stage("step", turn)
                args, done = mas.step(args)

            collector.set_stage("conclude", turn + 1)
            result = mas.conclude(args)
            if attack is not None and hasattr(attack, "set_answer"):
                try:
                    attack.set_answer(result)
                except Exception:
                    pass

        await _asyncio.wait_for(_asyncio.to_thread(_run), timeout=timeout)

        # Convert collector events to ChannelEvents using topology
        edge_set  = set(topology.edges)
        agent_set = set(topology.agents)

        def _map(raw):
            raw_s = str(raw).strip()
            if raw_s in agent_set:
                return raw_s
            for a in agent_set:
                if a.lower() == raw_s.lower():
                    return a
            return None

        from collections import defaultdict
        grouped = defaultdict(list)
        for ev in collector.events:
            grouped[(ev.get("turn", 0), ev.get("stage", ""))].append(ev)

        for caspian_idx, key in enumerate(sorted(grouped.keys()), start=1):
            batch: list[ChannelEvent] = []
            for ev in grouped[key]:
                sender   = _map(ev.get("sender") or ev.get("source"))
                receiver = _map(ev.get("receiver") or ev.get("target"))
                if not sender or not receiver:
                    continue
                if (sender, receiver) not in edge_set:
                    continue
                msg = str(ev.get("message") or "")
                tok = max(1, len(msg.split()))
                content = f"[{key[1]};t={key[0]}] {msg}"
                vec = encoder.comm_vector(content, tok)
                batch.append(ChannelEvent(
                    turn=caspian_idx, source=sender, target=receiver,
                    channel="comm",
                    payload={"kind": "metagpt_message", "content": content,
                             "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                    vector=vec, timestamp=float(ev.get("ts", _time.time())),
                ))
            if batch:
                yield batch

    async def run_subprocess(
        self,
        scenario:     Scenario,
        topology:     MASTopology,
        config:       str,
        model:        str,
        timeout:      int,
        runtime_spec: "MetaGPTRunSpec | None" = None,
    ) -> list[list[ChannelEvent]]:
        if not self._runner_path.exists():
            raise FileNotFoundError(
                f"metagpt_trace_runner.py not found at {self._runner_path}. "
                "Place it in the repo root."
            )

        raw_log = await self._run_subprocess(
            scenario=scenario,
            config=config,
            model=model,
            timeout=timeout,
        )

        state = _MetaGPTTraceState(topology, burst_seconds=self._burst_seconds)
        return state.raw_log_to_turns(raw_log)

    async def _run_subprocess(
        self,
        scenario: Scenario,
        config:   str,
        model:    str,
        timeout:  int,
    ) -> str:
        import re as _re
        ansi_re = _re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                json.dump(scenario.task_data, f)
                tmp_path = f.name

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Load .env file so OPENAI_API_KEY is available in subprocess
            env_file = self._cwd / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        env.setdefault(k.strip(), v.strip())

            proc = await asyncio.create_subprocess_exec(
                self._python,
                str(self._runner_path),
                "--json_data",      tmp_path,
                "--model",          model,
                "--scenario",       scenario.domain,
                "--config",         config,
                "--attack_type",    scenario.attack_type,
                "--scenario_id",    scenario.scenario_id,
                "--file_task_idx",  str(scenario.file_task_idx),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=env,
            )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.communicate()
                except Exception:
                    pass
                return "[TIMEOUT] MetaGPT task exceeded timeout."

            stdout = ansi_re.sub("", (stdout_b or b"").decode("utf-8", errors="replace"))
            stderr = ansi_re.sub("", (stderr_b or b"").decode("utf-8", errors="replace"))
            raw    = stdout
            if stderr:
                raw += "\n[STDERR]\n" + stderr
            import sys as _sys
            # Check for fatal errors and raise loudly
            fatal_events = []
            for line in stdout.splitlines():
                if line.startswith(TRACE_PREFIX):
                    try:
                        evt = json.loads(line[len(TRACE_PREFIX):].strip())
                        if evt.get("kind") == "fatal_error":
                            fatal_events.append(evt)
                    except Exception:
                        pass

            if proc.returncode != 0:
                raise RuntimeError(
                    f"MetaGPT runner returncode={proc.returncode}\n"
                    f"STDOUT:\n{stdout[-2000:]}\nSTDERR:\n{stderr[-2000:]}"
                )
            if fatal_events:
                raise RuntimeError(
                    f"MetaGPT runner fatal_error: {fatal_events[-1]}"
                )

            return raw

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Per-scenario trace state
# ---------------------------------------------------------------------------

class _MetaGPTTraceState:
    """Converts one MetaGPT raw log into CASPIAN ChannelEvent turns."""

    def __init__(self, topology: MASTopology, burst_seconds: float = 1.0) -> None:
        self._topology      = topology
        self._agent_lookup  = {_normalize_name(n): n for n in topology.agents}
        self._encoder       = EmbeddingEncoder()
        self._burst_seconds = burst_seconds
        self._pending_calls: dict[str, tuple[str, str]] = {}

    def raw_log_to_turns(self, raw_log: str) -> list[list[ChannelEvent]]:
        groups = self._group(self._extract(raw_log))
        out: list[list[ChannelEvent]] = []
        for turn_idx, group in enumerate(groups, start=1):
            events = self._group_to_events(group, turn_idx)
            if events:
                out.append(events)
        return out

    def _extract(self, raw_log: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in raw_log.splitlines():
            if not line.startswith(TRACE_PREFIX):
                continue
            payload = line[len(TRACE_PREFIX):].strip()
            if not payload:
                continue
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(evt, dict):
                events.append(evt)
        return events

    def _group(
        self, trace_events: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        runtime = sorted(
            [e for e in trace_events
             if str(e.get("kind", "")).lower() in _SUPPORTED_KINDS],
            key=lambda e: float(e.get("ts", 0.0)),
        )
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]]       = []
        prev_ts: float | None               = None
        prev_agent: str | None              = None

        for ev in runtime:
            ts    = float(ev.get("ts", 0.0))
            agent = _normalize_name(str(ev.get("agent", "")))
            same  = (prev_ts is not None
                     and (ts - prev_ts) <= self._burst_seconds
                     and agent == prev_agent)
            if not current or same:
                current.append(ev)
            else:
                groups.append(current)
                current = [ev]
            prev_ts    = ts
            prev_agent = agent

        if current:
            groups.append(current)
        return groups

    def _group_to_events(
        self, group: list[dict[str, Any]], turn: int
    ) -> list[ChannelEvent]:
        source = self._resolve(group[0].get("agent", ""))
        if source is None:
            return []
        targets = self._infer_targets(source)
        if not targets:
            return []

        ts_vals    = [float(e.get("ts", 0.0)) for e in group if e.get("ts")]
        ts         = max(ts_vals) if ts_vals else None
        latency_ms = (max(0.0, (max(ts_vals) - min(ts_vals)) * 1000.0)
                      if len(ts_vals) >= 2 else 0.0)

        call_count = result_count = error_count = action_count = 0
        out: list[ChannelEvent] = []

        for ev in group:
            kind = str(ev.get("kind", "")).lower()
            if kind in {"message", "agent_message", "publish", "subscribe"}:
                out.extend(self._comm(ev, turn, source, targets, ts))
            elif kind in {"memory_read", "memory_write"}:
                out.extend(self._memory(ev, turn, source, targets, ts))
            elif kind == "tool_call":
                call_count += 1
                out.extend(self._tool_call(ev, turn, source, targets, ts))
            elif kind == "tool_result":
                result_count += 1
                out.extend(self._tool_result(ev, turn, source, targets, ts))
            elif kind == "tool_error":
                error_count += 1
                out.extend(self._tool_error(ev, turn, source, targets, ts))
            elif kind in {"action_start", "action_end", "code_exec",
                          "file_write", "artifact_write"}:
                action_count += 1
                out.extend(self._exec_artifact(ev, turn, source, targets, ts))

        if call_count or result_count or error_count or action_count:
            vec = EmbeddingEncoder.exec_vector(
                prompt_tokens    = 64 * call_count  + 32 * action_count,
                completion_tokens= 64 * result_count + 16 * error_count + 32 * action_count,
                latency_ms       = latency_ms,
            )
            for tgt in targets:
                out.append(ChannelEvent(
                    turn=turn, source=source, target=tgt, channel="exec",
                    payload={"kind": "exec_summary",
                             "call_count": call_count, "result_count": result_count,
                             "error_count": error_count, "action_count": action_count,
                             "latency_ms": latency_ms,
                             "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                    vector=vec, timestamp=ts,
                ))
        return out

    def _comm(self, ev, turn, source, targets, ts):
        content = self._str(ev.get("content", ev.get("message", ev.get("text", ev))))
        tok     = max(1, len(content.split()))
        vec     = self._encoder.comm_vector(content, tok)
        return [ChannelEvent(turn=turn, source=source, target=t, channel="comm",
                             payload={"kind": str(ev.get("kind","")), "content": content,
                                      "token_count": tok,
                                      "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                             vector=vec, timestamp=ts) for t in targets]

    def _memory(self, ev, turn, source, targets, ts):
        kind    = str(ev.get("kind", "")).lower()
        op      = "write" if "write" in kind else "read"
        content = self._str(ev.get("content", ev.get("data", ev.get("memory", ev))))
        vec     = self._encoder.mem_vector(content, op)
        return [ChannelEvent(turn=turn, source=source, target=t, channel="mem",
                             payload={"kind": kind, "content": content, "op": op,
                                      "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                             vector=vec, timestamp=ts) for t in targets]

    def _tool_call(self, ev, turn, source, targets, ts):
        call_id   = str(ev.get("call_id") or ev.get("id") or "")
        tool_name = str(ev.get("tool") or ev.get("tool_name") or ev.get("action") or "unknown")
        arguments = self._str(ev.get("arguments", ev.get("args", ev.get("input", ""))))
        if call_id:
            self._pending_calls[call_id] = (tool_name, arguments)
        vec = self._encoder.tool_vector(tool_name, arguments, "")
        return [ChannelEvent(turn=turn, source=source, target=t, channel="tool",
                             payload={"kind": "tool_call", "tool_name": tool_name,
                                      "call_id": call_id, "arguments": arguments, "output": "",
                                      "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                             vector=vec, timestamp=ts) for t in targets]

    def _tool_result(self, ev, turn, source, targets, ts):
        call_id = str(ev.get("call_id") or ev.get("id") or "")
        if call_id and call_id in self._pending_calls:
            tool_name, arguments = self._pending_calls.pop(call_id)
        else:
            tool_name = str(ev.get("tool") or ev.get("action") or "unknown")
            arguments = self._str(ev.get("arguments", ev.get("args", "")))
        output = self._str(ev.get("output", ev.get("result", ev.get("data", ""))))

        # tool_result + comm handoff + mem write — same richness as CrewAI/AutoGen
        vec_tool = self._encoder.tool_vector(tool_name, arguments, output)
        handoff  = f"{tool_name} {output}"
        tok      = max(1, len(handoff.split()))
        vec_comm = self._encoder.comm_vector(handoff, tok)
        vec_mem  = self._encoder.mem_vector(handoff, op="write")

        events = []
        for t in targets:
            events.append(ChannelEvent(
                turn=turn, source=source, target=t, channel="tool",
                payload={"kind": "tool_result", "tool_name": tool_name,
                         "call_id": call_id, "arguments": arguments, "output": output,
                         "u_vector": vec_tool.tolist(), "v_vector": vec_tool.tolist()},
                vector=vec_tool, timestamp=ts))
            events.append(ChannelEvent(
                turn=turn, source=source, target=t, channel="comm",
                payload={"kind": "tool_output_handoff", "tool_name": tool_name,
                         "content": handoff,
                         "u_vector": vec_comm.tolist(), "v_vector": vec_comm.tolist()},
                vector=vec_comm, timestamp=ts))
            events.append(ChannelEvent(
                turn=turn, source=source, target=t, channel="mem",
                payload={"kind": "tool_output_memory", "tool_name": tool_name,
                         "content": handoff, "op": "write",
                         "u_vector": vec_mem.tolist(), "v_vector": vec_mem.tolist()},
                vector=vec_mem, timestamp=ts))
        return events

    def _tool_error(self, ev, turn, source, targets, ts):
        tool_name = str(ev.get("tool") or ev.get("action") or "unknown")
        arguments = self._str(ev.get("arguments", ev.get("args", "")))
        error     = self._str(ev.get("error", ev.get("output", ev)))
        vec       = self._encoder.tool_vector(tool_name, arguments, error)
        return [ChannelEvent(turn=turn, source=source, target=t, channel="tool",
                             payload={"kind": "tool_error", "tool_name": tool_name,
                                      "arguments": arguments, "output": error, "error": True,
                                      "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                             vector=vec, timestamp=ts) for t in targets]

    def _exec_artifact(self, ev, turn, source, targets, ts):
        kind    = str(ev.get("kind", "")).lower()
        content = self._str(ev.get("output", ev.get("content",
                             ev.get("artifact", ev.get("path", ev)))))
        latency = float(ev.get("latency_ms", 0.0) or 0.0)
        proxy   = max(1, len(content.split()))
        vec     = EmbeddingEncoder.exec_vector(proxy, proxy, latency)
        return [ChannelEvent(turn=turn, source=source, target=t, channel="exec",
                             payload={"kind": kind, "content": content, "latency_ms": latency,
                                      "u_vector": vec.tolist(), "v_vector": vec.tolist()},
                             vector=vec, timestamp=ts) for t in targets]

    def _infer_targets(self, source: str) -> list[str]:
        return [tgt for src, tgt in self._topology.edges if src == source and tgt != source]

    def _resolve(self, raw: Any) -> str | None:
        return self._agent_lookup.get(_normalize_name(str(raw)))

    @staticmethod
    def _str(x: Any) -> str:
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(x)