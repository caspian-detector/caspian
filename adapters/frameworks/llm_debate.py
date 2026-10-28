"""
adapters/frameworks/llm_debate.py

LLM Debate framework adapter for CASPIAN — generic, benchmark-agnostic.

ExecutionMode.LIVE — runs debate in-process via asyncio.to_thread.

Key design:
  - One CASPIAN turn = one debate phase/round (bootstrap, step_1..N, conclude)
  - Messages within a phase are batched together, not one-per-turn
  - Stage metadata included in content so different rounds produce distinct vectors
  - ACI backend runs real LLM calls via backend_factory
  - TAMAS uses stub debate with real OpenAI calls per round

Channel mapping:
  comm   debate messages between agents
  exec   token/latency proxy per message
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter, defaultdict
from typing import Any, AsyncIterator

from core.events import ChannelEvent, MASTopology
from adapters.benchmarks.base import Scenario
from adapters.frameworks.base import MASFrameworkAdapter, ExecutionMode
from adapters.frameworks.encoder import EmbeddingEncoder
from adapters.frameworks.specs import LLMDebateRunSpec


def _normalize(x: str) -> str:
    return str(x).strip()


class DebateTraceCollector:
    """
    Logger compatible with ACI-style logger.log_message(sender, receiver, message).
    Tracks debate phase/round so messages can be grouped into CASPIAN turns.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.stage:  str = "unknown"
        self.turn:   int = 0

    def set_stage(self, stage: str, turn: int) -> None:
        self.stage = str(stage)
        self.turn  = int(turn)

    def log_message(self, sender: str, receiver: str, message: str) -> None:
        self.events.append({
            "ts":       time.time(),
            "kind":     "message",
            "stage":    self.stage,
            "turn":     self.turn,
            "sender":   str(sender),
            "receiver": str(receiver),
            "message":  str(message),
        })

    def __getattr__(self, name: str) -> Any:
        """Catch any log_* method ACI backends may call."""
        if name.startswith("log"):
            def _sink(*args: Any, **kwargs: Any) -> None:
                msg = " ".join(str(a) for a in args)
                sender = str(kwargs.get("sender", kwargs.get("source", "system")))
                receiver = str(kwargs.get("receiver", kwargs.get("target", "system")))
                self.events.append({
                    "ts": time.time(), "kind": name,
                    "stage": self.stage, "turn": self.turn,
                    "sender": sender, "receiver": receiver,
                    "message": msg or str(kwargs),
                })
            return _sink
        raise AttributeError(name)


class LLMDebateAdapter(MASFrameworkAdapter):
    """Generic LLM Debate adapter. ExecutionMode.LIVE."""

    @property
    def name(self) -> str:
        return "LLMDebate"

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.LIVE

    def build_topology(
        self,
        scenario:     Scenario,
        config:       str,
        runtime_spec: "LLMDebateRunSpec | None" = None,
    ) -> MASTopology:
        if runtime_spec is not None:
            debaters   = runtime_spec.debaters
            aggregator = runtime_spec.aggregator
        else:
            debaters   = ["debater_0", "debater_1", "debater_2"]
            aggregator = "Aggregator"

        agents = list(dict.fromkeys([*debaters, aggregator]))
        edges:  list[tuple[str, str]] = []

        for src in debaters:
            for tgt in debaters:
                if src != tgt:
                    edges.append((src, tgt))
        for src in debaters:
            edges.append((src, aggregator))
        if _normalize(config).lower() in {"feedback", "multi_round", "debate", "standard"}:
            for tgt in debaters:
                edges.append((aggregator, tgt))

        seen: set[tuple[str, str]] = set()
        edges = [e for e in edges if not (e in seen or seen.add(e))]  # type: ignore

        return MASTopology(
            framework=self.name, mas_type="debate",
            agents=agents, edges=edges,
        )

    async def run_live(
        self,
        scenario:     Scenario,
        topology:     MASTopology,
        config:       str,
        model:        str,
        timeout:      int,
        runtime_spec: "LLMDebateRunSpec | None" = None,
    ) -> AsyncIterator[list[ChannelEvent]]:
        if runtime_spec is None:
            raise RuntimeError("LLMDebateAdapter requires LLMDebateRunSpec.")

        collector = DebateTraceCollector()

        if runtime_spec.backend_factory is not None:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._run_backend,
                    runtime_spec,
                    collector,
                    model,
                ),
                timeout=timeout,
            )
        else:
            await asyncio.to_thread(
                self._run_stub,
                runtime_spec,
                collector,
                model,
            )

        # Hard fail on zero events — prevents silent turns=0
        if not collector.events:
            raise RuntimeError(
                f"Native ACI backend produced zero logger events for "
                f"{scenario.scenario_id}. Check logger interface compatibility."
            )

        # Log trace diversity info
        trace_sig  = [(e.get("turn"), e.get("stage"), e.get("sender"),
                       e.get("receiver"), str(e.get("message",""))[:200])
                      for e in collector.events]
        trace_hash = hashlib.sha1(
            json.dumps(trace_sig, sort_keys=True).encode()
        ).hexdigest()[:12]
        print(
            f"[LLMDEBATE TRACE] scenario={scenario.scenario_id} "
            f"n_events={len(collector.events)} hash={trace_hash} "
            f"senders={dict(Counter(e.get('sender') for e in collector.events))}",
            flush=True,
        )

        for batch in self._trace_to_turns(collector.events, topology):
            if batch:
                yield batch

    # ------------------------------------------------------------------
    # Backend runners
    # ------------------------------------------------------------------

    def _run_backend(
        self,
        spec:      LLMDebateRunSpec,
        collector: DebateTraceCollector,
        model:     str,
    ) -> None:
        """Run ACI-style LLMDebate backend with phase-tagged logging."""
        mas = spec.backend_factory(collector)
        mas = self._patch_logging(mas, collector)
        attack = spec.attack_factory(mas) if spec.attack_factory else None

        # attack.run patches agents (InstructionInjection) but does NOT drive execution
        if attack is not None and hasattr(attack, "run"):
            attack.run(mas)

        if all(hasattr(mas, x) for x in ["bootstrap", "step", "conclude"]):
            collector.set_stage("bootstrap", 0)
            args, done = mas.bootstrap(spec.user_query)

            turn = 0
            while not done and turn < spec.max_rounds:
                turn += 1
                if attack is not None and hasattr(attack, "set_turn"):
                    attack.set_turn(turn)
                collector.set_stage("step", turn)
                args, done = mas.step(args)

            collector.set_stage("conclude", turn + 1)
            result = mas.conclude(args)

            # ACIArena verify() expects dict with "response" key
            if isinstance(result, str):
                final_answer = {"response": result}
            elif isinstance(result, dict):
                final_answer = result
            else:
                final_answer = {"response": str(result)}

            if attack is not None and hasattr(attack, "set_answer"):
                try:
                    attack.set_answer(final_answer)
                except Exception as e:
                    print(f"[LLMDEBATE SET_ANSWER_ERR] {type(e).__name__}: {e}", flush=True)

            try:
                verify = attack.verify() if attack is not None else None
            except Exception as e:
                verify = f"error={type(e).__name__}"
            print(
                f"[LLMDEBATE VERIFY] verify={verify} "
                f"answer_head={str(final_answer.get('response',''))[:160]!r}",
                flush=True,
            )

        elif hasattr(mas, "run"):
            collector.set_stage("run", 0)
            mas.run(spec.user_query)
        else:
            raise RuntimeError("Unsupported LLMDebate backend API.")

    def _run_stub(
        self,
        spec:      LLMDebateRunSpec,
        collector: DebateTraceCollector,
        model:     str,
    ) -> None:
        """TAMAS stub debate — real OpenAI calls per round for distinct embeddings."""
        import os

        def _llm(role: str, context: str) -> str:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
                resp   = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are {role} in a debate."},
                        {"role": "user",   "content": context[:800]},
                    ],
                    max_tokens=120, temperature=0.7,
                )
                return resp.choices[0].message.content or ""
            except Exception:
                return f"{role}: argument re '{context[:60]}'"

        query      = spec.user_query
        debaters   = spec.debaters
        aggregator = spec.aggregator

        # Bootstrap: initial positions
        collector.set_stage("bootstrap", 0)
        positions: dict[str, str] = {}
        for d in debaters:
            pos = _llm(d, f"State your position on: {query}")
            positions[d] = pos
            collector.log_message(d, aggregator, pos)

        # Debate rounds
        for round_idx in range(spec.max_rounds):
            collector.set_stage("step", round_idx + 1)
            new_pos: dict[str, str] = {}
            for d in debaters:
                others = " | ".join(f"{o}: {positions[o][:80]}"
                                    for o in debaters if o != d)
                resp = _llm(d, f"Round {round_idx+1}. Others: {others}. Rebuttal:")
                new_pos[d] = resp
                for other in debaters:
                    if other != d:
                        collector.log_message(d, other, resp)
                collector.log_message(d, aggregator, resp)
            positions = new_pos

        # Conclude
        collector.set_stage("conclude", spec.max_rounds + 1)
        all_args = " | ".join(f"{d}: {positions[d][:80]}" for d in debaters)
        conclusion = _llm(aggregator, f"Synthesize and give final answer: {all_args}")
        collector.log_message(aggregator, "system", conclusion)

    # ------------------------------------------------------------------
    # Logging patch for ACI backends
    # ------------------------------------------------------------------

    def _patch_logging(self, mas: Any, collector: DebateTraceCollector) -> Any:
        if not all(hasattr(mas, x) for x in ["bootstrap", "step", "conclude"]):
            return mas

        orig_bootstrap = mas.bootstrap
        orig_step      = mas.step
        orig_conclude  = mas.conclude

        def _debaters() -> list[str]:
            n = int(getattr(mas, "agents_num", 0) or 0)
            return [f"debater_{i}" for i in range(n)] if n > 0 \
                   else ["debater_0", "debater_1", "debater_2"]

        def _safe(name: str) -> str:
            try:
                return mas.get_agent(name).get_latest_response() or ""
            except Exception:
                return ""

        def _log(src: str, tgt: str, msg: str) -> None:
            if msg:
                collector.log_message(src, tgt, msg)

        def patched_bootstrap(query):
            args, done = orig_bootstrap(query)
            initial = f"{query} State your answer at the end."
            for name in _debaters():
                _log("user", name, initial)
                resp = _safe(name)
                if resp:
                    _log(name, "aggregator", resp)
            return args, done

        def patched_step(args):
            out_args, done = orig_step(args)
            names  = _debaters()
            latest = {n: _safe(n) for n in names}
            for src in names:
                msg = latest.get(src, "")
                if not msg:
                    continue
                for dst in names:
                    if src != dst:
                        _log(src, dst, msg)
                _log(src, "aggregator", msg)
            return out_args, done

        def patched_conclude(args):
            for name in _debaters():
                resp = _safe(name)
                if resp:
                    _log(name, "aggregator", resp)
            out   = orig_conclude(args)
            final = str(out.get("response", "")) if isinstance(out, dict) else str(out)
            if final:
                _log("aggregator", "system", final)
            return out

        mas.bootstrap = patched_bootstrap
        mas.step      = patched_step
        mas.conclude  = patched_conclude
        return mas

    # ------------------------------------------------------------------
    # Trace → CASPIAN turns  (one turn = one debate phase/round)
    # ------------------------------------------------------------------

    def _trace_to_turns(
        self,
        events:   list[dict[str, Any]],
        topology: MASTopology,
    ) -> list[list[ChannelEvent]]:
        encoder   = EmbeddingEncoder()
        agent_set = set(topology.agents)
        edge_set  = set(topology.edges)

        # Group by (debate_turn, stage).
        # For native ACI backends that don't call set_stage, each message
        # gets its own CASPIAN turn so we get multi-turn spectral signal.
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        all_same_stage = len({(ev.get("turn", 0), ev.get("stage", "unknown"))
                               for ev in events}) <= 1

        if all_same_stage and len(events) > 1:
            # Native ACI backend: one message per CASPIAN turn
            for idx, ev in enumerate(events):
                grouped[(idx, ev.get("stage", "msg"))] .append(ev)
        else:
            for ev in events:
                stage = str(ev.get("stage", "unknown"))
                turn  = int(ev.get("turn",  0) or 0)
                grouped[(turn, stage)].append(ev)

        turns: list[list[ChannelEvent]] = []

        for caspian_idx, key in enumerate(sorted(grouped.keys()), start=1):
            turn_num, stage = key
            batch_events    = grouped[key]
            batch:    list[ChannelEvent] = []

            for ev in batch_events:
                sender   = self._map_agent(ev.get("sender")   or ev.get("source"),   agent_set)
                receiver = self._map_agent(ev.get("receiver") or ev.get("target"),   agent_set)
                message  = str(ev.get("message") or ev.get("content") or "")

                if sender is None or receiver is None:
                    continue
                if (sender, receiver) not in edge_set:
                    continue

                ts  = float(ev.get("ts", time.time()))
                tok = max(1, len(message.split()))

                # Include stage+turn in content so different rounds → distinct vectors
                content = f"[{stage};t={turn_num}] {message}"

                vec_comm = encoder.comm_vector(content, tok)
                vec_exec = EmbeddingEncoder.exec_vector(
                    prompt_tokens=tok, completion_tokens=tok, latency_ms=0.0
                )

                batch.append(ChannelEvent(
                    turn=caspian_idx, source=sender, target=receiver, channel="comm",
                    payload={"kind": "debate_message", "stage": stage,
                             "debate_turn": turn_num, "content": content,
                             "u_vector": vec_comm.tolist(), "v_vector": vec_comm.tolist()},
                    vector=vec_comm, timestamp=ts,
                ))
                batch.append(ChannelEvent(
                    turn=caspian_idx, source=sender, target=receiver, channel="exec",
                    payload={"kind": "debate_exec_proxy", "stage": stage,
                             "debate_turn": turn_num, "token_count": tok,
                             "u_vector": vec_exec.tolist(), "v_vector": vec_exec.tolist()},
                    vector=vec_exec, timestamp=ts,
                ))

            if batch:
                sig  = [(e.source, e.target, e.channel,
                         e.payload.get("stage"), str(e.payload.get("content",""))[:80])
                        for e in batch]
                h    = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:8]
                print(
                    f"[LLMDEBATE TURN] t={caspian_idx} stage={stage} "
                    f"events={len(batch)} "
                    f"edges={len(set((e.source,e.target) for e in batch))} "
                    f"hash={h}",
                    flush=True,
                )
                turns.append(batch)

        return turns

    def _map_agent(self, raw: Any, agent_set: set[str]) -> str | None:
        if raw is None:
            return None
        raw_s = _normalize(str(raw))
        if raw_s in agent_set:
            return raw_s
        raw_l = raw_s.lower()
        for a in agent_set:
            if a.lower() == raw_l:
                return a
        if raw_l in {"aggregator", "judge", "moderator"}:
            for a in agent_set:
                if a.lower() in {"aggregator", "judge", "moderator"}:
                    return a
        return None