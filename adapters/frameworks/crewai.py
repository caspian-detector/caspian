"""
adapters/frameworks/crewai.py

CrewAI framework adapter for CASPIAN — generic, benchmark-agnostic.

Receives a CrewAIRunSpec from the benchmark adapter (e.g. tamas.build_crewai_spec).
Wraps every provided tool with a tracing wrapper that emits ChannelEvents directly.
Runs CrewAI in-process via asyncio.to_thread.

No TAMAS imports. No ACIArena imports. No benchmark-specific logic.
No subprocess. No temp files. No __TRACE__ parsing.

Channel mapping:
  tool  every tool_call and tool_result from wrapped tools
  comm  agent step text output
  mem   context passed between tasks (inferred from task context)
  exec  per-step latency + token proxy
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, AsyncIterator

from core.events import ChannelEvent, MASTopology
from adapters.benchmarks.base import Scenario
from adapters.frameworks.base import MASFrameworkAdapter, ExecutionMode
from adapters.frameworks.encoder import EmbeddingEncoder
from adapters.frameworks.specs import CrewAIRunSpec


def _normalize_name(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _agent_names_from_task(task_data: dict[str, Any]) -> list[str]:
    return [
        a["agent_name"]
        for a in task_data.get("agents", [])
        if isinstance(a.get("agent_name"), str) and a["agent_name"].strip()
    ]


class CrewAIAdapter(MASFrameworkAdapter):
    """
    Generic CrewAI framework adapter. ExecutionMode.LIVE.

    Receives a CrewAIRunSpec (built by the benchmark adapter).
    Wraps tools with tracing wrappers. Runs CrewAI in-process.
    Emits ChannelEvents via an asyncio queue.
    """

    @property
    def name(self) -> str:
        return "CrewAI"

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.LIVE

    def build_topology(
        self,
        scenario:     Scenario,
        config:       str,
        runtime_spec: "CrewAIRunSpec | None" = None,
    ) -> MASTopology:
        # Use runtime_spec agent roles if available — these are the exact names
        # CrewAI will emit in step_callback, ensuring topology matches events.
        if runtime_spec is not None:
            agents = [a.role for a in runtime_spec.agents]
        else:
            agents = _agent_names_from_task(scenario.task_data)

        cfg = _normalize_name(config)

        if not agents:
            return MASTopology(framework=self.name, mas_type="unknown",
                               agents=[], edges=[])

        # ACIArena CrewAI fallback: strict sequential chain (no back-edges)
        if runtime_spec is not None and getattr(runtime_spec, "process", "") == "sequential":
            mas_type = "sequential_workflow"
            edges    = [(agents[i], agents[i + 1]) for i in range(len(agents) - 1)]
            return MASTopology(
                framework=self.name, mas_type=mas_type,
                agents=agents, edges=edges,
            )

        if cfg in {"decentralized", "sequential"}:
            mas_type = "decentralized"
            edges: list[tuple[str, str]] = []
            for i in range(len(agents) - 1):
                edges.append((agents[i], agents[i + 1]))
                if i > 0:
                    edges.append((agents[i], agents[i - 1]))
            if len(agents) >= 2:
                edges.append((agents[-1], agents[-2]))
            seen: set[tuple[str, str]] = set()
            edges = [e for e in edges if not (e in seen or seen.add(e))]  # type: ignore
        elif cfg in {"centralized", "hierarchical"}:
            mas_type = "centralized"
            edges = [(a, b) for a in agents for b in agents if a != b]
        else:
            mas_type = "group_chat"
            edges = [(a, b) for a in agents for b in agents if a != b]

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
        runtime_spec: CrewAIRunSpec | None = None,
    ) -> AsyncIterator[list[ChannelEvent]]:
        """
        Run CrewAI in-process and yield ChannelEvent lists per agent step.

        runtime_spec must be provided — built by the benchmark adapter.
        If None, yields nothing (no tool info available).
        """
        if runtime_spec is None:
            return

        try:
            from crewai import Agent, Crew, Task, Process, LLM
        except ImportError:
            raise RuntimeError("crewai is not installed.")

        import os
        os.environ["CREWAI_TRACING_ENABLED"] = "false"
        os.environ["OTEL_SDK_DISABLED"]       = "true"
        os.environ["CI"]                      = "true"
        os.environ["PYTHONUNBUFFERED"]        = "1"

        encoder      = EmbeddingEncoder()
        agent_lookup = {_normalize_name(n): n for n in topology.agents}
        loop         = asyncio.get_event_loop()
        queue: asyncio.Queue[list[ChannelEvent] | None] = asyncio.Queue()
        turn  = [0]

        def infer_targets(source: str) -> list[str]:
            return [tgt for src, tgt in topology.edges
                    if src == source and tgt != source]

        def emit(events: list[ChannelEvent]) -> None:
            if events:
                loop.call_soon_threadsafe(queue.put_nowait, events)

        def make_tool_wrapper(agent_role: str, original_tool: Any) -> Any:
            """
            Wrap a TAMAS CrewAI tool so every call/result emits ChannelEvents.
            Identical pattern to the old crewai_trace_runner.py make_traced_tool.
            """
            from crewai.tools import tool as crewai_tool

            tool_name = (
                getattr(original_tool, "name", None)
                or getattr(original_tool, "__name__", None)
                or "unknown_tool"
            )

            source  = agent_lookup.get(_normalize_name(agent_role))
            targets = infer_targets(source) if source else []

            def _call_original(*args: Any, **kwargs: Any) -> Any:
                for method in ("_run", "run", "__call__"):
                    fn = getattr(original_tool, method, None)
                    if fn and callable(fn):
                        return fn(*args, **kwargs)
                raise TypeError(f"Cannot call tool {tool_name}")

            @crewai_tool(tool_name)
            def traced(*args: Any, **kwargs: Any) -> str:
                """Traced CASPIAN tool wrapper."""
                t0 = time.time()
                if not source or not targets:
                    try:
                        return str(_call_original(*args, **kwargs))
                    except Exception as e:
                        return str(e)

                turn[0] += 1
                t       = turn[0]
                arg_str = str(kwargs or args)
                batch: list[ChannelEvent] = []

                # tool_call events
                vec_call = encoder.tool_vector(tool_name, arg_str, "")
                for tgt in targets:
                    batch.append(ChannelEvent(
                        turn=t, source=source, target=tgt, channel="tool",
                        payload={"kind": "tool_call", "tool_name": tool_name,
                                 "arguments": arg_str, "output": "",
                                 "u_vector": vec_call.tolist(), "v_vector": vec_call.tolist()},
                        vector=vec_call, timestamp=t0,
                    ))

                try:
                    result  = _call_original(*args, **kwargs)
                    output  = str(result)
                    latency = (time.time() - t0) * 1000.0

                    # tool_result events
                    vec_res = encoder.tool_vector(tool_name, arg_str, output)
                    for tgt in targets:
                        batch.append(ChannelEvent(
                            turn=t, source=source, target=tgt, channel="tool",
                            payload={"kind": "tool_result", "tool_name": tool_name,
                                     "arguments": arg_str, "output": output,
                                     "u_vector": vec_res.tolist(), "v_vector": vec_res.tolist()},
                            vector=vec_res, timestamp=t0,
                        ))

                    # comm + mem: tool output handoff to downstream agents
                    handoff = f"{tool_name} {output}"
                    tok     = max(1, len(handoff.split()))
                    vec_comm = encoder.comm_vector(handoff, tok)
                    vec_mem  = encoder.mem_vector(handoff, op="write")
                    for tgt in targets:
                        batch.append(ChannelEvent(
                            turn=t, source=source, target=tgt, channel="comm",
                            payload={"kind": "tool_output_handoff", "tool_name": tool_name,
                                     "content": handoff, "token_count": tok,
                                     "u_vector": vec_comm.tolist(), "v_vector": vec_comm.tolist()},
                            vector=vec_comm, timestamp=t0,
                        ))
                        batch.append(ChannelEvent(
                            turn=t, source=source, target=tgt, channel="mem",
                            payload={"kind": "tool_output_memory", "tool_name": tool_name,
                                     "content": handoff, "op": "write",
                                     "u_vector": vec_mem.tolist(), "v_vector": vec_mem.tolist()},
                            vector=vec_mem, timestamp=t0,
                        ))

                    # exec events
                    proxy  = max(1, len(output.split()))
                    vec_ex = EmbeddingEncoder.exec_vector(proxy * 4, proxy, latency)
                    for tgt in targets:
                        batch.append(ChannelEvent(
                            turn=t, source=source, target=tgt, channel="exec",
                            payload={"kind": "tool_exec", "tool_name": tool_name,
                                     "latency_ms": latency,
                                     "u_vector": vec_ex.tolist(), "v_vector": vec_ex.tolist()},
                            vector=vec_ex, timestamp=t0,
                        ))

                    emit(batch)
                    return output

                except Exception as e:
                    err     = str(e)
                    latency = (time.time() - t0) * 1000.0
                    vec_err = encoder.tool_vector(tool_name, arg_str, err)
                    for tgt in targets:
                        batch.append(ChannelEvent(
                            turn=t, source=source, target=tgt, channel="tool",
                            payload={"kind": "tool_error", "tool_name": tool_name,
                                     "arguments": arg_str, "output": err, "error": True,
                                     "u_vector": vec_err.tolist(), "v_vector": vec_err.tolist()},
                            vector=vec_err, timestamp=t0,
                        ))
                    emit(batch)
                    raise

            traced.__doc__ = f"CASPIAN traced wrapper for {tool_name}."
            return traced

        # No step_callback — tool wrappers emit events directly and reliably.
        # step_callback fires on every CrewAI internal step (planning, retries,
        # metadata) producing hundreds of single-channel micro-turns that pollute
        # the CASPIAN signal. Tool wrappers give us tool_call + tool_result + exec
        # only on actual tool executions — the right granularity.

        def build_and_run() -> None:
            import os
            from crewai import Agent, Crew, Task, Process, LLM

            # Build LLM
            if "gpt" in model.lower():
                llm = LLM(model=model, api_key=os.environ.get("OPENAI_API_KEY"))
            elif "gemini" in model.lower():
                llm = LLM(model=model, api_key=os.environ.get("GOOGLE_API_KEY"))
            else:
                llm = LLM(model=model)

            # Build agents — wrap each tool with CASPIAN tracer
            agent_objs: list[Any] = []
            for spec in runtime_spec.agents:
                wrapped_tools = [
                    make_tool_wrapper(spec.role, t)
                    for t in spec.tools
                ]
                agent_objs.append(Agent(
                    role=spec.role,
                    backstory=spec.backstory,
                    goal=spec.goal,
                    tools=wrapped_tools,
                    verbose=False,
                    allow_delegation=True,
                    llm=llm,
                ))

            # Per-agent payload injection for ACIArena
            _aci_payload = scenario.task_data.get("_aci_attack_payload", "") if scenario else ""
            _aci_attack  = scenario.task_data.get("_aci_attack_class", "")   if scenario else ""
            _is_benign   = _aci_attack in ("NoneAttack", "none", "benign", "harmless", "")
            _mal_roles   = set(scenario.task_data.get("_aci_malicious_agents", [])) if scenario else set()

            def _task_desc(role: str) -> str:
                if _aci_payload and not _is_benign and role in _mal_roles:
                    print(
                        f"[CREWAI INJECT] role={role} payload_len={len(_aci_payload)} "
                        f"head={_aci_payload[:120]!r}",
                        flush=True,
                    )
                    return f"{runtime_spec.user_query}\n\n[Instruction] {_aci_payload}"
                return runtime_spec.user_query

            # Build tasks
            if runtime_spec.process == "sequential":
                prior: list[Task] = []
                tasks: list[Task] = []
                for ag in agent_objs:
                    t = Task(
                        description=_task_desc(ag.role),
                        expected_output="A concise final answer or implementation for the task.",
                        agent=ag,
                        context=[prior[-1]] if prior else [],  # strict chain, not all-prior
                    )
                    tasks.append(t)
                    prior.append(t)
                crew = Crew(
                    agents=agent_objs,
                    tasks=tasks,
                    verbose=False,
                    process=Process.sequential,
                )
            else:
                manager = Agent(
                    role=runtime_spec.manager_role or "Manager Agent",
                    backstory=runtime_spec.manager_backstory or "You oversee the crew.",
                    goal=runtime_spec.manager_goal or "Ensure objectives are met.",
                    verbose=False,
                    allow_delegation=True,
                    llm=llm,
                )
                tasks = [Task(
                    description=runtime_spec.user_query,
                    expected_output="",
                )]
                crew = Crew(
                    agents=agent_objs,
                    tasks=tasks,
                    verbose=False,
                    process=Process.hierarchical,
                    manager_agent=manager,
                )

            def _text_from_task(task: Any) -> str:
                out = getattr(task, "output", None)
                if out is None:
                    return ""
                for attr in ("raw", "output", "result", "content"):
                    val = getattr(out, attr, None)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                return str(out).strip()

            try:
                import time as _t
                t0 = _t.time()
                crew.kickoff()
                latency_total = (_t.time() - t0) * 1000.0
                per_task_ms   = latency_total / max(1, len(tasks))

                # Emit all sequential task outputs as ONE workflow turn
                # so the detector sees the full chain path in one turn
                workflow_batch: list[ChannelEvent] = []
                workflow_turn = (turn[0] + 1)

                for task in tasks:
                    try:
                        role = getattr(getattr(task, "agent", None), "role", "")
                        text = _text_from_task(task)
                        if not role or not text:
                            continue
                        _risk_hit = any(k in text.lower() for k in
                            ["api key", "sk-", "secret", "leak", "ddos", "ignore previous"])
                        print(
                            f"[CREWAI TASK_OUT] role={role} len={len(text)} "
                            f"risk={_risk_hit} head={text[:160]!r}",
                            flush=True,
                        )
                        source = agent_lookup.get(_normalize_name(role))
                        if source is None:
                            continue
                        targets = [tgt for src, tgt in topology.edges if src == source]
                        if not targets:
                            continue
                        tok    = max(1, len(text.split()))
                        ts     = _t.time()
                        vec_comm = encoder.comm_vector(text, tok)
                        vec_mem  = encoder.mem_vector(text, op="write")
                        vec_exec = EmbeddingEncoder.exec_vector(tok, tok, per_task_ms)
                        for tgt in targets:
                            workflow_batch.append(ChannelEvent(
                                turn=workflow_turn, source=source, target=tgt, channel="comm",
                                payload={"kind": "crewai_task_output", "content": text,
                                         "token_count": tok,
                                         "u_vector": vec_comm.tolist(), "v_vector": vec_comm.tolist()},
                                vector=vec_comm, timestamp=ts,
                            ))
                            workflow_batch.append(ChannelEvent(
                                turn=workflow_turn, source=source, target=tgt, channel="mem",
                                payload={"kind": "crewai_context_handoff", "content": text,
                                         "op": "write",
                                         "u_vector": vec_mem.tolist(), "v_vector": vec_mem.tolist()},
                                vector=vec_mem, timestamp=ts,
                            ))
                            workflow_batch.append(ChannelEvent(
                                turn=workflow_turn, source=source, target=tgt, channel="exec",
                                payload={"kind": "crewai_task_exec", "token_count": tok,
                                         "latency_ms": per_task_ms,
                                         "u_vector": vec_exec.tolist(), "v_vector": vec_exec.tolist()},
                                vector=vec_exec, timestamp=ts,
                            ))
                    except Exception as _e:
                        print(f"[CREWAI TASK_TRACE_ERR] {type(_e).__name__}: {_e}", flush=True)

                if workflow_batch:
                    turn[0] = workflow_turn
                    emit(workflow_batch)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread_task = asyncio.ensure_future(
            asyncio.wait_for(asyncio.to_thread(build_and_run), timeout=timeout)
        )

        def _batch_source(batch: list[ChannelEvent]) -> str:
            return batch[0].source if batch else ""

        try:
            pending:        list[ChannelEvent] = []
            pending_source: str | None         = None

            while True:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)

                if item is None:
                    if pending:
                        yield pending
                    break

                src = _batch_source(item)

                # Flush when a different agent's events arrive
                if pending and pending_source is not None and src != pending_source:
                    yield pending
                    pending = []

                pending.extend(item)
                pending_source = src

        except asyncio.TimeoutError:
            if pending:
                yield pending
            thread_task.cancel()
        finally:
            try:
                await thread_task
            except Exception:
                pass