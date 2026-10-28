"""
adapters/frameworks/specs.py

Generic runtime specs passed from benchmark adapters to framework adapters.

The benchmark adapter knows what agents/tools/tasks to build.
The framework adapter knows how to run the framework with those specs.
Neither knows about the other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """
    Framework-agnostic agent specification.
    The framework adapter maps this to its own agent class.
    """
    role:      str
    backstory: str
    goal:      str
    tools:     list[Any] = field(default_factory=list)


@dataclass
class AutoGenRunSpec:
    """Runtime spec for AutoGen — consumed by AutoGenAdapter.run_live."""
    agents:     "list[AgentSpec]"
    user_query: str
    config:     str = "standard"


@dataclass
class ACIArenaNativeRunSpec:
    """
    Generic native ACIArena spec for any framework that has an ACI backend.
    Framework adapters consume backend_factory + attack_factory.

    For LLMDebate: use LLMDebateRunSpec directly (already has this structure).
    For MetaGPT/CrewAI/AutoGen: use this spec when a native ACI backend exists.
    """
    framework:         str
    user_query:        str
    task_data:         "dict[str, Any]"
    llm_config:        "dict[str, Any]"
    attack_class_name: str
    malicious_agents:  "list[str]"
    max_turn:          int
    backend_factory:   "Any"   # Callable[[logger], mas]
    attack_factory:    "Any"   # Callable[[mas], attack]


@dataclass
class LLMDebateRunSpec:
    """
    Everything LLMDebateAdapter needs to run a debate.
    Built by the benchmark adapter — LLMDebate adapter never touches benchmark code.

    For TAMAS: debaters = TAMAS agent names, stub path (no backend_factory).
    For ACIArena: debaters = ["debater_0"...], backend_factory + attack_factory provided.
    """
    user_query:      str
    debaters:        list[str]
    aggregator:      str                        = "Aggregator"
    max_rounds:      int                        = 3
    backend_factory: "Any | None"               = None  # ACI-style backend constructor
    attack_factory:  "Any | None"               = None  # ACI-style attack constructor
    llm_config:      "dict[str, Any] | None"    = None
    malicious_agents: "list[str] | None"        = None


@dataclass
class MetaGPTRunSpec:
    """
    Everything MetaGPTAdapter needs to run MetaGPT.
    Built by the benchmark adapter — MetaGPT adapter never touches benchmark code.

    For TAMAS: uses metagpt_trace_runner.py subprocess (no backend_factory).
    For ACIArena: backend_factory provides native ACI MetaGPT backend.
    """
    agents:          list[AgentSpec]
    user_query:      str
    process:         str                    # "sequential" or "fully_connected"
    backend_factory: "Any | None" = None   # ACI native backend constructor
    attack_factory:  "Any | None" = None   # ACI attack constructor


@dataclass
class CrewAIRunSpec:
    """
    Everything CrewAIAdapter needs to build and run a crew.
    Built by the benchmark adapter — CrewAI adapter never touches benchmark code.
    """
    agents:             list[AgentSpec]
    user_query:         str
    process:            str              # "sequential" or "hierarchical"
    manager_role:       str | None = None
    manager_backstory:  str | None = None
    manager_goal:       str | None = None