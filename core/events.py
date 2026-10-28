"""
core/events.py

Stable data contracts shared by all CASPIAN layers.

Core modules should not import from adapters.
Adapters may import these contracts to convert raw framework logs into
CASPIAN-compatible traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


Channel = Literal["comm", "mem", "tool", "exec"]
CHANNELS: tuple[Channel, ...] = ("comm", "mem", "tool", "exec")


@dataclass
class ChannelEvent:
    """
    A single normalized interaction event.

    source and target should usually be agent IDs. The channel field specifies
    whether influence moved through communication, memory, tool, or execution.
    """

    turn: int
    source: str
    target: str
    channel: Channel
    payload: dict[str, Any]
    vector: np.ndarray
    timestamp: float | None = None


@dataclass
class MASTopology:
    """
    Structural possibility graph G_0 for a multi-agent system.

    edges contains feasible directed agent-agent influence routes.
    channel_mask optionally restricts which channels are allowed on each edge.
    """

    framework: str
    agents: list[str]
    edges: list[tuple[str, str]]
    mas_type: str = "unknown"
    channel_mask: dict[tuple[str, str], set[Channel]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._agent_to_idx: dict[str, int] = {
            agent: idx for idx, agent in enumerate(self.agents)
        }
        self._edge_set: set[tuple[str, str]] = set(self.edges)

    def agent_index(self, name: str) -> int:
        if name not in self._agent_to_idx:
            raise KeyError(f"Unknown agent: {name}")
        return self._agent_to_idx[name]

    def n_agents(self) -> int:
        return len(self.agents)

    def has_agent(self, name: str) -> bool:
        return name in self._agent_to_idx

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._edge_set

    def feasible_channels(self, source: str, target: str) -> set[Channel]:
        edge = (source, target)

        if edge not in self._edge_set:
            return set()

        if self.channel_mask:
            return self.channel_mask.get(edge, set())

        return set(CHANNELS)

    def is_feasible(self, source: str, target: str, channel: Channel) -> bool:
        return channel in self.feasible_channels(source, target)


@dataclass
class FrameworkTrace:
    """
    A complete framework execution trace after adapter normalization.
    """

    scenario_id: str
    framework: str
    topology: MASTopology
    events_by_turn: list[list[ChannelEvent]]
    raw_log_path: str | None = None

    def n_turns(self) -> int:
        return len(self.events_by_turn)