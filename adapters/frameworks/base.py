"""
adapters/frameworks/base.py

Abstract base class for CASPIAN MAS framework adapters.

A framework adapter answers:
  - Given a Scenario, how do I build the agent topology (G0)?
  - Given a Scenario, how do I run or parse the framework and get a trace?
  - Given a raw trace, how do I produce ChannelEvents per turn?

The framework adapter knows nothing about:
  - Which benchmark the scenario came from
  - The CASPIAN core pipeline
  - Detection or attribution logic

Two execution modes:
  LIVE   — adapter runs the framework in-process (AutoGen async stream)
  SUBPROCESS — adapter launches an external process and parses its log (CrewAI)

For LIVE mode, implement run_live(scenario) -> AsyncIterator[list[ChannelEvent]]
For SUBPROCESS mode, implement run_subprocess(scenario) -> list[list[ChannelEvent]]

The generic runner (run_matrix.py) calls run(scenario) which dispatches
to the correct mode automatically based on execution_mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, AsyncIterator

from core.events import ChannelEvent, MASTopology
from adapters.benchmarks.base import Scenario


class ExecutionMode(Enum):
    LIVE       = auto()   # in-process async stream (AutoGen)
    SUBPROCESS = auto()   # external process + log parsing (CrewAI, MetaGPT)


class MASFrameworkAdapter(ABC):
    """
    Abstract base class for all CASPIAN MAS framework adapters.

    Subclasses implement build_topology() and either
    run_live() or run_subprocess() depending on execution_mode.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework name string, e.g. 'AutoGen', 'CrewAI'."""
        ...

    @property
    @abstractmethod
    def execution_mode(self) -> ExecutionMode:
        """Whether this adapter runs live (in-process) or via subprocess."""
        ...

    @abstractmethod
    def build_topology(self, scenario: Scenario, config: str) -> MASTopology:
        """
        Build the structural possibility graph G0 for this scenario.

        Parameters
        ----------
        scenario : Scenario from a benchmark adapter
        config   : framework-specific configuration string
                   (e.g. "RoundRobin", "Swarm", "Magentic_one" for AutoGen;
                    "Decentralized", "Centralized" for CrewAI)

        Returns
        -------
        MASTopology with agents and edges set for this scenario + config.
        """
        ...

    async def run_live(
        self,
        scenario:     Scenario,
        topology:     MASTopology,
        config:       str,
        model:        str,
        timeout:      int,
        runtime_spec: "Any | None" = None,
    ) -> AsyncIterator[list[ChannelEvent]]:
        """
        Run the framework live and yield ChannelEvent lists turn by turn.

        Implement this for LIVE execution mode (e.g. AutoGen).
        Each yielded list is all ChannelEvents for one pipeline turn.

        Default raises NotImplementedError — override in LIVE adapters.
        """
        raise NotImplementedError(
            f"{self.name} does not implement run_live(). "
            f"Use execution_mode=SUBPROCESS and implement run_subprocess()."
        )
        # Make this an async generator
        return
        yield  # type: ignore[misc]

    async def run_subprocess(
        self,
        scenario: Scenario,
        topology: MASTopology,
        config:   str,
        model:    str,
        timeout:  int,
    ) -> list[list[ChannelEvent]]:
        """
        Run the framework as a subprocess, parse the log, and return all turns.

        Implement this for SUBPROCESS execution mode (e.g. CrewAI, MetaGPT).
        Returns all turns at once after the subprocess completes.

        Default raises NotImplementedError — override in SUBPROCESS adapters.
        """
        raise NotImplementedError(
            f"{self.name} does not implement run_subprocess(). "
            f"Use execution_mode=LIVE and implement run_live()."
        )