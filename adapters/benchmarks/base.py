"""
adapters/benchmarks/base.py

Abstract base classes for CASPIAN benchmark adapters.

A benchmark adapter answers:
  - What scenarios exist in this benchmark?
  - For each scenario: what is the task, label, attack type, domain?
  - What is the ground truth attribution (if available)?

The benchmark adapter knows nothing about:
  - Which MAS framework runs the scenario
  - How to produce ChannelEvents
  - The CASPIAN core pipeline

Usage pattern:
    benchmark = TAMASAdapter(data_dir)
    for scenario in benchmark.load_scenarios(attack_only=True):
        gt = benchmark.get_ground_truth(scenario)
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Optional


AttackCategory = Literal["intent", "execution", "coordination", "benign", "unknown"]


@dataclass
class Scenario:
    """
    One benchmark scenario instance, fully self-contained.

    scenario_id     : globally unique string ID
    benchmark       : benchmark name ("TAMAS", "ACIArena", ...)
    source_file     : filename the scenario came from
    file_task_idx   : index within that file (0 for single-task files)
    domain          : task domain (education, finance, healthcare, legal, news, unknown)
    attack_type     : benchmark-specific attack folder/label (Byzantine, DPI, benign, ...)
    attack_category : CASPIAN paper category (intent/coordination/execution/benign/unknown)
    label           : 1 for attack, 0 for benign
    task_data       : raw dict from the benchmark JSON — framework adapters read this
    metadata        : any additional benchmark-specific fields
    """
    scenario_id:     str
    benchmark:       str
    source_file:     str
    file_task_idx:   int
    domain:          str
    attack_type:     str
    attack_category: AttackCategory
    label:           int
    task_data:       dict[str, Any]
    metadata:        dict[str, Any] = field(default_factory=dict)

    def user_query(self) -> str:
        """Return the task prompt from task_data, if present."""
        return str(self.task_data.get("user query", self.task_data.get("task", "")))

    def agent_names(self) -> list[str]:
        """Return agent names from task_data agents list, if present."""
        return [
            a["agent_name"]
            for a in self.task_data.get("agents", [])
            if isinstance(a.get("agent_name"), str) and a["agent_name"].strip()
        ]


@dataclass
class GroundTruth:
    """
    Ground truth attribution labels for a scenario.

    Not all benchmarks provide all fields.
    None means the benchmark does not label that component.

    origin          : agent name where cascade injection begins
    amplifier       : agent name that most strongly reinforces propagation
    bridge          : agent name that relays influence across the system
    spines          : list of propagation paths (each a list of agent names)
    dominant_channel: primary interaction channel ("comm"/"mem"/"tool"/"exec")
    injection_turn  : turn index where adversarial injection occurs
    attack_success  : whether the attack succeeded in the benchmark evaluation
    """
    origin:           Optional[str]       = None
    amplifier:        Optional[str]       = None
    bridge:           Optional[str]       = None
    spines:           Optional[list[list[str]]] = None
    dominant_channel: Optional[str]       = None
    injection_turn:   Optional[int]       = None
    attack_success:   Optional[bool]      = None


class BenchmarkAdapter(ABC):
    """
    Abstract base class for all CASPIAN benchmark adapters.

    Subclasses implement load_scenarios() and get_ground_truth()
    for a specific benchmark (TAMAS, ACIArena, ...).

    The benchmark adapter is stateless with respect to scenarios —
    it does not run anything, it only loads and describes scenarios.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name string, e.g. 'TAMAS' or 'ACIArena'."""
        ...

    @abstractmethod
    def load_scenarios(
        self,
        *,
        attack_only:  bool = False,
        benign_only:  bool = False,
        limit:        Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Scenario]:
        """
        Yield Scenario objects from this benchmark.

        Parameters
        ----------
        attack_only : if True, skip benign scenarios
        benign_only : if True, skip attack scenarios
        limit       : stop after this many scenarios
        **kwargs    : benchmark-specific filters
        """
        ...

    def get_ground_truth(self, scenario: Scenario) -> GroundTruth:
        """
        Return ground truth attribution for a scenario.

        Default implementation returns an empty GroundTruth.
        Override in subclasses that provide attribution labels.
        """
        return GroundTruth()