"""
core/detector.py

Cascade detector — stateful machine that consumes SpectralSignals
and A_tilde each turn and declares cascade onset.

Two Watch modes:

  StrictWatch (same-turn onset):
      amp > 1 + AMP_EPS
      AND delta_gap > DELTA_GAP_EPS
      AND lambda1 > prev_lambda1 + LAMBDA_GROWTH_EPS
      → eligible for INSTANT cascade

  DelayedWatch (gradual onset, for multi-turn cascades):
      any growth in last ONSET_WINDOW turns
      AND any transition (delta_gap > eps OR phase_shift) in last ONSET_WINDOW turns
      → enters MULTI_TURN candidate window only

InstantCascade:
    StrictWatch AND (PhaseShift OR CrossChannel) AND WeakLink
    Fires at t_0 = t_w.

MultiTurnCascade:
    watch_active (strict or delayed) starts candidate window at t_w.
    W_tw = min(MAX_WINDOW, ceil(1 / (gap_tw + eps)))
    Support per turn = lambda1 >= lambda1_tw OR gap <= gap_tw
                       OR cross_channel OR phase_shift
    MajoritySupport = sum(support_log) >= ceil(0.5 * window)
    Transition = at least one PhaseShift or CrossChannel in window
    WeakLink at t_0.

Paper refs: §3.3, §4.3.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from .spectral import SpectralSignals


EPS:                 float = 1e-10
WEAK_LINK_QUANTILE:  float = 0.75
MAX_WINDOW:          int   = 10
ONSET_WINDOW:        int   = 3     # rolling window for delayed watch
AMP_EPS:             float = 1e-4
DELTA_GAP_EPS:       float = 1e-4
LAMBDA_GROWTH_EPS:   float = 1e-4

# Security confirmation constants
AMP_CONFIRM_EPS:     float = 0.03    # minimum amp above 1.0 for multi-turn confirmation
SECURITY_RISK_THRESH: float = 0.50   # min security_risk/workflow_risk to confirm cascade
SEMANTIC_CARRY_THRESH: float = 0.65  # min carry score to confirm cascade

# Structural cascade scoring — MULTI_TURN only, never INSTANT
STRUCT_MIN_TURNS:        int   = 3
STRUCT_MIN_SUPPORT_FRAC: float = 0.60
STRUCT_PAIR_AMP_EPS:     float = 0.06   # strong amp near transition (not benign drift)
STRUCT_PEAK_AMP_EPS:     float = 0.08   # strong peak in window
STRUCT_ENERGY_RATIO:     float = 1.08
STRUCT_GAP_DROP_FRAC:    float = 0.08
STRUCT_SCORE_THRESH:     float = 2.5

# Dyadic cascade constants (N=2 reciprocal topology)
DYADIC_RAW_AMP_THRESH: float = 1.05

# Workflow cascade constants (N>=3 feed-forward DAG, e.g. MetaGPT chain)
WORKFLOW_PATH_THRESH:  float = 0.80   # min chain edge coverage fraction
WORKFLOW_PERSIST_TURNS: int  = 2      # turns of path-flow needed for MULTI_TURN_WORKFLOW   # raw energy increase across loop
DYADIC_CARRY_THRESH:   float = 0.70   # cosine similarity carry between directions
DYADIC_PERSIST_TURNS:  int   = 2      # turns of dyadic watch needed for MULTI_TURN


class CascadeType(Enum):
    INSTANT         = auto()
    MULTI_TURN      = auto()
    INSTANT_DYADIC    = auto()   # two-agent instant feedback cascade
    MULTI_TURN_DYADIC  = auto()  # two-agent persistent feedback cascade
    INSTANT_WORKFLOW   = auto()  # feed-forward DAG one-pass cascade
    MULTI_TURN_WORKFLOW = auto() # feed-forward DAG multi-pass cascade


@dataclass
class DetectionResult:
    cascade:      bool
    cascade_type: Optional[CascadeType]
    t_w:          Optional[int]
    t_0:          Optional[int]
    watch:        bool


def _weak_link(A_tilde: np.ndarray, min_path_edges: int = 1) -> bool:
    """
    True if a strong feasible propagation path exists in A_tilde.
    min_path_edges=1: any single strong directed edge is sufficient at onset.
    For multi-hop, uses boolean matrix power iteration.
    """
    A = np.asarray(A_tilde, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)

    edge_weights = A[A > 0.0].ravel()
    if edge_weights.size == 0:
        return False

    threshold = float(np.quantile(edge_weights, WEAK_LINK_QUANTILE))
    strong    = (A >= threshold)
    N         = A.shape[0]

    if min_path_edges <= 1:
        candidate = strong.copy()
        np.fill_diagonal(candidate, False)
        return bool(np.any(candidate))

    if N <= 2:
        return bool(np.any(strong))

    path_len_reach = strong.copy().astype(bool)
    for path_len in range(2, N):
        path_len_reach = (
            path_len_reach.astype(np.int32) @ strong.astype(np.int32)
        ) > 0
        if path_len >= min_path_edges:
            candidate = path_len_reach.copy()
            np.fill_diagonal(candidate, False)
            if np.any(candidate):
                return True

    return False


class _State(Enum):
    IDLE  = auto()
    WATCH = auto()


@dataclass
class Detector:
    """
    Stateful cascade detector. One instance per trace.
    Call update(signals, A_tilde) each turn in order.

    Internal state
    --------------
    _state          : IDLE or WATCH
    _t_w            : turn watch onset for current candidate
    _gap_tw         : spectral gap at Watch onset — fixes W_tw
    _lambda1_tw     : lambda1 at Watch onset — for support tracking
    _support_log    : per-turn support boolean over candidate window
    _transition     : whether PhaseShift or CrossChannel fired in window
    _prev_lambda1   : lambda1 from previous turn

    Rolling onset buffers (ONSET_WINDOW turns):
    _growth_log     : amp growth + lambda growth events
    _transition_log : gap contraction or phase shift events
    _cross_log      : cross_channel events
    """
    _state:          _State        = field(default=_State.IDLE,  init=False)
    # Dyadic state
    _dyadic_watch_log:    list[bool] = field(default_factory=list, init=False)
    _dyadic_evidence_log: list[bool] = field(default_factory=list, init=False)
    _prev_raw_energy:     float      = field(default=0.0,         init=False)
    # Workflow state (feed-forward DAG)
    _workflow_watch_log:  list[bool] = field(default_factory=list, init=False)
    _prev_path_coverage:  float      = field(default=0.0,         init=False)
    _t_w:            Optional[int] = field(default=None,         init=False)
    _gap_tw:         float         = field(default=1.0,          init=False)
    _lambda1_tw:     float         = field(default=0.0,          init=False)
    _support_log:    list[bool]    = field(default_factory=list, init=False)
    _transition:     bool          = field(default=False,        init=False)
    _prev_lambda1:   float         = field(default=0.0,          init=False)

    # Rolling onset buffers
    _growth_log:     list[bool]    = field(default_factory=list, init=False)
    _trans_log:      list[bool]    = field(default_factory=list, init=False)
    _cross_log:      list[bool]    = field(default_factory=list, init=False)

    _watch_age         : int         = field(default=0,   init=False)
    _risk_roll         : list[bool]  = field(default_factory=list, init=False)
    _candidate_risk_log: list[bool]  = field(default_factory=list, init=False)

    # Rolling numeric onset buffers — seeded into candidate window at watch start
    _amp_roll   : list[float] = field(default_factory=list, init=False)
    _energy_roll: list[float] = field(default_factory=list, init=False)
    _gap_roll   : list[float] = field(default_factory=list, init=False)
    _lambda_roll: list[float] = field(default_factory=list, init=False)

    # Structural cascade scoring state
    _energy_tw           : float       = field(default=0.0, init=False)
    _candidate_amp_log   : list[float] = field(default_factory=list, init=False)
    _candidate_trans_log : list[bool]  = field(default_factory=list, init=False)
    _candidate_energy_log: list[float] = field(default_factory=list, init=False)
    _candidate_gap_log   : list[float] = field(default_factory=list, init=False)
    _candidate_lambda_log: list[float] = field(default_factory=list, init=False)

    def update(
        self,
        signals: SpectralSignals,
        A_tilde: np.ndarray,
    ) -> DetectionResult:
        t = signals.turn

        # ------------------------------------------------------------------
        # Per-turn event booleans
        # ------------------------------------------------------------------
        growth_event = (
            signals.amp > 1.0 + AMP_EPS
            and signals.lambda1 > self._prev_lambda1 + LAMBDA_GROWTH_EPS
        )

        # Single-channel burst: large amp + dominant single channel (low entropy)
        single_channel_burst = (
            signals.amp >= 1.25
            and signals.H_norm <= 0.25
            and signals.lambda1 > self._prev_lambda1 + LAMBDA_GROWTH_EPS
        )

        transition_event = (
            signals.delta_gap > DELTA_GAP_EPS
            or signals.phase_shift
            or single_channel_burst
        )
        # NOTE: cross_channel removed from transition_event — benign multi-channel
        # group_chats always have comm+exec active so it's not a discriminative signal.
        cross_event = signals.cross_channel

        # Update rolling onset buffers
        self._growth_log.append(growth_event)
        self._trans_log.append(transition_event)
        self._cross_log.append(cross_event)
        self._growth_log = self._growth_log[-ONSET_WINDOW:]
        self._trans_log  = self._trans_log[-ONSET_WINDOW:]
        self._cross_log  = self._cross_log[-ONSET_WINDOW:]
        self._amp_roll.append(signals.amp)
        self._energy_roll.append(signals.energy)
        self._gap_roll.append(signals.gap)
        self._lambda_roll.append(signals.lambda1)
        self._amp_roll    = self._amp_roll[-ONSET_WINDOW:]
        self._energy_roll = self._energy_roll[-ONSET_WINDOW:]
        self._gap_roll    = self._gap_roll[-ONSET_WINDOW:]
        self._lambda_roll = self._lambda_roll[-ONSET_WINDOW:]

        # Dyadic evidence: excludes cross_channel to avoid overfiring on
        # normal comm+exec dyadic exchanges
        dyadic_extra = (
            growth_event
            or signals.delta_gap > DELTA_GAP_EPS
            or signals.phase_shift
            or single_channel_burst
        )
        self._dyadic_evidence_log.append(dyadic_extra)
        self._dyadic_evidence_log = self._dyadic_evidence_log[-ONSET_WINDOW:]

        # Rolling risk buffer — persists security events across watch window
        _sec_event = self._has_security_evidence(signals)
        self._risk_roll.append(_sec_event)
        self._risk_roll = self._risk_roll[-ONSET_WINDOW:]

        # ------------------------------------------------------------------
        # Watch conditions
        # ------------------------------------------------------------------
        strict_watch = (
            growth_event
            and transition_event
        )

        delayed_watch = (
            any(self._growth_log)
            and any(self._trans_log)
        )

        watch_active = strict_watch or delayed_watch

        self._prev_lambda1 = signals.lambda1

        # ------------------------------------------------------------------
        # Dyadic cascade — before IDLE early return so N=2 traces are
        # evaluated regardless of standard watch_active status
        # ------------------------------------------------------------------
        dyadic_result = self._check_dyadic(signals, A_tilde, t)
        if dyadic_result is not None:
            return dyadic_result

        # Workflow cascade — for N>=3 feed-forward DAG topologies (e.g. MetaGPT)
        # ------------------------------------------------------------------
        workflow_result = self._check_workflow(signals, A_tilde, t)
        if workflow_result is not None:
            return workflow_result

        # ------------------------------------------------------------------
        # IDLE
        # ------------------------------------------------------------------
        if self._state == _State.IDLE:
            if not watch_active:
                return DetectionResult(
                    cascade=False, cascade_type=None,
                    t_w=None, t_0=None, watch=False,
                )

            # Strict watch can trigger instant cascade — but only with security evidence
            if strict_watch and self._structural(signals, A_tilde, single_channel_burst):
                if self._has_security_evidence(signals):
                    return self._declare(CascadeType.INSTANT, t_w=t, t_0=t, watch=True)

            # Any watch starts multi-turn candidate window
            self._enter_watch(t, signals)
            return DetectionResult(
                cascade=False, cascade_type=None, t_w=t, t_0=None, watch=True,
            )

        # ------------------------------------------------------------------
        # WATCH
        # ------------------------------------------------------------------
        support_active = (
            signals.lambda1 >= self._lambda1_tw
            or signals.gap   <= self._gap_tw
            or signals.phase_shift
        )
        # NOTE: cross_channel deliberately excluded from support —
        # normal multi-channel group_chat always has comm+exec active,
        # which would make support trivially true for benign traffic.
        self._support_log.append(support_active)
        self._watch_age += 1
        self._candidate_amp_log.append(signals.amp)
        self._candidate_trans_log.append(transition_event)
        self._candidate_energy_log.append(signals.energy)
        self._candidate_gap_log.append(signals.gap)
        self._candidate_lambda_log.append(signals.lambda1)
        self._candidate_amp_log    = self._candidate_amp_log[-MAX_WINDOW:]
        self._candidate_trans_log  = self._candidate_trans_log[-MAX_WINDOW:]
        self._candidate_energy_log = self._candidate_energy_log[-MAX_WINDOW:]
        self._candidate_gap_log    = self._candidate_gap_log[-MAX_WINDOW:]
        self._candidate_lambda_log = self._candidate_lambda_log[-MAX_WINDOW:]
        self._candidate_risk_log.append(self._has_security_evidence(signals))
        self._candidate_risk_log = self._candidate_risk_log[-MAX_WINDOW:]

        if transition_event:
            self._transition = True

        assert self._t_w is not None
        W_tw = min(MAX_WINDOW, max(1, math.ceil(1.0 / (self._gap_tw + EPS))))
        t_0  = self._t_w + W_tw - 1

        # Early confirmation for short/medium traces.
        # Without this, tiny spectral gaps push W_tw to MAX_WINDOW and attacks
        # that spike early then stabilize are missed.
        if self._watch_age >= 2 and len(self._support_log) >= STRUCT_MIN_TURNS:
            majority_so_far = (sum(self._support_log)
                               >= math.ceil(STRUCT_MIN_SUPPORT_FRAC * len(self._support_log)))
            security_ok   = self._has_security_evidence(signals)
            struct_score  = self._structural_cascade_score(signals)
            structural_ok = struct_score >= STRUCT_SCORE_THRESH
            if majority_so_far and self._transition and _weak_link(A_tilde):
                # structural_ok elevates confidence but does NOT alone declare cascade
                # Only security_ok (adversarial content evidence) declares cascade
                risk_seen = any(self._candidate_risk_log)
                if risk_seen and structural_ok:
                    return self._declare(
                        CascadeType.MULTI_TURN, t_w=self._t_w, t_0=t, watch=watch_active,
                    )

        if t < t_0:
            return DetectionResult(
                cascade=False, cascade_type=None,
                t_w=self._t_w, t_0=None, watch=watch_active,
            )

        majority_support = (
            sum(self._support_log) >= math.ceil(0.5 * len(self._support_log))
        )

        if majority_support and self._transition and _weak_link(A_tilde):
            security_ok   = self._has_security_evidence(signals)
            struct_score  = self._structural_cascade_score(signals)
            structural_ok = struct_score >= STRUCT_SCORE_THRESH
            risk_seen = any(self._candidate_risk_log)
            if risk_seen and structural_ok:
                print(
                    f"[DETECTOR DECLARE] t={t} MULTI_TURN "
                    f"risk_seen={risk_seen} security_ok={security_ok} "
                    f"structural_ok={structural_ok} struct_score={struct_score:.3f} "
                    f"watch_age={self._watch_age}",
                    flush=True,
                )
                return self._declare(
                    CascadeType.MULTI_TURN, t_w=self._t_w, t_0=t_0, watch=watch_active,
                )

        # Failed — reset and re-evaluate this turn
        self._reset()
        if watch_active:
            if strict_watch and self._structural(signals, A_tilde, single_channel_burst):
                if self._has_security_evidence(signals):
                    return self._declare(CascadeType.INSTANT, t_w=t, t_0=t, watch=True)
            self._enter_watch(t, signals)

        return DetectionResult(
            cascade=False, cascade_type=None,
            t_w=self._t_w, t_0=None, watch=watch_active,
        )

    def _check_dyadic(
        self,
        signals: SpectralSignals,
        A_tilde: np.ndarray,
        t: int,
    ) -> "Optional[DetectionResult]":
        """
        Dyadic cascade detector for N=2 reciprocal topologies.

        Standard spectral onset cannot fire for N=2 (gap always ≈ 0).
        Instead:
          INSTANT_DYADIC   : t∈[2,3], both directions active, loop amplification
          MULTI_TURN_DYADIC: t≥3, persistent reciprocal cross-channel loop

        Conservative: never fires on first exchange (t=1).
        """
        A = np.asarray(A_tilde, dtype=np.float64)

        if A.shape[0] != 2:
            self._dyadic_watch_log.clear()
            self._prev_raw_energy = 0.0
            return None

        reciprocal  = bool(A[0, 1] > EPS and A[1, 0] > EPS)
        loop_energy = float(A[0, 1] + A[1, 0])
        had_prev    = self._prev_raw_energy > EPS
        loop_amp    = loop_energy / (self._prev_raw_energy + EPS)
        channel_carry = bool(signals.cross_channel)

        loop_active = reciprocal and channel_carry
        self._dyadic_watch_log.append(loop_active)
        self._dyadic_watch_log = self._dyadic_watch_log[-3:]
        self._prev_raw_energy  = loop_energy

        # Never declare on first observed exchange
        if not had_prev or not loop_active:
            return None

        semantic_carry    = float(getattr(signals, "dyadic_carry", 0.0) or 0.0)
        security_evidence = self._has_security_evidence(signals)

        # INSTANT_DYADIC: semantic carry OR (amplification AND security evidence)
        instant_evidence = (
            semantic_carry >= DYADIC_CARRY_THRESH
            or (loop_amp > DYADIC_RAW_AMP_THRESH and security_evidence)
        )
        if 2 <= t <= 3 and instant_evidence:
            self._dyadic_watch_log.clear()
            return DetectionResult(
                cascade=True, cascade_type=CascadeType.INSTANT_DYADIC,
                t_w=t, t_0=t, watch=True,
            )

        # MULTI_TURN_DYADIC: persistent loop + security/carry evidence
        recent_std_evidence = any(self._dyadic_evidence_log[-DYADIC_PERSIST_TURNS:])
        multi_evidence = (
            semantic_carry >= DYADIC_CARRY_THRESH
            or (security_evidence and (
                loop_amp > DYADIC_RAW_AMP_THRESH
                or recent_std_evidence
            ))
        )
        if t >= 3 and multi_evidence and sum(self._dyadic_watch_log[-DYADIC_PERSIST_TURNS:]) >= DYADIC_PERSIST_TURNS:
            self._dyadic_watch_log.clear()
            return DetectionResult(
                cascade=True, cascade_type=CascadeType.MULTI_TURN_DYADIC,
                t_w=t - DYADIC_PERSIST_TURNS + 1, t_0=t, watch=True,
            )

        return None

    def _structural_cascade_score(self, signals: "SpectralSignals") -> float:
        """
        Structural cascade score for subtle integrity/Byzantine cascades.
        Only used for MULTI_TURN — never INSTANT.
        Requires temporal co-occurrence of amplification + transition,
        which discriminates real cascades from benign late-turn fluctuations.
        """
        if len(self._support_log) < STRUCT_MIN_TURNS:
            return 0.0

        support_frac = sum(self._support_log) / max(1, len(self._support_log))
        amps     = list(self._candidate_amp_log)
        trans    = list(self._candidate_trans_log)
        gaps     = list(self._candidate_gap_log)
        energies = list(self._candidate_energy_log)
        lambdas  = list(self._candidate_lambda_log)

        if not amps or not trans:
            return 0.0

        # Key signal: STRONG amplification paired with transition (±1 turn)
        paired_growth_transition = False
        for i, a in enumerate(amps):
            if a >= 1.0 + STRUCT_PAIR_AMP_EPS:
                lo = max(0, i - 1)
                hi = min(len(trans), i + 2)
                if any(trans[j] for j in range(lo, hi)):
                    paired_growth_transition = True
                    break

        peak_amp      = max(amps) if amps else 1.0
        energy_ratio  = (max(energies) / (self._energy_tw + EPS)
                         if self._energy_tw > EPS and energies else 1.0)
        min_gap       = min(gaps) if gaps else self._gap_tw
        gap_drop_frac = max(0.0, self._gap_tw - min_gap) / (abs(self._gap_tw) + EPS)
        trans_count   = sum(bool(x) for x in trans)
        support_frac  = sum(self._support_log) / max(1, len(self._support_log))

        # Hard gates: required, not scored
        if support_frac < STRUCT_MIN_SUPPORT_FRAC:
            return 0.0
        if trans_count < 1:
            return 0.0
        # Weak benign drift near a transition must not count
        if not paired_growth_transition:
            return 0.0

        strong_peak_amp = peak_amp >= 1.0 + STRUCT_PEAK_AMP_EPS
        strong_energy   = energy_ratio >= STRUCT_ENERGY_RATIO
        strong_gap_drop = gap_drop_frac >= STRUCT_GAP_DROP_FRAC

        # Require paired strong growth plus at least one macro-structural change
        if not (strong_peak_amp or strong_energy or strong_gap_drop):
            return 0.0

        score = 1.5  # strong paired growth-transition
        if strong_peak_amp: score += 1.0
        if strong_energy:   score += 1.0
        if strong_gap_drop: score += 1.0
        return score

    def _has_security_evidence(self, signals: "SpectralSignals") -> bool:
        """
        True if adversarial/attack-relevant content is being carried through
        the cascade structure. Required for standard spectral cascade alerts
        to avoid false positives on benign MAS coordination.

        Currently populated by pipeline.py from comm event content scanning.
        Future: replace with semantic similarity to injected payload.
        """
        security_risk  = float(getattr(signals, "security_risk",  0.0) or 0.0)
        workflow_risk  = float(getattr(signals, "workflow_risk",   0.0) or 0.0)
        dyadic_carry   = float(getattr(signals, "dyadic_carry",    0.0) or 0.0)
        workflow_carry = float(getattr(signals, "workflow_carry",  0.0) or 0.0)

        return (
            security_risk  >= SECURITY_RISK_THRESH
            or workflow_risk  >= SECURITY_RISK_THRESH
            or dyadic_carry   >= SEMANTIC_CARRY_THRESH
            or workflow_carry >= SEMANTIC_CARRY_THRESH
        )

    def _check_workflow(
        self,
        signals: SpectralSignals,
        A_tilde: np.ndarray,
        t: int,
    ) -> "Optional[DetectionResult]":
        """
        Workflow cascade detector for N>=3 feed-forward DAG topologies.

        Pure DAG adjacency matrices are nilpotent (λ1=0). Standard spectral
        onset cannot fire. This branch detects chain propagation using:

          chain_active / (n-1) — fraction of consecutive chain edges active
          extra_evidence        — nontrivial beyond just normal chain execution

        INSTANT_WORKFLOW  : t==1, full chain + extra evidence
        MULTI_TURN_WORKFLOW: t>=2, persistent chain + extra evidence

        Conservative: does NOT fire on benign chain execution alone.
        Only fires for N>=3 with λ1<0.05 (pure DAG regime).
        """
        A = np.asarray(A_tilde, dtype=np.float64)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.maximum(A, 0.0)
        np.fill_diagonal(A, 0.0)
        n = A.shape[0]

        # Workflow/DAG branch: N>=3, sparse edge structure
        # active_edges <= n means strict chain (n-1 chain + max 1 feedback)
        # This excludes group_chat (fully-connected: n*(n-1) edges)
        if n < 3:
            self._workflow_watch_log.clear()
            self._prev_path_coverage = 0.0
            return None

        active_edges_total = int(np.sum(A > EPS))
        max_chain_edges    = n  # n-1 chain + 1 feedback = n
        workflow_like      = active_edges_total <= max_chain_edges

        if not workflow_like:
            # Not a sparse workflow — too many active edges for a chain topology
            self._workflow_watch_log.clear()
            self._prev_path_coverage = 0.0
            return None

        # Chain coverage: consecutive edges i→i+1 (workflow order from topology)
        expected_edges = max(1, n - 1)
        chain_active   = sum(1 for i in range(n - 1) if A[i, i + 1] > EPS)
        path_coverage  = float(chain_active) / float(expected_edges)
        path_mass      = float(sum(A[i, i + 1] for i in range(n - 1)))

        coverage_growth = (
            t > 1
            and self._prev_path_coverage > EPS
            and path_coverage > self._prev_path_coverage + 1e-4
        )

        # Semantic carry and security risk signals
        workflow_carry = float(getattr(signals, "workflow_carry", 0.0) or 0.0)
        workflow_risk  = float(getattr(signals, "workflow_risk",  0.0) or 0.0)

        # instant/multi evidence: security or carry only — no structural signals
        # phase/delta/coverage_growth happen in benign sparse workflows too
        instant_evidence = (
            workflow_risk  >= SECURITY_RISK_THRESH
            or workflow_carry >= SEMANTIC_CARRY_THRESH
        )
        multi_evidence = instant_evidence  # same gate for multi-turn

        workflow_watch = (
            path_coverage >= WORKFLOW_PATH_THRESH
            and path_mass  > EPS
            and multi_evidence
        )

        self._workflow_watch_log.append(workflow_watch)
        self._workflow_watch_log = self._workflow_watch_log[
            -max(WORKFLOW_PERSIST_TURNS + 1, 3):
        ]
        self._prev_path_coverage = path_coverage

        if not workflow_watch and not (t == 1 and path_coverage >= WORKFLOW_PATH_THRESH and instant_evidence):
            return None

        # INSTANT_WORKFLOW: one-pass propagation with nontrivial semantic/transition evidence
        # Does NOT fire on coverage_growth alone — normal benign chains also grow from 0
        if t == 1 and path_coverage >= WORKFLOW_PATH_THRESH and path_mass > EPS and instant_evidence:
            return DetectionResult(
                cascade=True, cascade_type=CascadeType.INSTANT_WORKFLOW,
                t_w=t, t_0=t, watch=True,
            )

        # MULTI_TURN_WORKFLOW: persistent path-flow across repeated passes
        if (t >= 2
                and sum(self._workflow_watch_log[-WORKFLOW_PERSIST_TURNS:])
                >= WORKFLOW_PERSIST_TURNS):
            self._workflow_watch_log.clear()
            return DetectionResult(
                cascade=True, cascade_type=CascadeType.MULTI_TURN_WORKFLOW,
                t_w=t - WORKFLOW_PERSIST_TURNS + 1, t_0=t, watch=True,
            )

        return None

    def _structural(
        self,
        signals: SpectralSignals,
        A_tilde: np.ndarray,
        single_channel_burst: bool = False,
    ) -> bool:
        return (
            (signals.phase_shift or signals.cross_channel or single_channel_burst)
            and _weak_link(A_tilde)
        )

    def _enter_watch(self, t: int, signals: SpectralSignals) -> None:
        self._state      = _State.WATCH
        self._t_w        = t
        self._gap_tw     = signals.gap
        self._lambda1_tw = signals.lambda1
        self._energy_tw  = signals.energy

        # Seed candidate window with rolling onset history so delayed Watch
        # captures the spike that triggered it, not just the current turn
        self._candidate_amp_log    = list(self._amp_roll)    or [signals.amp]
        self._candidate_energy_log = list(self._energy_roll) or [signals.energy]
        self._candidate_gap_log    = list(self._gap_roll)    or [signals.gap]
        self._candidate_lambda_log = list(self._lambda_roll) or [signals.lambda1]
        self._candidate_trans_log  = (list(self._trans_log)
                                      or [signals.delta_gap > DELTA_GAP_EPS
                                          or signals.phase_shift])
        # Support: only current turn is confirmed — rolling history is for scoring only
        self._support_log = [True]
        # Transition from recent onset history, but doesn't alone confirm cascade
        self._transition       = any(self._candidate_trans_log)
        self._watch_age        = 0
        self._candidate_risk_log = (list(self._risk_roll)
                                    or [self._has_security_evidence(signals)])

    def _declare(self, cascade_type, t_w, t_0, watch) -> DetectionResult:
        self._reset()
        return DetectionResult(
            cascade=True, cascade_type=cascade_type,
            t_w=t_w, t_0=t_0, watch=watch,
        )

    def _reset(self) -> None:
        """Reset candidate window state. _prev_lambda1 and rolling onset buffers
        are NOT reset — they track true history across candidate boundaries."""
        self._energy_tw            = 0.0
        self._watch_age            = 0
        self._candidate_amp_log    = []
        self._candidate_trans_log  = []
        self._candidate_energy_log = []
        self._candidate_gap_log    = []
        self._candidate_lambda_log = []
        self._candidate_risk_log    = []
        self._state           = _State.IDLE
        self._t_w             = None
        self._gap_tw          = 1.0
        self._lambda1_tw      = 0.0
        self._support_log     = []
        self._transition      = False
        # Dyadic watch log NOT reset here — persists across candidate windows