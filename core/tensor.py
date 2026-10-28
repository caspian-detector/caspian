"""
core/tensor.py

Builds the causal influence tensor A_tensor and raw matrix A_raw
from LI-CTE estimators at each turn.

A_tensor[i, j, c] = LI-CTE scalar for edge (i->j) channel c at turn t
                    shape (N, N, |C|), dtype float64, all entries >= 0

A_raw[i, j]       = sum over channels of A_tensor[i, j, :]
                    shape (N, N), dtype float64
                    Used by attribution (bridge score) at true interaction
                    volume scale — normalization happens in normalization.py.

This module owns:
  - one LICTEState per (i, j, c) triplet
  - the mapping from ChannelEvents to (u, v) feature vectors
  - the per-turn update loop over feasible edges and channels

It does NOT normalize, detect, or attribute anything.

Paper refs:
  §4.2  Unified causal influence modeling
  §3    Interaction channels and state slices
  Appendix F  LI-CTE computation pipeline
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from .events import ChannelEvent, MASTopology, CHANNELS
from .states import LICTEState
from . import licte

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Channel feature dimensions
# Must match what the framework adapters produce in ChannelEvent.vector.
# Defined here as the single source of truth for tensor construction.
# ---------------------------------------------------------------------------

# Channel feature dimensions — must match encoder.py CHANNEL_DIM
# comm: 12-d MiniLM projection + 4 structured = 16
# mem:  10-d MiniLM projection + 2 structured = 12
# tool:  8-d joint + 2-d name projection + 2 structured = 12
# exec:  8-d execution metadata only
CHANNEL_DIM: dict[str, int] = {
    "comm": 16,
    "mem":  12,
    "tool": 12,
    "exec": 8,
}


class InfluenceTensorEstimator:
    """
    Maintains one LICTEState per (i, j, c) and produces A_tensor / A_raw
    from a list of ChannelEvents at each turn.

    Parameters
    ----------
    topology : MASTopology
        Defines agents (name -> index mapping) and feasible edges E0.
        Only edges in topology.edges get estimators; all others stay zero.

    Attributes
    ----------
    N        : number of agents
    n_ch     : number of channels (always 4)
    _states  : dict mapping (i: int, j: int, c: str) -> LICTEState
    """

    def __init__(self, topology: MASTopology) -> None:
        self.topology = topology
        self.N        = topology.n_agents()
        self.n_ch     = len(CHANNELS)

        # Initialise one LICTEState per feasible (i, j, c)
        self._states: dict[tuple[int, int, str], LICTEState] = {}
        self.channel_to_idx: dict[str, int] = {c: k for k, c in enumerate(CHANNELS)}

        for src_name, tgt_name in topology.edges:
            i = topology.agent_index(src_name)
            j = topology.agent_index(tgt_name)
            for c in topology.feasible_channels(src_name, tgt_name):
                self._states[(i, j, c)] = LICTEState(
                    channel=c,
                    d_c=CHANNEL_DIM[c],
                    alpha=_channel_alpha(c),
                )

    # ------------------------------------------------------------------
    # Main per-turn update
    # ------------------------------------------------------------------

    def update(
        self,
        events: list[ChannelEvent],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process all ChannelEvents for one turn.

        Parameters
        ----------
        events : list of ChannelEvent for this turn
            Each event carries a .vector of shape (d_c,) produced by
            the framework adapter.

        Returns
        -------
        A_tensor : np.ndarray, shape (N, N, n_ch), dtype float64
            A_tensor[i, j, k] = LI-CTE score for edge (i->j), channel k.
            Zero for non-feasible edges.

        A_raw : np.ndarray, shape (N, N), dtype float64
            Channel sum: A_raw[i, j] = sum_c A_tensor[i, j, c].
            Preserves raw interaction volume for attribution (bridge).
        """
        # Aggregate events by (source_idx, target_idx, channel)
        # Multiple events on the same (i, j, c) within one turn are
        # averaged — preserves vector direction while reducing noise.
        agg = self._aggregate_events(events)

        A_tensor = np.zeros((self.N, self.N, self.n_ch), dtype=np.float64)

        for (i, j, c), (u, v) in agg.items():
            key = (i, j, c)
            if key not in self._states:
                # Edge or channel not in G0 — skip
                continue

            state = self._states[key]
            score = licte.update(state, u, v)

            A_tensor[i, j, self.channel_to_idx[c]] = score

        # For edges with no events this turn, LI-CTE state is not advanced.
        # This treats missing observations as unobserved rather than zero influence.
        # Temporal decay occurs only when the edge/channel receives a new observation.

        A_raw = A_tensor.sum(axis=2)

        return A_tensor, A_raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_events(
        self,
        events: list[ChannelEvent],
    ) -> dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]]:
        """
        Group events by (i, j, channel). Average source-side u vectors and
        target-side v vectors separately within each group.
        """
        buckets: dict[
            tuple[int, int, str], list[tuple[np.ndarray, np.ndarray]]
        ] = defaultdict(list)

        for event in events:
            try:
                i = self.topology.agent_index(event.source)
                j = self.topology.agent_index(event.target)
            except KeyError:
                continue

            if event.channel not in CHANNELS:
                continue

            if not self.topology.is_feasible(event.source, event.target, event.channel):
                continue

            u, v = self._extract_uv(event)
            buckets[(i, j, event.channel)].append((u, v))

        result: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
        for key, pairs in buckets.items():
            us = np.stack([p[0] for p in pairs], axis=0)
            vs = np.stack([p[1] for p in pairs], axis=0)
            result[key] = (np.mean(us, axis=0), np.mean(vs, axis=0))

        return result

    def _extract_uv(self, event: ChannelEvent) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract source-side u and target-side v projections.

        Adapters may provide separate u_vector and v_vector in payload.
        If absent, falls back to event.vector for both sides.
        """
        d_c = CHANNEL_DIM[event.channel]

        if "u_vector" in event.payload and "v_vector" in event.payload:
            u = np.asarray(event.payload["u_vector"], dtype=np.float64).reshape(-1)
            v = np.asarray(event.payload["v_vector"], dtype=np.float64).reshape(-1)
        else:
            vec = np.asarray(event.vector, dtype=np.float64).reshape(-1)
            u, v = vec, vec

        if u.shape[0] != d_c or v.shape[0] != d_c:
            raise ValueError(
                f"ChannelEvent vector dimension mismatch: "
                f"channel={event.channel}, expected {d_c}, "
                f"got u={u.shape[0]}, v={v.shape[0]}"
            )

        return u, v


    def state_for(self, i: int, j: int, c: str) -> LICTEState | None:
        """Return the LICTEState for edge (i, j, c), or None if not feasible."""
        return self._states.get((i, j, c))

    def all_states(self) -> dict[tuple[int, int, str], LICTEState]:
        """Return all (i, j, c) -> LICTEState mappings (read-only use)."""
        return self._states


# ---------------------------------------------------------------------------
# Channel alpha values
# Must match the paper's Table (§4, channel-specific decay).
# ---------------------------------------------------------------------------

_ALPHA: dict[str, float] = {
    "comm": 0.5,   # messages immediately relevant — fast decay
    "mem":  0.1,   # writes persist long — slow decay
    "tool": 0.4,   # tool chains medium-range
    "exec": 0.6,   # execution signals ephemeral — fastest decay
}


def _channel_alpha(c: str) -> float:
    if c not in _ALPHA:
        raise ValueError(f"Unknown channel: {c!r}")
    return _ALPHA[c]