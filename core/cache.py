"""
core/cache.py

Influence matrix cache — stores per-turn snapshots of A_tilde, A_tensor,
and A_raw so that attribution can operate over the detection window
[t_w, t_0] without recomputation.

What is stored per turn:
  A_tilde    shape (N, N)      — topology-normalized, for origin/amplifier
  A_tilde_c  shape (N, N, |C|) — per-channel normalized, for spine channels
  A_tensor   shape (N, N, |C|) — raw per-channel LI-CTE values
  A_raw      shape (N, N)      — unnormalized channel sum, for bridge

What is derived on retrieval:
  A_bar_max     elementwise max of A_tilde over [t_w, t_0]   (paper §4.4)
  A_ch_bar_max  elementwise max of A_tilde_c over [t_w, t_0]

Cache is bounded: evicts turns older than MAX_CACHE_TURNS = MAX_WINDOW + 2.

Paper refs: §4.4, §3.4
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .detector import MAX_WINDOW

MAX_CACHE_TURNS: int = MAX_WINDOW + 2


@dataclass
class TurnSnapshot:
    turn:      int
    A_tilde:   np.ndarray   # (N, N)
    A_tilde_c: np.ndarray   # (N, N, |C|)
    A_tensor:  np.ndarray   # (N, N, |C|)
    A_raw:     np.ndarray   # (N, N)


class InfluenceCache:
    """
    Bounded ordered cache of TurnSnapshots keyed by turn index.
    Evicts turns older than MAX_CACHE_TURNS on each push.
    """

    def __init__(self) -> None:
        self._store: OrderedDict[int, TurnSnapshot] = OrderedDict()

    def push(self, snapshot: TurnSnapshot) -> None:
        self._store[snapshot.turn] = snapshot
        cutoff   = snapshot.turn - MAX_CACHE_TURNS
        to_evict = [t for t in self._store if t < cutoff]
        for t in to_evict:
            del self._store[t]

    def window(self, t_w: int, t_0: int) -> list[TurnSnapshot]:
        """Snapshots for turns in [t_w, t_0], in order. Raises if incomplete."""
        missing = [t for t in range(t_w, t_0 + 1) if t not in self._store]
        if missing:
            raise ValueError(
                f"Missing cached turns {missing} for window [{t_w}, {t_0}]. "
                f"Available: {sorted(self._store.keys())}"
            )
        return [self._store[t] for t in range(t_w, t_0 + 1)]

    def A_bar_max(self, t_w: int, t_0: int) -> np.ndarray:
        """
        Elementwise max of A_tilde over [t_w, t_0].
        A_bar_max[i,j] = max_τ A_tilde_τ[i,j]. Paper §4.4.
        """
        snaps = self.window(t_w, t_0)
        if not snaps:
            raise ValueError(
                f"No cached snapshots for window [{t_w}, {t_0}]. "
                f"Available: {sorted(self._store.keys())}"
            )
        return np.maximum.reduce([s.A_tilde for s in snaps])

    def A_ch_bar_max(self, t_w: int, t_0: int) -> np.ndarray:
        """
        Elementwise max of A_tilde_c over [t_w, t_0]. Shape (N, N, |C|).
        Used for dominant channel attribution per spine. Paper §4.4.
        """
        snaps = self.window(t_w, t_0)
        if not snaps:
            raise ValueError(f"No cached snapshots for window [{t_w}, {t_0}].")
        return np.maximum.reduce([s.A_tilde_c for s in snaps])

    def A_tildes(self, t_w: int, t_0: int) -> list[np.ndarray]:
        return [s.A_tilde for s in self.window(t_w, t_0)]

    def A_raws(self, t_w: int, t_0: int) -> list[np.ndarray]:
        return [s.A_raw for s in self.window(t_w, t_0)]

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, turn: int) -> bool:
        return turn in self._store