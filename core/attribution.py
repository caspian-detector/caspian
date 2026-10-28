"""
core/attribution.py

Cascade attribution via spectral flow decomposition.

Operates entirely on matrices retrieved from cache.py after cascade
detection.  Requires no additional parameters beyond those already
defined in the detection pipeline.

Identifies four components of a detected cascade (paper §4.4):

  Origin
      Agent with the strongest total outgoing influence at cascade onset t_w.
      Uses A_tilde at t_w (topology-normalized) so hub degree does not
      spuriously inflate the score.
      i_origin = argmax_i  sum_j  A_tilde_tw[i, j]

  Amplifier
      Agent that most strongly reinforces propagation relative to what it
      receives, accumulated over the confirmation window [t_w, t_0].
      i_amp = argmax_i  sum_τ  (outflow_i(τ) / (inflow_i(τ) + eps))

  Bridge
      Agent that both receives and redistributes the largest volume of
      influence over [t_w, t_0].  Computed on A_raw (unnormalized) to
      preserve true interaction volume.
      i_bridge = argmax_i  sum_τ  (sum_j A_raw_τ[i,j]) * (sum_k A_raw_τ[k,i])

  Top-K propagation spines
      K paths ranked by weakest-link strength on A_bar_max.
      Path length bounded by graph diameter.
      Dominant channel per spine from A_ch_bar_max.

Paper refs: §3.4, §4.4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import CHANNELS
from .cache import InfluenceCache


@dataclass
class AttributionResult:
    """
    origin         : index of agent where cascade begins
    amplifier      : index of agent that most strongly reinforces propagation
    bridge         : index of agent that relays influence across the network
    spines         : top-K propagation paths, each a list of agent indices
    spine_channels : dominant interaction channel for each spine
    t_w            : watch onset turn
    t_0            : cascade declaration turn
    """
    origin:         int
    amplifier:      int
    bridge:         int
    spines:         list[list[int]]
    spine_channels: list[str]
    t_w:            int
    t_0:            int


def _origin(A_tilde_tw: np.ndarray) -> int:
    """
    argmax_i sum_j A_tilde_tw[i, j]
    Topology-normalized so hub degree does not spuriously inflate score.
    Paper §4.4 eq.10.
    """
    return int(np.argmax(A_tilde_tw.sum(axis=1)))


def _amplifier(A_tildes: list[np.ndarray], eps: float = 1e-10) -> int:
    """
    argmax_i sum_τ (outflow_i(τ) / (inflow_i(τ) + eps))
    Ratio > 1 = net amplification. Paper §4.4 eq.11.
    """
    if not A_tildes:
        raise ValueError("A_tildes is empty — cannot compute amplifier.")
    N      = A_tildes[0].shape[0]
    scores = np.zeros(N, dtype=np.float64)
    for A in A_tildes:
        scores += A.sum(axis=1) / (A.sum(axis=0) + eps)
    return int(np.argmax(scores))


def _bridge(A_raws: list[np.ndarray]) -> int:
    """
    argmax_i sum_τ outflow_raw_i(τ) * inflow_raw_i(τ)
    Uses A_raw to preserve true interaction volume. Paper §4.4 eq.12.
    """
    if not A_raws:
        raise ValueError("A_raws is empty — cannot compute bridge.")
    N      = A_raws[0].shape[0]
    scores = np.zeros(N, dtype=np.float64)
    for A in A_raws:
        scores += A.sum(axis=1) * A.sum(axis=0)
    return int(np.argmax(scores))


def _top_k_spines(
    A_bar_max: np.ndarray,
    diameter:  int,
    k:         int,
) -> list[list[int]]:
    """
    Top-K simple directed paths ranked by weakest-link strength on A_bar_max.
    DFS with bottleneck tracking, path length bounded by diameter.
    Paper §4.4 eq.13.
    """
    N       = A_bar_max.shape[0]
    results: list[tuple[float, list[int]]] = []

    for src in range(N):
        stack: list[tuple[int, list[int], float]] = [(src, [src], float("inf"))]
        while stack:
            node, path, bottleneck = stack.pop()
            if len(path) >= 2:
                results.append((bottleneck, path[:]))
            if len(path) > diameter:
                continue
            for nxt in range(N):
                w = float(A_bar_max[node, nxt])
                if w <= 0.0 or nxt in path:
                    continue
                stack.append((nxt, path + [nxt], min(bottleneck, w)))

    if not results:
        return []

    results.sort(key=lambda x: x[0], reverse=True)
    seen:   set[tuple[int, ...]] = set()
    spines: list[list[int]]      = []
    for _, path in results:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            spines.append(path)
        if len(spines) >= k:
            break
    return spines


def _dominant_channel(spine: list[int], A_ch_bar_max: np.ndarray) -> str:
    """
    argmax_c sum_{(i,j) in spine} A_ch_bar_max[i,j,c]
    Returns "unknown" if all channel scores are zero.
    Paper §4.4.
    """
    n_ch           = A_ch_bar_max.shape[2]
    channel_scores = np.zeros(n_ch, dtype=np.float64)
    for i, j in zip(spine[:-1], spine[1:]):
        channel_scores += A_ch_bar_max[i, j, :]
    if channel_scores.sum() <= 0.0:
        return "unknown"
    return CHANNELS[int(np.argmax(channel_scores))]


def attribute(
    cache:    InfluenceCache,
    t_w:      int,
    t_0:      int,
    diameter: int,
    k:        int = 3,
) -> AttributionResult:
    """
    Full cascade attribution from cached influence matrices.

    Parameters
    ----------
    cache    : InfluenceCache populated during detection
    t_w      : watch onset turn
    t_0      : cascade declaration turn
    diameter : G0 graph diameter — bounds spine length
               Use max(1, N - 1) if G0 is disconnected.
    k        : number of top spines (default 3)
    """
    snaps = cache.window(t_w, t_0)
    if not snaps:
        raise ValueError(
            f"No cached snapshots for attribution window [{t_w}, {t_0}]."
        )

    A_tildes     = cache.A_tildes(t_w, t_0)
    A_raws       = cache.A_raws(t_w, t_0)
    A_bar_max    = cache.A_bar_max(t_w, t_0)
    A_ch_bar_max = cache.A_ch_bar_max(t_w, t_0)

    spines         = _top_k_spines(A_bar_max, diameter=diameter, k=k)
    spine_channels = [_dominant_channel(sp, A_ch_bar_max) for sp in spines]

    return AttributionResult(
        origin=_origin(snaps[0].A_tilde),
        amplifier=_amplifier(A_tildes),
        bridge=_bridge(A_raws),
        spines=spines,
        spine_channels=spine_channels,
        t_w=t_w,
        t_0=t_0,
    )