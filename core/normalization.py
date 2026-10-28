"""
core/normalization.py

Normalizes the raw influence tensor A_tensor into:

  A_matrix  — channel-normalized and l1-fused, shape (N, N)
  A_tilde   — topology-normalized, shape (N, N)
               Primary input to spectral detection and attribution.
  A_tilde_c — per-channel topology-normalized tensor, shape (N, N, |C|)
               Used for cross-channel entropy H_norm in spectral.py.

Normalization steps:

  Step 1 — Per-channel normalization to [0, 1]:
      A_norm[:, :, c] = A_tensor[:, :, c] / (max(A_tensor[:, :, c]) + eps)

  Step 2 — l1 channel fusion:
      A_matrix[i, j] = sum_c A_norm[i, j, c]

  Step 3 — Topology degree-aware normalization:
      r_i = sum_j A_matrix[i, j]
      c_j = sum_i A_matrix[i, j]
      A_tilde[i, j] = A_matrix[i, j] / (sqrt(r_i * c_j) + eps)

  Per-channel A_tilde_c applies topology normalization independently
  to each normalized channel slice.

Note: A_raw from tensor.py is kept separately for bridge attribution.
A_matrix must not replace it — A_matrix is channel-normalized and loses
true interaction volume that bridge scoring requires.

Paper refs: §4.2, §4.3, §4.4
"""

from __future__ import annotations

import numpy as np

from .events import CHANNELS

EPS: float = 1e-10


def normalize(
    A_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    A_tensor : shape (N, N, |C|), dtype float64, all entries >= 0

    Returns
    -------
    A_matrix  : shape (N, N)        — fused, not topology-corrected
    A_tilde   : shape (N, N)        — topology-normalized
    A_tilde_c : shape (N, N, |C|)  — per-channel topology-normalized
    """
    if A_tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {A_tensor.shape}")
    if A_tensor.shape[2] != len(CHANNELS):
        raise ValueError(
            f"Expected {len(CHANNELS)} channels, got {A_tensor.shape[2]}"
        )

    A_tensor = np.asarray(A_tensor, dtype=np.float64)
    A_tensor = np.nan_to_num(A_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    A_tensor = np.maximum(A_tensor, 0.0)

    # Step 1: per-channel normalize to [0, 1]
    A_norm = np.zeros_like(A_tensor, dtype=np.float64)
    for k in range(len(CHANNELS)):
        ch     = A_tensor[:, :, k]
        ch_max = float(np.max(ch))
        if ch_max > EPS:
            A_norm[:, :, k] = ch / (ch_max + EPS)

    # Step 2: l1 fuse
    A_matrix = np.maximum(A_norm.sum(axis=2), 0.0)

    # Step 3: topology normalize
    A_tilde = _topo_normalize(A_matrix)

    # Per-channel topo-normalized for cross-channel entropy
    A_tilde_c = np.stack(
        [_topo_normalize(A_norm[:, :, k]) for k in range(len(CHANNELS))],
        axis=2,
    )

    return A_matrix, A_tilde, A_tilde_c


def _topo_normalize(A: np.ndarray) -> np.ndarray:
    """
    A_tilde[i, j] = A[i, j] / (sqrt(r_i * c_j) + eps)

    Denominator is sqrt(outer(r, c)) + eps, not sqrt(outer(r, c) + eps),
    to avoid distorting weak but nonzero edges.
    nan_to_num guards zero-row/zero-column agents.
    """
    A     = np.asarray(A, dtype=np.float64)
    A     = np.maximum(A, 0.0)
    r     = A.sum(axis=1)
    c     = A.sum(axis=0)
    denom = np.sqrt(np.outer(r, c)) + EPS
    out   = A / denom
    out   = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(out, 0.0)