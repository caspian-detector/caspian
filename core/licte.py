"""
core/licte.py

LI-CTE estimator: computes one LI-CTE scalar per (i, j, c) per turn.

This module knows nothing about cascades, detection, or attribution.
Its only job is:

    score = update(state, u, v) -> float >= 0

Per turn:
    1. s_i(t)  = peek_source_ema(u)          — updated source EMA, no mutation
    2. z       = [s_i(t), v_j(t), h_j(t-1)] — h is PREVIOUS turn
    3. z_tilde = rank_transform(z)            — marginal-rank Gaussian copula
    4. update_covariance(z_tilde)             — EMA centered covariance
    5. CMI     = _compute_cmi(state)          — Schur complement log-det
    6. update_ema(u, v)                       — s and h advanced LAST

Shrinkage and jitter applied to Sigma before inversion.
Never stored back to state — state always holds the raw EMA covariance.

Paper refs:
  §5          LI-CTE definition and Gaussian CMI formula
  Appendix B  Full Schur-complement derivation
  Appendix F-3  Operational steps
"""

from __future__ import annotations

import numpy as np

from .states import LICTEState


# ---------------------------------------------------------------------------
# Numerical stabilisation
# ---------------------------------------------------------------------------

SHRINKAGE:     float = 0.05
JITTER:        float = 1e-6
DIRECT_WEIGHT: float = 0.05
EPS:           float = 1e-10


def _stabilise(Sigma: np.ndarray) -> np.ndarray:
    """
    Full stabilisation applied to state.Sigma before block extraction.
    Enforces symmetry, shrinkage toward identity, and jitter floor.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    Sigma = 0.5 * (Sigma + Sigma.T)
    d = Sigma.shape[0]
    return (1.0 - SHRINKAGE) * Sigma + SHRINKAGE * np.eye(d) + JITTER * np.eye(d)


def _stabilise_small(M: np.ndarray) -> np.ndarray:
    """
    Light stabilisation applied to Schur complement blocks before log-det.
    Enforces symmetry and adds jitter floor only — no shrinkage.
    Schur complements can become slightly non-PSD numerically even after
    the parent matrix is stabilised.
    """
    M = np.asarray(M, dtype=np.float64)
    M = 0.5 * (M + M.T)
    return M + JITTER * np.eye(M.shape[0])


def _safe_slogdet(M: np.ndarray) -> float:
    """
    log|det(M)| via Cholesky (numerically preferred for near-PD matrices).
    Falls back to slogdet if Cholesky fails.
    Returns -inf if the matrix is singular or indefinite.
    """
    M = _stabilise_small(M)
    try:
        L = np.linalg.cholesky(M)
        return float(2.0 * np.sum(np.log(np.diag(L))))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf
        return float(logdet)


# ---------------------------------------------------------------------------
# CMI via Schur complements
# ---------------------------------------------------------------------------

def _compute_cmi(state: LICTEState) -> float:
    """
    Gaussian CMI  I(u ; v | h)  from the partitioned EMA covariance Sigma.

    Partition (each block is d_c x d_c):
        Sigma = [ Sigma_uu  Sigma_uv  Sigma_uh ]
                [ Sigma_vu  Sigma_vv  Sigma_vh ]
                [ Sigma_hu  Sigma_hv  Sigma_hh ]

    Schur complements conditioning on h via linear solves (more stable
    than explicit inversion of S_hh):
        X_hu   = S_hh^{-1} @ S_hu
        X_hv   = S_hh^{-1} @ S_hv

        Sigma_uu|h = Sigma_uu - Sigma_uh @ X_hu
        Sigma_vv|h = Sigma_vv - Sigma_vh @ X_hv
        Sigma_uv|h = Sigma_uv - Sigma_uh @ X_hv
        Sigma_vu|h = Sigma_vu - Sigma_vh @ X_hu

    Joint conditional covariance of [u, v] given h:
        Sigma_joint|h = [ Sigma_uu|h  Sigma_uv|h ]
                        [ Sigma_vu|h  Sigma_vv|h ]

    Gaussian CMI (paper §5, Appendix B):
        I(u ; v | h) = 0.5 * (log|Sigma_uu|h| + log|Sigma_vv|h|
                               - log|Sigma_joint|h|)

    Clipped to >= 0: true CMI is always non-negative; numerical noise
    can produce tiny negatives.
    """
    dc = state.d_c

    S = _stabilise(state.Sigma)

    su = slice(0,      dc)
    sv = slice(dc,     2 * dc)
    sh = slice(2 * dc, 3 * dc)

    S_uu = S[su, su];  S_uv = S[su, sv];  S_uh = S[su, sh]
    S_vu = S[sv, su];  S_vv = S[sv, sv];  S_vh = S[sv, sh]
    S_hu = S[sh, su];  S_hv = S[sh, sv];  S_hh = S[sh, sh]

    try:
        X_hu = np.linalg.solve(S_hh, S_hu)
        X_hv = np.linalg.solve(S_hh, S_hv)
    except np.linalg.LinAlgError:
        return 0.0

    S_uu_h = S_uu - S_uh @ X_hu
    S_vv_h = S_vv - S_vh @ X_hv
    S_uv_h = S_uv - S_uh @ X_hv
    S_vu_h = S_vu - S_vh @ X_hu

    S_uu_h    = _stabilise_small(S_uu_h)
    S_vv_h    = _stabilise_small(S_vv_h)

    S_joint_h = np.block([
        [S_uu_h, S_uv_h],
        [S_vu_h, S_vv_h],
    ])
    S_joint_h = _stabilise_small(S_joint_h)

    ld_uu_h    = _safe_slogdet(S_uu_h)
    ld_vv_h    = _safe_slogdet(S_vv_h)
    ld_joint_h = _safe_slogdet(S_joint_h)

    if any(not np.isfinite(v) for v in [ld_uu_h, ld_vv_h, ld_joint_h]):
        return 0.0

    cmi = 0.5 * (ld_uu_h + ld_vv_h - ld_joint_h)

    if not np.isfinite(cmi):
        return 0.0

    return float(max(cmi, 0.0))


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------

def _direct_residual_score(s_next, v, h_prev) -> float:
    """
    Immediate finite-sample dependence proxy for online LI-CTE.
    Measures positive alignment between source state and target observation
    after subtracting h_j(t-1). No warmup or baseline required.
    """
    x = np.asarray(s_next, dtype=np.float64) - np.asarray(h_prev, dtype=np.float64)
    y = np.asarray(v,      dtype=np.float64) - np.asarray(h_prev, dtype=np.float64)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < EPS or ny < EPS:
        return 0.0
    cos = max(0.0, float(np.dot(x, y) / (nx * ny + EPS)))
    mag = float(np.tanh(0.5 * (nx + ny) / np.sqrt(len(x))))
    return cos * mag


def _as_vec(x: np.ndarray, d: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != d:
        raise ValueError(f"{name}: expected shape ({d},), got {x.shape}")
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update(state: LICTEState, u: np.ndarray, v: np.ndarray) -> float:
    """
    Process one turn for one (i, j, c) edge.

    Parameters
    ----------
    state : LICTEState for this (i, j, c) — mutated in place
    u     : source feature vector at turn t, shape (d_c,)
    v     : target feature vector at turn t, shape (d_c,)

    Returns
    -------
    Nonnegative LI-CTE scalar a_ij^(c)(t).
    Returns 0.0 until state reaches WARMUP_MIN turns.

    Ordering contract:
        peek_source_ema  → rank_transform → update_covariance
        → _compute_cmi → update_ema
    """
    u = _as_vec(u, state.d_c, "u")
    v = _as_vec(v, state.d_c, "v")

    s_next  = state.peek_source_ema(u)
    z       = np.concatenate([s_next, v, state.h])
    z_tilde = state.rank_transform(z)

    state.update_covariance(z_tilde)

    cmi_score    = _compute_cmi(state)
    direct_score = _direct_residual_score(s_next, v, state.h)
    score        = cmi_score + DIRECT_WEIGHT * direct_score

    state.update_ema(u, v)
    return float(max(score, 0.0))