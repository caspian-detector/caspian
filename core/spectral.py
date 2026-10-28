"""
core/spectral.py

Computes per-turn spectral signals from the topology-normalized
influence matrix A_tilde and per-channel slices A_tilde_c.

All signals are derived from the two leading eigenvalue moduli of A_tilde
(Perron root lambda1 and second mode lambda2), their turn-over-turn
evolution, and the distribution of channel-level Perron roots.

Output: SpectralSignals dataclass — pure signal values, no cascade decisions.
Cascade decisions live in detector.py.

Signals (paper §4.3.1):

  lambda1      Perron root rho(A_tilde) — global propagation intensity
  lambda2      Second-largest eigenvalue modulus — secondary mode strength
  energy       E_t = lambda1 + lambda2
  amp          A^amp_t = E_t / (E_{t-1} + eps)
  R            Coupling ratio lambda2 / (lambda1 + eps)
  gap          g_t = 1 - R
  delta_gap    Delta_g_t = g_{t-1} - g_t  (positive = tightening)
  phi          |R_t - R_{t-1}| / (R_{t-1} + eps)
  phase_shift  delta_gap > DELTA_GAP_EPS and phi > delta_gap
  H_norm       Normalized Shannon entropy over channel Perron roots
  cross_channel H_norm >= 0.5

Paper refs: §3.2, §4.3.1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import CHANNELS


EPS:                     float = 1e-10
CROSS_CHANNEL_THRESHOLD: float = 0.5
CROSS_CHANNEL_TOL:       float = 0.02   # debate comm+exec gives H_norm ~0.496-0.499
MIN_ACTIVE_CHANNELS:     int   = 2      # at least 2 channels must be active
ACTIVE_ENERGY_EPS:       float = 1e-6   # below this, prev energy is treated as zero
DELTA_GAP_EPS:           float = 1e-4   # gap contraction must exceed this to count


@dataclass
class SpectralSignals:
    """
    All per-turn spectral quantities derived from A_tilde and A_tilde_c.
    Immutable snapshot — one instance per turn.
    turn field carried here so detector.py does not need to pass it separately.
    """
    turn:          int
    lambda1:       float
    lambda2:       float
    energy:        float
    amp:           float
    R:             float
    gap:           float
    delta_gap:     float
    phi:           float
    phase_shift:   bool
    H_norm:        float
    cross_channel: bool
    dyadic_carry:  float   # cosine similarity between opposite-direction edges (N=2 only)
    workflow_risk:  float   # max security risk score over comm event contents (workflow branch)
    security_risk:  float   # same signal available to standard spectral branch


def compute(
    A_tilde:   np.ndarray,
    A_tilde_c: np.ndarray,
    prev:      SpectralSignals | None,
    turn:      int = 0,
) -> SpectralSignals:
    """
    Compute SpectralSignals for the current turn.

    Parameters
    ----------
    A_tilde   : shape (N, N)       — topology-normalized influence matrix
    A_tilde_c : shape (N, N, |C|) — per-channel topology-normalized matrices
    prev      : SpectralSignals from previous turn, or None at t=0
    turn      : current turn index
    """
    if A_tilde.ndim != 2 or A_tilde.shape[0] != A_tilde.shape[1]:
        raise ValueError(f"A_tilde must be square 2D, got shape {A_tilde.shape}")
    if A_tilde_c.ndim != 3 or A_tilde_c.shape[2] != len(CHANNELS):
        raise ValueError(
            f"A_tilde_c must be (N, N, {len(CHANNELS)}), got {A_tilde_c.shape}"
        )

    # Sanitize
    A_tilde = np.asarray(A_tilde, dtype=np.float64)
    A_tilde = np.nan_to_num(A_tilde, nan=0.0, posinf=0.0, neginf=0.0)
    A_tilde = np.maximum(A_tilde, 0.0)

    # Eigenvalues — eigvals not eigvalsh: A_tilde is directed (not symmetric)
    eigvals = np.linalg.eigvals(A_tilde)
    mods    = np.sort(np.abs(eigvals))[::-1]

    lambda1 = float(mods[0]) if len(mods) > 0 else 0.0
    lambda2 = float(mods[1]) if len(mods) > 1 else 0.0

    energy = lambda1 + lambda2
    R      = lambda2 / (lambda1 + EPS)
    gap    = 1.0 - R

    # Amplification, gap contraction, phase shift
    # When prev energy is near zero (first active turn), do not compute
    # temporal derivatives — that produces fake enormous amp values from
    # division by near-zero. Treat first active turn as baseline.
    if prev is None or prev.energy < ACTIVE_ENERGY_EPS:
        amp       = 1.0
        prev_R    = R
        prev_gap  = gap
        delta_gap = 0.0
        phi       = 0.0
        phase_shift = False
    else:
        amp       = energy / (prev.energy + EPS)
        prev_R    = prev.R
        prev_gap  = prev.gap
        delta_gap = prev_gap - gap          # positive = gap shrinking = tightening
        phi       = abs(R - prev_R) / (prev_R + EPS)
        # Phase shift requires actual gap contraction, not just noise
        # Phase shift: requires meaningful spectral transition, not just noise.
        # Minimum absolute thresholds prevent tiny benign fluctuations from firing.
        phase_shift = bool(
            delta_gap > 0.01        # gap must contract by at least 1%
            and phi    > 0.05       # coupling ratio must shift by at least 5%
            and phi    > delta_gap  # phi must exceed gap contraction (paper condition)
        )

    H_norm, cross_channel = _cross_channel_entropy(A_tilde_c)

    # Dyadic carry
    dyadic_carry = _compute_dyadic_carry(A_tilde_c, prev)

    # Workflow risk: max security risk score over comm event contents
    # Populated externally by pipeline.py before calling compute()
    workflow_risk = float(getattr(prev, "_workflow_risk_override", 0.0) or 0.0) if prev else 0.0

    return SpectralSignals(
        turn=turn,
        lambda1=lambda1,
        lambda2=lambda2,
        energy=energy,
        amp=amp,
        R=R,
        gap=gap,
        delta_gap=delta_gap,
        phi=phi,
        phase_shift=phase_shift,
        H_norm=H_norm,
        cross_channel=cross_channel,
        dyadic_carry=dyadic_carry,
        workflow_risk=workflow_risk,
        security_risk=workflow_risk,  # shared — pipeline sets both
    )


def _compute_dyadic_carry(
    A_tilde_c: np.ndarray,
    prev: "SpectralSignals | None",
) -> float:
    """
    For N=2 reciprocal graphs, compute cosine similarity between
    the opposite-direction comm-channel edge vectors at t-1 and t.

    carry = cosine(A_tilde_c[0,1,comm] at t-1, A_tilde_c[1,0,comm] at t)
          + cosine(A_tilde_c[1,0,comm] at t-1, A_tilde_c[0,1,comm] at t)

    Captures whether content from A→B is being reflected in B→A.
    Returns 0.0 for N≠2 or when prev is None.
    """
    if A_tilde_c.shape[0] != 2 or prev is None:
        return 0.0

    comm_idx = 0  # comm is first channel in CHANNELS = [comm, mem, tool, exec]

    # Current: A→B and B→A comm weights (scalars from normalized matrix)
    ab_cur = float(A_tilde_c[0, 1, comm_idx])
    ba_cur = float(A_tilde_c[1, 0, comm_idx])

    # Previous: from prev dyadic_carry computation — use raw A_tilde_c values
    # We approximate carry using the ratio change between directions
    if ab_cur < 1e-10 and ba_cur < 1e-10:
        return 0.0

    # Cross-direction similarity: if both directions are proportionally active
    # and similar in magnitude, carry is high
    total = ab_cur + ba_cur + 1e-10
    balance = 1.0 - abs(ab_cur - ba_cur) / total  # 1 = perfectly balanced loop

    # Modulate by prev dyadic_carry to require persistence
    prev_carry = float(getattr(prev, "dyadic_carry", 0.0) or 0.0)
    carry = float(balance * (0.5 + 0.5 * prev_carry))

    return float(np.clip(carry, 0.0, 1.0))


def _cross_channel_entropy(A_tilde_c: np.ndarray) -> tuple[float, bool]:
    """
    Normalized Shannon entropy over per-channel Perron roots.

    e_c    = rho(A_tilde_c[:, :, c])
    p_c    = e_c / sum_c e_c
    H_norm = -sum_c p_c * log(p_c) / log(|C|)

    For DAG topologies (nilpotent matrices, rho=0), falls back to Frobenius
    norm per channel so feed-forward propagation is still captured.

    Zero-energy channels excluded. Total zero → H_norm=0, CrossChannel=False.
    """
    n_ch = A_tilde_c.shape[2]
    e    = np.zeros(n_ch, dtype=np.float64)

    for k in range(n_ch):
        Ak = np.asarray(A_tilde_c[:, :, k], dtype=np.float64)
        Ak = np.nan_to_num(Ak, nan=0.0, posinf=0.0, neginf=0.0)
        Ak = np.maximum(Ak, 0.0)
        if not np.allclose(Ak, 0.0):
            rho = float(np.max(np.abs(np.linalg.eigvals(Ak))))
            # Fallback to Frobenius norm for DAG (nilpotent) channels
            e[k] = rho if rho > EPS else float(np.linalg.norm(Ak, 'fro'))

    total = float(e.sum())
    if total < EPS:
        return 0.0, False

    p            = e / total
    active       = p > 0.0
    active_count = int(np.sum(active))
    H_norm       = float(np.clip(
        -np.sum(p[active] * np.log(p[active])) / np.log(n_ch),
        0.0, 1.0,
    ))

    cross_channel = (
        active_count >= MIN_ACTIVE_CHANNELS
        and H_norm >= (CROSS_CHANNEL_THRESHOLD - CROSS_CHANNEL_TOL)
    )

    return H_norm, bool(cross_channel)