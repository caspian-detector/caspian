"""
core/states.py

Per-(i, j, c) EMA state maintained by the LI-CTE estimator.

Each LICTEState instance holds everything needed to produce one
LI-CTE scalar at turn t without any external buffer:

  s_i^(c)(t)    source EMA over u vectors              shape (d_c,)
  h_j^(c)(t)    conditioning EMA over v vectors         shape (d_c,)
  mu            EMA mean of z-tilde for centering        shape (3*d_c,)
  Sigma         EMA centered covariance over z-tilde     shape (3*d_c, 3*d_c)
  ecdfs         P2-style multi-quantile streaming ECDF   one per z-dimension

Correct per-turn ordering (enforced by licte.py, documented here):
  1. receive u(t), v(t)
  2. form z = [u(t), v(t), h_j(t-1)]   <- h is PREVIOUS turn
  3. rank-transform z -> z_tilde
  4. update_covariance(z_tilde)
  5. CMI computed in licte.py from Sigma
  6. update_ema(u, v)                   <- h updated AFTER CMI

Paper refs:
  §4   EMA source/history formulation
  §5   LI-CTE, rank-copula, Schur complement CMI
  Appendix B  full CMI derivation
  Appendix F-3  operational steps
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Warmup thresholds
# ---------------------------------------------------------------------------

# Minimum turns before the exact-buffer ECDF is reliable enough for detection.
# Below this, CMI estimates exist but should be treated as low-confidence.
WARMUP_MIN: int = 0   # no warmup — paper §4.3.2 fires from turn 1

# Turns after which P2-style multi-quantile markers are fully initialised.
# Before this, we use exact order statistics on the buffered observations,
# which is actually more accurate for small n, so estimates are not discarded.
WARMUP_FULL: int = 101

# Number of percentile markers for the P2-style streaming ECDF.
# Markers sit at p = 0.00, 0.01, ..., 1.00 (101 evenly spaced quantiles).
N_MARKERS: int = WARMUP_FULL

_DESIRED_P: np.ndarray = np.linspace(0.0, 1.0, N_MARKERS)


# ---------------------------------------------------------------------------
# P2-style multi-quantile streaming ECDF
# ---------------------------------------------------------------------------

class StreamingECDF:
    """
    Single-dimension streaming empirical CDF approximation.

    Uses a P2-style multi-quantile marker scheme with 101 markers at
    percentiles p = 0.00, 0.01, ..., 1.00.  O(1) memory per dimension,
    no stored observations after initialisation.

    Before N_MARKERS observations have been seen, falls back to exact
    order statistics on a small buffer, which is more accurate for small n
    and means pre-warmup CMI estimates are not pure noise.

    Reference: Jain & Chlamtac (1985), "The P2 Algorithm for Dynamic
    Calculation of Quantiles and Histograms Without Storing Observations."
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._q = np.zeros(N_MARKERS, dtype=np.float64)   # marker heights
        self._n_pos = np.arange(N_MARKERS, dtype=np.float64)  # actual positions
        self._initialized: bool = False
        self._buf: list[float] = []   # exact buffer pre-initialisation

    def update(self, x: float) -> None:
        self._n += 1

        if not self._initialized:
            self._buf.append(x)
            if len(self._buf) >= N_MARKERS:
                self._buf.sort()
                self._q = np.array(self._buf, dtype=np.float64)
                self._n_pos = np.arange(N_MARKERS, dtype=np.float64)
                self._initialized = True
                self._buf = []
            return

        # --- P2 update ---
        # Step 1: find cell k such that q[k] <= x < q[k+1]
        if x < self._q[0]:
            self._q[0] = x
            k = 0
        elif x >= self._q[-1]:
            self._q[-1] = x
            k = N_MARKERS - 2
        else:
            k = int(np.searchsorted(self._q, x, side='right')) - 1
            k = int(np.clip(k, 0, N_MARKERS - 2))

        # Step 2: increment positions of markers k+1 .. end
        self._n_pos[k + 1:] += 1.0

        # Step 3: desired positions for this n
        desired = _DESIRED_P * (self._n - 1)

        # Step 4: adjust marker heights via piecewise-parabolic interpolation
        for i in range(1, N_MARKERS - 1):
            d = desired[i] - self._n_pos[i]
            ni  = self._n_pos[i]
            nip = self._n_pos[i + 1]
            nim = self._n_pos[i - 1]
            qi  = self._q[i]
            qp  = self._q[i + 1]
            qm  = self._q[i - 1]

            if (d >= 1.0 and nip - ni > 1) or (d <= -1.0 and nim - ni < -1):
                sign_d = 1.0 if d > 0 else -1.0
                # Piecewise-parabolic (P2) interpolation
                q_new = qi + sign_d / (nip - nim) * (
                    (ni - nim + sign_d) * (qp - qi) / (nip - ni)
                    + (nip - ni - sign_d) * (qi - qm) / (ni - nim)
                )
                # Fall back to linear if parabolic overshoots neighbours
                if qm < q_new < qp:
                    self._q[i] = q_new
                else:
                    idx = i + int(sign_d)
                    self._q[i] = qi + sign_d * (
                        self._q[idx] - qi
                    ) / (self._n_pos[idx] - ni)

                self._n_pos[i] += sign_d

    def cdf(self, x: float) -> float:
        """
        Returns F̂(x) ∈ (eps, 1-eps) so Φ⁻¹ is always finite.
        Uses exact order statistics before P2 is initialised.
        """
        eps = 1e-6

        if not self._initialized:
            if not self._buf:
                return 0.5
            arr = np.sort(self._buf)
            # Continuity-corrected rank to keep result in (0, 1)
            rank = int(np.searchsorted(arr, x, side='right'))
            return float(np.clip(
                (rank + 0.5) / (len(arr) + 1), eps, 1.0 - eps
            ))

        if x <= self._q[0]:
            return eps
        if x >= self._q[-1]:
            return 1.0 - eps

        idx = int(np.searchsorted(self._q, x, side='right')) - 1
        idx = int(np.clip(idx, 0, N_MARKERS - 2))

        lo, hi   = self._q[idx], self._q[idx + 1]
        p_lo, p_hi = _DESIRED_P[idx], _DESIRED_P[idx + 1]

        if hi == lo:
            return float(np.clip(p_lo, eps, 1.0 - eps))

        p = p_lo + (x - lo) / (hi - lo) * (p_hi - p_lo)
        return float(np.clip(p, eps, 1.0 - eps))


# ---------------------------------------------------------------------------
# Per-(i, j, c) LI-CTE state
# ---------------------------------------------------------------------------

@dataclass
class LICTEState:
    """
    Complete running state for one (source agent i, target agent j, channel c).

    Parameters
    ----------
    channel : channel name ("comm" | "mem" | "tool" | "exec")
    d_c     : feature dimension for this channel (from config)
    alpha   : EMA decay for this channel (from config)

    State fields (all updated incrementally, no external buffer required)
    ------------
    s     : source EMA s_i^(c)(t),              shape (d_c,)
    h     : conditioning EMA h_j^(c)(t-1),      shape (d_c,)
            NOTE: h always holds the value from the PREVIOUS turn.
            licte.py must use h BEFORE calling update_ema().
    mu    : EMA mean of z_tilde for centering,   shape (3*d_c,)
    Sigma : EMA centered covariance of z_tilde,  shape (3*d_c, 3*d_c)
            Initialised to identity (paper §5).
    ecdfs : P2-style streaming ECDF per z-dim,   list of length 3*d_c
    t     : number of turns seen so far
    """

    channel: str
    d_c:     int
    alpha:   float

    s:     np.ndarray          = field(init=False)
    h:     np.ndarray          = field(init=False)
    mu:    np.ndarray          = field(init=False)
    Sigma: np.ndarray          = field(init=False)
    ecdfs: list[StreamingECDF] = field(init=False)
    t:     int                 = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.s     = np.zeros(self.d_c,           dtype=np.float64)
        self.h     = np.zeros(self.d_c,           dtype=np.float64)
        self.mu    = np.zeros(3 * self.d_c,       dtype=np.float64)
        self.Sigma = np.eye(3 * self.d_c,         dtype=np.float64)
        self.ecdfs = [StreamingECDF() for _ in range(3 * self.d_c)]

    # ------------------------------------------------------------------
    # Called by licte.py in the correct order each turn
    # ------------------------------------------------------------------

    def peek_source_ema(self, u: np.ndarray) -> np.ndarray:
        """
        Returns the source EMA that would result from incorporating u(t),
        without mutating state.

        Lets licte.py form z = [s_i(t), v_j(t), h_j(t-1)] using the
        updated source EMA while h_j(t-1) is still held fixed until
        after CMI is computed.
        """
        return self.alpha * u.astype(np.float64) + (1.0 - self.alpha) * self.s

    def rank_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Marginal-rank Gaussian copula transform (paper §5):
            z̃_l = Φ⁻¹(F̂_l(z_l))   for each dimension l

        Updates each StreamingECDF with z[l] then queries it,
        so the CDF includes the current observation.
        """
        from scipy.stats import norm as sp_norm

        z = z.astype(np.float64)
        expected = 3 * self.d_c
        if z.shape[0] != expected:
            raise ValueError(
                f"rank_transform: expected z with shape ({expected},), got {z.shape}"
            )

        z_tilde = np.empty_like(z)
        for l, ecdf in enumerate(self.ecdfs):
            if ecdf._n == 0:
                # First observation: seed ECDF and use the raw z value
                # clipped to a standard normal range so Sigma gets structure.
                ecdf.update(float(z[l]))
                z_tilde[l] = float(np.clip(z[l], -3.0, 3.0))
            else:
                # Query CDF BEFORE updating — rank among previous observations.
                p = ecdf.cdf(float(z[l]))
                ecdf.update(float(z[l]))
                z_tilde[l] = sp_norm.ppf(p)
        return z_tilde

    def update_covariance(self, z_tilde: np.ndarray) -> None:
        """
        EMA centered covariance update (paper §5):
            mu(t)    = alpha * z_tilde + (1-alpha) * mu(t-1)
            Sigma(t) = alpha * outer(z_tilde - mu_old, z_tilde - mu_new)
                     + (1-alpha) * Sigma(t-1)

        Centering uses the old mu before the new observation shifts it,
        which gives an unbiased EMA covariance estimate.

        Symmetry is enforced explicitly to guard against float drift.
        Shrinkage and jitter are applied in licte.py before Schur inversion.
        """
        old_mu  = self.mu.copy()
        self.mu = self.alpha * z_tilde + (1.0 - self.alpha) * self.mu

        d_old = z_tilde - old_mu
        d_new = z_tilde - self.mu

        self.Sigma = (
            self.alpha * np.outer(d_old, d_new)
            + (1.0 - self.alpha) * self.Sigma
        )
        # Enforce symmetry
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

    def update_ema(self, u: np.ndarray, v: np.ndarray) -> None:
        """
        Update source EMA s and conditioning EMA h (paper §4):
            s(t) = alpha * u(t) + (1-alpha) * s(t-1)
            h(t) = alpha * v(t) + (1-alpha) * h(t-1)

        MUST be called AFTER rank_transform, update_covariance, and
        CMI computation in licte.py, so that h holds h_j(t-1) during
        the current turn's CMI calculation.
        """
        self.s = self.alpha * u + (1.0 - self.alpha) * self.s
        self.h = self.alpha * v + (1.0 - self.alpha) * self.h
        self.t += 1

    # ------------------------------------------------------------------
    # Warmup flags
    # ------------------------------------------------------------------

    @property
    def minimally_warmed(self) -> bool:
        """
        True after WARMUP_MIN turns.
        CMI estimates exist and are usable for detection.
        The exact-buffer ECDF is already more accurate than P2 for small n,
        so estimates are not discarded before full warmup.
        """
        return self.t >= WARMUP_MIN

    @property
    def fully_warmed(self) -> bool:
        """
        True after WARMUP_FULL turns (P2 markers fully initialised).
        Use this flag only for confidence reporting, not for gating detection.
        """
        return self.t >= WARMUP_FULL