"""RINSE power spectrum: intensity-weighted reciprocal-space descriptor.

Descriptor formulation
----------------------
Given a set of reflections {G_i} with intensities {I_i}, compute:

    A_{nlm} = Σ_i  I_i · R_n(|G_i|) · Y_l^m(Ĝ_i)

where R_n is the chosen radial basis and Y_l^m are real spherical harmonics.

The rotationally invariant power spectrum is:

    p_{nl} = Σ_{m=-l}^{l}  |A_{nlm}|²

Because the intensity field is centrosymmetric (I(-G) = I(G), Friedel's law),
contributions from odd-l harmonics cancel exactly.  Only even l contribute:

    l ∈ {0, 2, 4, …, 2*(L-1)}  for L angular levels.

Default parameters:
    n_max = 8  → radial indices 0 … 7
    l_max = 16  → angular levels: l ∈ {0, 2, 4, …, 30}
    Output: (8, 16) matrix  → flattened to 128-element vector
            axis-0 = radial index n
            axis-1 = angular level index (l//2)

Spherical harmonics convention
-------------------------------
Real spherical harmonics Y_lm are computed via scipy.special.lpmv (associated
Legendre ufunc) combined with a pre-computed cos(mφ)/sin(mφ) table.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from math import pi as _pi
from math import sqrt as _msqrt

import numpy as np
from numpy.typing import NDArray

from ._radial_basis import RadialBasisType, evaluate_radial_basis
from ._structure_factors import ReflectionList

# ---------------------------------------------------------------------------
# Descriptor parameters
# ---------------------------------------------------------------------------


@dataclass
class RinseParams:
    """Hyper-parameters for the RINSE descriptor.

    Attributes
    ----------
    n_max:
        Number of radial basis functions (n = 0 … n_max-1).  Default 8.
    l_max:
        Number of angular levels (ℓ = 0, 2, 4, …, 2*(l_max-1)).  Default 16.
        Total descriptor length = n_max * l_max.
    sin_theta_over_lambda_max:
        Resolution cutoff.  Default 0.6 Å⁻¹ → |G| ≤ 1.2 Å⁻¹.
    radial_basis:
        ``"chebyshev"`` (default), ``"bessel"``, or
        ``"smooth_shells_cw"``/``"smooth_shells_nl"``.
    """

    n_max: int = 8
    l_max: int = 16
    sin_theta_over_lambda_max: float = 0.6
    radial_basis: RadialBasisType = "chebyshev"

    @property
    def q_max(self) -> float:
        """|G| cutoff in Å⁻¹."""
        return 2.0 * self.sin_theta_over_lambda_max

    @property
    def l_values(self) -> list[int]:
        """Even ℓ values used."""
        return [2 * k for k in range(self.l_max)]

    @property
    def descriptor_length(self) -> int:
        """Total number of elements in the flattened descriptor."""
        return self.n_max * self.l_max

    @property
    def descriptor_shape(self) -> tuple[int, int]:
        """Shape of the 2-D descriptor matrix (n_max, l_max)."""
        return (self.n_max, self.l_max)


# ---------------------------------------------------------------------------
# Pre-computed spherical harmonic cache
# ---------------------------------------------------------------------------


class _SphHarmCache:
    """Evaluate real spherical harmonics Y_lm(θ, φ) for a fixed set of
    Cartesian unit vectors.

    We cache by (l_values, unit_vectors) so repeated calls for the same
    reflection list are free.
    """

    def __init__(self, unit_vecs: NDArray[np.float64], l_values: list[int]) -> None:
        """Pre-compute all Y_lm values.

        Parameters
        ----------
        unit_vecs : (M, 3) float64  normalised Cartesian vectors
        l_values  : list of even ints

        Stored as dict l → real_Ylm array of shape (M, 2l+1).
        """
        self._cache = _build_ylm_cache(unit_vecs, l_values)

    def get(self, degree: int) -> NDArray[np.float64]:
        """Return (M, 2l+1) real spherical harmonics for degree l."""
        return self._cache[degree]


def _build_ylm_cache(
    unit_vecs: NDArray[np.float64],
    l_values: list[int],
) -> dict[int, NDArray[np.float64]]:
    """Compute real spherical harmonics for all l in l_values using numpy.

    Returns dict l → (M, 2l+1) float64.
    """
    M = unit_vecs.shape[0]
    x_vec, y_vec, z_vec = unit_vecs[:, 0], unit_vecs[:, 1], unit_vecs[:, 2]
    ct = np.clip(z_vec, -1.0, 1.0)  # cos(θ)
    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))  # sin(θ) ≥ 0

    l_max_deg = max(l_values) if l_values else 0
    l_set = set(l_values)

    # ------------------------------------------------------------------
    # Trig table: cos_mphi[m-1] = cos(mφ), sin_mphi[m-1] = sin(mφ)
    # cos(φ) = x / r_xy,  sin(φ) = y / r_xy  (no arctan2 needed).
    # At the poles r_xy = 0: Y_l^{m≠0} = 0, so any finite value works.
    # ------------------------------------------------------------------
    if l_max_deg > 0:
        r_xy = np.sqrt(x_vec * x_vec + y_vec * y_vec)
        safe_r = np.where(r_xy > 0.0, r_xy, 1.0)
        cos_phi = x_vec / safe_r  # (M,)
        sin_phi = y_vec / safe_r  # (M,)

        cos_mphi = np.empty((l_max_deg, M), dtype=np.float64)
        sin_mphi = np.empty((l_max_deg, M), dtype=np.float64)
        cos_mphi[0] = cos_phi
        sin_mphi[0] = sin_phi
        if l_max_deg >= 2:
            two_cos = 2.0 * cos_phi
            cos_mphi[1] = two_cos * cos_phi - 1.0  # cos(2φ)
            sin_mphi[1] = two_cos * sin_phi  # sin(2φ)
            for m in range(3, l_max_deg + 1):
                cos_mphi[m - 1] = two_cos * cos_mphi[m - 2] - cos_mphi[m - 3]
                sin_mphi[m - 1] = two_cos * sin_mphi[m - 2] - sin_mphi[m - 3]
    else:
        cos_mphi = np.empty((0, M), dtype=np.float64)
        sin_mphi = np.empty((0, M), dtype=np.float64)

    # ------------------------------------------------------------------
    # Normalization constants N_l^m = sqrt((2l+1)(l-m)! / (4π(l+m)!))
    # ------------------------------------------------------------------
    _4pi = 4.0 * _pi
    max_fact = 2 * l_max_deg + 1
    fact = [1] * (max_fact + 1)
    for k in range(1, max_fact + 1):
        fact[k] = fact[k - 1] * k

    norm_table: dict[int, NDArray[np.float64]] = {}
    for degree in l_values:
        norms = np.empty(degree + 1, dtype=np.float64)
        two_lp1 = 2 * degree + 1
        for m in range(degree + 1):
            norms[m] = _msqrt(two_lp1 * fact[degree - m] / (_4pi * fact[degree + m]))
        norm_table[degree] = norms  # shape (l+1,)

    # ------------------------------------------------------------------
    # Plm recurrence: outer loop over ORDER m (0 … l_max_deg),
    # inner recurrence over DEGREE l (m … l_max_deg), vectorised over M.
    # Each P_l^m is computed exactly once (vs lpmv which recomputes
    # the recurrence from P_m^m for every (m, l) call).
    # ------------------------------------------------------------------
    # plm_store[l][m, :] = N_l^m * P_l^m(cos θ)  shape (l+1, M)
    plm_store: dict[int, NDArray[np.float64]] = {
        degree: np.empty((degree + 1, M), dtype=np.float64) for degree in l_values
    }

    p_sector = np.ones(M, dtype=np.float64)  # P_0^0 = 1 (with CS phase)

    for m in range(l_max_deg + 1):
        # --- Update sector harmonic P_m^m ---
        # P_m^m = -(2m-1) sinθ P_{m-1}^{m-1}  (Condon-Shortley convention)
        if m > 0:
            p_sector = p_sector * (-(2 * m - 1)) * st  # new array each iteration

        # --- Roll recurrence upward for this order m ---
        p_prev = np.zeros(M, dtype=np.float64)  # P_{m-1}^m (sentinel, not used at l=m+1)
        p_curr = p_sector  # P_m^m; p_sector reassigned each outer iter

        # l == m (sector)
        if m in l_set:
            plm_store[m][m] = norm_table[m][m] * p_curr

        for degree in range(m + 1, l_max_deg + 1):
            if degree == m + 1:
                # First step uses 2-term relation: P_{m+1}^m = (2m+1) cosθ P_m^m
                p_new = (2 * m + 1) * ct * p_curr
            else:
                # General 3-term recurrence
                p_new = ((2 * degree - 1) * ct * p_curr - (degree + m - 1) * p_prev) / (degree - m)
            p_prev = p_curr
            p_curr = p_new

            if degree in l_set:
                plm_store[degree][m] = norm_table[degree][m] * p_curr

    # ------------------------------------------------------------------
    # Assemble Y_lm from plm_store and trig tables
    # ------------------------------------------------------------------
    cache: dict[int, NDArray[np.float64]] = {}
    for degree in l_values:
        Y = np.empty((M, 2 * degree + 1), dtype=np.float64)
        pn = plm_store[degree]  # (degree+1, M)
        Y[:, degree] = pn[0]  # m = 0
        if degree > 0:
            # m > 0: columns degree+1 … 2*degree
            Y[:, degree + 1 :] = (_SQRT2 * pn[1:] * cos_mphi[:degree]).T
            # m < 0: columns 0 … degree-1  (|m| = degree … 1 reversed)
            Y[:, :degree] = (_SQRT2 * pn[degree:0:-1] * sin_mphi[degree - 1 :: -1]).T
        cache[degree] = Y
    return cache


_SQRT2 = float(np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Core descriptor computation
# ---------------------------------------------------------------------------


def compute_power_spectrum(
    reflections: ReflectionList,
    params: RinseParams | None = None,
    *,
    debug: bool = False,
    log1p: bool = True,
    l2: bool = True,
) -> NDArray[np.float64]:
    """Compute the RINSE power spectrum from a :class:`ReflectionList`.

    Parameters
    ----------
    reflections:
        Pre-computed reflection list from
        :func:`~rinse_descriptor._structure_factors.compute_structure_factors`.
    params:
        Descriptor hyper-parameters.  Uses defaults if *None*.
    log1p:
        Apply log1p normalisation to the raw power spectrum before L2 normalisation.
    l2:
        Apply global L2 normalisation to the log1p power spectrum.

    Returns
    -------
    descriptor : (n_max, l_max) float64
        Power spectrum matrix.  Flatten to 1-D for use as a feature vector.
        ``descriptor[n, k]`` corresponds to radial order *n* and angular level
        ℓ = 2*k (k = 0 … l_max-1).
    """
    if params is None:
        params = RinseParams()

    q = reflections.q_magnitudes  # (M,)
    q_vecs = reflections.q_vectors  # (M, 3)
    intensities = reflections.intensities  # (M,)

    M = len(q)
    if M == 0:
        return np.zeros(params.descriptor_shape, dtype=np.float64)

    # Normalise q-vectors to unit vectors (handle |G|=0 gracefully)
    norms = np.maximum(q, 1e-12)
    unit_vecs = q_vecs / norms[:, np.newaxis]  # (M, 3)

    # --- Radial basis R[i, n] = R_n(q_i) ---
    _t = time.perf_counter()
    R = evaluate_radial_basis(
        q,
        q_max=params.q_max,
        n_max=params.n_max,
        basis=params.radial_basis,
    )  # (M, n_max)
    if debug:
        print(
            f"[rinse_descriptor]   ps: radial basis:     {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"({M} reflections, basis={params.radial_basis!r})",
            file=sys.stderr,
        )

    # Weight radial basis by intensities: W[i, n] = I_i * R_n(q_i)
    W = intensities[:, np.newaxis] * R  # (M, n_max)

    # --- Spherical harmonics cache ---
    _t = time.perf_counter()
    sph_cache = _SphHarmCache(unit_vecs, params.l_values)
    if debug:
        print(
            f"[rinse_descriptor]   ps: sph harm cache:   {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"(l_max={params.l_max})",
            file=sys.stderr,
        )

    # --- Accumulate power spectrum ---
    _t = time.perf_counter()
    P = np.zeros((params.n_max, params.l_max), dtype=np.float64)

    for k, degree in enumerate(params.l_values):
        Y = sph_cache.get(degree)  # (M, 2l+1)

        # A[n, m] = Σ_i W[i, n] * Y[i, m]  →  shape (n_max, 2l+1)
        A = W.T @ Y  # (n_max, 2l+1)

        # p[n] = Σ_m A[n,m]²
        P[:, k] = np.einsum("nm,nm->n", A, A)

    if debug:
        print(
            f"[rinse_descriptor]   ps: accumulate:       {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"(n_max={params.n_max})",
            file=sys.stderr,
        )

    _t = time.perf_counter()
    result = normalise_power_spectrum(P, log1p=log1p, l2=l2)
    if debug:
        print(
            f"[rinse_descriptor]   ps: normalise:        {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"(method='{'log1p+' if log1p else ''}{'l2' if l2 else ''}')",
            file=sys.stderr,
        )
    return result


def normalise_power_spectrum(
    P: NDArray[np.float64], log1p: bool = True, l2: bool = True
) -> NDArray[np.float64]:
    """Apply fixed post-hoc normalisation to a raw power-spectrum matrix.

    Parameters
    ----------
    P:
        (n_max, l_max) raw power-spectrum matrix.

    Returns
    -------
    (n_max, l_max) float64 — normalised matrix via ``log1p`` then global L2.
    """
    p_norm = np.log1p(np.maximum(P, 0.0)) if log1p else np.maximum(P, 0.0)
    l2_norm = float(np.linalg.norm(p_norm)) if l2 else 1.0
    if l2_norm > 0.0:
        p_norm /= l2_norm
    return p_norm


def power_spectrum_to_vector(P: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten a (n_max, l_max) power spectrum matrix to a 1-D vector.

    The layout is column-major: element index i*l_max + k corresponds to
     radial order n=i and angular level ℓ=2k.
    """
    return P.T.ravel()
