"""Radial basis functions for the RINSE descriptor.

Two families are supported:

``"chebyshev"``
    Chebyshev polynomials of the first kind, T_n(x), evaluated with the
    argument mapped from q ∈ [0, q_max] → x ∈ [-1, 1]:
        x(q) = 2q/q_max - 1
    Indices n = 0, 1, …, n_max-1.

``"bessel"``
    A 1-D spherical Bessel j_0 radial basis with modes
        R_n(q) = j_0(z_n(q)),   z_n(q) = α_n · q / q_max
    where α_n is the n-th positive root of j_0 (i.e. α_n = nπ).  This gives
    n_max radial functions that vanish at q = q_max.  The same radial basis
    is intended to be reused for each angular component in the descriptor.

``"smooth_shells_cw"``
    Smooth overlapping shells over q ∈ [0, q_max], built from Gaussian
    windows centered at uniformly spaced shell centers.  Rows are
    normalised so Σ_n R_n(q) = 1 at each q.

``"smooth_shells_nl"``
    Smooth overlapping shells over q ∈ [0, q_max], built from Gaussian
    windows centered at non-linearly spaced shell centers.  Rows are
    normalised so Σ_n R_n(q) = 1 at each q.

Both bases return an (M, n_max) array for M query points.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import spherical_jn

RadialBasisType = Literal[
    "chebyshev",
    "bessel",
    "smooth_shells_cw",
    "smooth_shells_nl",
    "",
]


def evaluate_radial_basis(
    q: NDArray[np.float64],
    *,
    q_max: float,
    n_max: int = 8,
    basis: RadialBasisType = "chebyshev",
) -> NDArray[np.float64]:
    """Evaluate radial basis functions at reciprocal-space magnitudes *q*.

    Parameters
    ----------
    q:
        (M,) array of |G| values in Å⁻¹.
    q_max:
        Cutoff radius in Å⁻¹.  Values outside [0, q_max] are clamped.
    n_max:
        Number of radial basis functions (n = 0 … n_max-1).
    basis:
        ``"chebyshev"``, ``"bessel"``, or
        ``"smooth_shells_cw"``/``"smooth_shells_nl"``.

    Returns
    -------
    R : (M, n_max) float64
        R[i, n] = R_n(q[i]).
    """
    q = np.asarray(q, dtype=np.float64)
    if basis == "chebyshev":
        return _chebyshev_basis(q, q_max=q_max, n_max=n_max)
    elif basis in {"bessel"}:
        return _bessel_basis(q, q_max=q_max, n_max=n_max)
    elif basis == "smooth_shells_cw":
        return _smooth_shell_basis_cw(q, q_max=q_max, n_max=n_max)
    elif basis == "smooth_shells_nl":
        return _smooth_shell_basis_nl(q, q_max=q_max, n_max=n_max)
    else:
        raise ValueError(
            f"Unknown radial basis '{basis}'. "
            "Choose 'chebyshev', 'bessel', or "
            "'smooth_shells_cw'/'smooth_shells_nl'."
        )


# ---------------------------------------------------------------------------
# Chebyshev basis
# ---------------------------------------------------------------------------


def _chebyshev_basis(
    q: NDArray[np.float64],
    *,
    q_max: float,
    n_max: int,
) -> NDArray[np.float64]:
    """Chebyshev T_n basis evaluated via the three-term recurrence.

    Maps q ∈ [0, q_max] → x ∈ [-1, 1] as x = 2q/q_max - 1,
    so that x = -1 at q = 0 and x = 1 at q = q_max.
    """
    x = np.clip(2.0 * q / q_max - 1.0, -1.0, 1.0)  # (M,)
    M = x.shape[0]
    R = np.empty((M, n_max), dtype=np.float64)

    if n_max == 0:
        return R

    # Three-term recurrence: T_0 = 1, T_1 = x, T_{n+1} = 2x T_n - T_{n-1}
    R[:, 0] = 1.0
    if n_max == 1:
        return R
    R[:, 1] = x
    for n in range(2, n_max):
        R[:, n] = 2.0 * x * R[:, n - 1] - R[:, n - 2]

    return R


# ---------------------------------------------------------------------------
# Spherical Bessel radial basis
# ---------------------------------------------------------------------------

# Use the spherical Bessel function j_0 as a 1-D radial basis.
# We vary the radial mode index n through successive roots α_n = n*π,
# giving
#   R_n(q) = j_0(α_n * q / q_max)  for n = 1 … n_max
# so every radial function vanishes at q = q_max.  These radial functions
# are shared across angular channels rather than changing Bessel order.
#
# Index mapping: basis index 0 → α_1 = π, …, n-1 → α_n = n*π


def _bessel_basis(
    q: NDArray[np.float64],
    *,
    q_max: float,
    n_max: int,
) -> NDArray[np.float64]:
    """Evaluate 1-D spherical Bessel j_0 radial modes on [0, q_max]."""
    q_clipped = np.clip(q, 0.0, q_max)
    M = q_clipped.shape[0]
    R = np.empty((M, n_max), dtype=np.float64)

    for i in range(n_max):
        # Radial mode n = i+1 uses the corresponding j_0 root alpha = n*pi.
        alpha = (i + 1) * np.pi
        z = alpha * q_clipped / q_max  # dimensionless Bessel argument, shape (M,)
        # j_0(z) = sin(z)/z, but use scipy for numerical safety at z=0.
        R[:, i] = spherical_jn(0, z)

    return R


# ---------------------------------------------------------------------------
# Smooth shell basis
# ---------------------------------------------------------------------------


def _smooth_shell_basis_cw(
    q: NDArray[np.float64],
    *,
    q_max: float,
    n_max: int,
) -> NDArray[np.float64]:
    """Smooth overlapping radial shells with partition-of-unity normalisation.

    Shell centers are uniformly spaced on [0, q_max].  Each shell is a
    Gaussian in q, and rows are normalised so the basis sums to 1 at each q.
    """
    q_clipped = np.clip(q, 0.0, q_max)
    M = q_clipped.shape[0]
    R = np.empty((M, n_max), dtype=np.float64)

    if n_max == 0:
        return R
    if n_max == 1:
        R[:, 0] = 1.0
        return R

    centers = np.linspace(0.0, q_max, n_max, dtype=np.float64)
    spacing = q_max / (n_max - 1)
    sigma = spacing

    scaled = (q_clipped[:, np.newaxis] - centers[np.newaxis, :]) / sigma
    R[:, :] = np.exp(-0.5 * scaled**2)

    row_sum = R.sum(axis=1, keepdims=True)
    R /= np.maximum(row_sum, np.finfo(np.float64).tiny)
    return R


def _smooth_shell_basis_nl(
    q: NDArray[np.float64],
    *,
    q_max: float,
    n_max: int,
) -> NDArray[np.float64]:
    """Smooth overlapping radial shells with width-based normalisation.

    Shell means are uniformly spaced in spherical-volume coordinate
    u = (q/q_max)^3, giving non-linear spacing in q.  Shell widths follow
    local shell thickness and each Gaussian is scaled by 1/σ_n.
    """
    q_clipped = np.clip(q, 0.0, q_max)
    M = q_clipped.shape[0]
    R = np.empty((M, n_max), dtype=np.float64)

    if n_max == 0:
        return R
    if n_max == 1:
        R[:, 0] = 1.0
        return R

    # Partition shells uniformly in spherical-volume coordinate u = (q/q_max)^3.
    u_edges = np.linspace(0.0, 1.0, n_max + 1, dtype=np.float64)
    q_edges = q_max * np.cbrt(u_edges)

    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    centers = q_max * np.cbrt(u_centers)
    widths = q_edges[1:] - q_edges[:-1]
    sigma = np.maximum(widths, np.finfo(np.float64).tiny)

    scaled = (q_clipped[:, np.newaxis] - centers[np.newaxis, :]) / sigma[np.newaxis, :]
    # Width-based normalisation: broader shells have proportionally lower peak height.
    R[:, :] = np.exp(-0.5 * scaled**2) / sigma[np.newaxis, :]
    return R
