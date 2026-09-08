"""Radial basis functions for the RINSE descriptor.

Shells are *anchored by index* through a single ``radial_scale`` factor.
This means the position of shell *n* is fixed regardless of ``n_max``:
increasing ``n_max`` simply adds further shells at higher *q* while leaving
the already-computed shells unchanged.  The reciprocal-space cutoff ``q_max``
is therefore a derived quantity (see :func:`radial_basis_q_max`), not an input.

Two families are supported:

``"cv_gaussian"`` (constant volume, default)
    Volume-uniform overlapping shells.  Shell edges are placed at
        q_edge(n) = radial_scale · n^(1/3)
    so every shell spans an equal reciprocal-space volume ∝ radial_scale³.
    Each shell is a Gaussian scaled so its q²-weighted integral is constant
    across shells.  ``q_max = radial_scale · n_max^(1/3)``.

``"lin_gaussian"`` (linear spacing)
    Linearly-spaced overlapping shells with partition-of-unity normalisation.
    Shell centers are placed at
        q_center(n) = radial_scale · n,
    the Gaussian width equals ``radial_scale``, and rows are normalised so
    Σ_n R_n(q) = 1 at each q.  ``q_max = radial_scale · (n_max - 1)``.

Both bases return an (M, n_max) array for M query points.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

RadialBasisType = Literal[
    "cv_gaussian",
    "lin_gaussian",
    "",
]


def radial_basis_q_max(
    radial_scale: float,
    n_max: int,
    basis: RadialBasisType = "cv_gaussian",
) -> float:
    """Derived reciprocal-space cutoff |G|_max for a given shell layout.

    This is the outer edge of the shells implied by ``radial_scale`` and
    ``n_max``.  It grows with ``n_max`` while leaving inner shells fixed.

    Parameters
    ----------
    radial_scale:
        Per-shell scale factor in Å⁻¹.
    n_max:
        Number of radial shells (n = 0 … n_max-1).
    basis:
        ``"cv_gaussian"`` (volume-uniform) or ``"lin_gaussian"``
        (linear).

    Returns
    -------
    q_max : float
        |G| cutoff in Å⁻¹.
    """
    if n_max <= 0:
        return 0.0
    if basis == "cv_gaussian":
        return float(float(radial_scale) * float(n_max) ** (1.0 / 3.0))
    elif basis == "lin_gaussian":
        return float(radial_scale) * float(max(n_max - 1, 1))
    else:
        raise ValueError(f"Unknown radial basis '{basis}'. Choose 'cv_gaussian' or 'lin_gaussian'.")


def evaluate_radial_basis(
    q: NDArray[np.float64],
    *,
    radial_scale: float,
    n_max: int = 16,
    basis: RadialBasisType = "cv_gaussian",
) -> NDArray[np.float64]:
    """Evaluate radial basis functions at reciprocal-space magnitudes *q*.

    Parameters
    ----------
    q:
        (M,) array of |G| values in Å⁻¹.
    radial_scale:
        Per-shell scale factor in Å⁻¹.  Sets the distance between shells;
        shell positions are anchored by index and independent of ``n_max``.
    n_max:
        Number of radial basis functions (n = 0 … n_max-1).
    basis:
        ``"cv_gaussian"`` (default) or ``"lin_gaussian"``.

    Returns
    -------
    R : (M, n_max) float64
        R[i, n] = R_n(q[i]).
    """
    q = np.asarray(q, dtype=np.float64)
    if basis == "lin_gaussian":
        return _lin_gaussian_basis(q, radial_scale=radial_scale, n_max=n_max)
    elif basis == "cv_gaussian":
        return _cv_gaussian_basis(q, radial_scale=radial_scale, n_max=n_max)
    else:
        raise ValueError(f"Unknown radial basis '{basis}'. Choose 'cv_gaussian' or 'lin_gaussian'.")


# ---------------------------------------------------------------------------
# Smooth shell basis
# ---------------------------------------------------------------------------


def _lin_gaussian_basis(
    q: NDArray[np.float64],
    *,
    radial_scale: float,
    n_max: int,
) -> NDArray[np.float64]:
    """Smooth overlapping radial shells with partition-of-unity normalisation.

    Shell centers are anchored at ``q_center(n) = radial_scale · n``.  Each
    shell is a Gaussian of width ``radial_scale`` in q, and rows are normalised
    so the basis sums to 1 at each q.
    """
    q_clipped = np.maximum(q, 0.0)
    M = q_clipped.shape[0]
    R = np.empty((M, n_max), dtype=np.float64)

    if n_max == 0:
        return R
    if n_max == 1:
        R[:, 0] = 1.0
        return R

    centers = radial_scale * np.arange(n_max, dtype=np.float64)
    sigma = radial_scale

    scaled = (q_clipped[:, np.newaxis] - centers[np.newaxis, :]) / sigma
    R[:, :] = np.exp(-0.5 * scaled**2)

    row_sum = R.sum(axis=1, keepdims=True)
    R /= np.maximum(row_sum, np.finfo(np.float64).tiny)
    return R


def _cv_gaussian_basis(
    q: NDArray[np.float64],
    *,
    radial_scale: float,
    n_max: int,
) -> NDArray[np.float64]:
    """Smooth overlapping radial shells with volume-based normalisation.

    Shell edges are anchored at ``q_edge(n) = radial_scale · n^(1/3)``, so each
    shell spans an equal reciprocal-space volume ∝ radial_scale³ regardless of
    ``n_max``.  Shell centers are the volume-midpoints
    ``q_center(n) = radial_scale · (n + 1/2)^(1/3)`` and widths follow the local
    shell thickness.  Each Gaussian is scaled by 1/(σ_n · c_n²) so that its
    integral against the reciprocal-space volume element (∝ q² dq) is the same
    for every shell.  Because equal-volume shells then contribute equally for a
    flat (resolution-independent) intensity field, the ℓ=0 monopole varies only
    with the intensity envelope, not with shell geometry.
    """
    q_clipped = np.maximum(q, 0.0)
    M = q_clipped.shape[0]
    R = np.empty((M, n_max), dtype=np.float64)

    if n_max == 0:
        return R
    if n_max == 1:
        R[:, 0] = 1.0
        return R

    # Anchor shells by index in the spherical-volume coordinate u = (q/scale)^3:
    # edge n sits at u = n, i.e. q_edge(n) = radial_scale · n^(1/3).
    q_edges = radial_scale * np.cbrt(np.arange(n_max + 1, dtype=np.float64))
    centers = radial_scale * np.cbrt(np.arange(n_max, dtype=np.float64) + 0.5)
    widths = q_edges[1:] - q_edges[:-1]
    sigma = np.maximum(widths, np.finfo(np.float64).tiny)

    scaled = (q_clipped[:, np.newaxis] - centers[np.newaxis, :]) / sigma[np.newaxis, :]
    # Volume-based normalisation: the q²-weighted integral of each Gaussian is
    # constant across shells.  Peak height ∝ 1/(σ_n · c_n²): thinner *and* more
    # distant shells (which cover more reciprocal-space surface at ∝ q²) are
    # scaled down so equal-volume shells contribute equally for flat intensity.
    norm = sigma[np.newaxis, :] * (centers[np.newaxis, :] ** 2)
    R[:, :] = np.exp(-0.5 * scaled**2) / np.maximum(norm, np.finfo(np.float64).tiny)
    return R
