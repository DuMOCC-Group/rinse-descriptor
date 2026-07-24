"""Structure factor calculation via cctbx (direct summation).

This module computes structure factors F(hkl) for a :class:`cctbx.xray.structure`.

The calculation works on the Miller asymmetric unit only, then expands to the
full diffraction sphere via symmetry operations:

1. ``xrs.structure_factors(anomalous_flag=True)`` → unique reflections including
   both Bijvoet mates (F(hkl) and F(-h-k-l) treated as independent when anomalous
   scattering is present).
2. ``expand_to_p1()`` → all space-group equivalents; with anomalous=True this
   already covers the full sphere.

Because anomalous scattering breaks Friedel symmetry (F(hkl) ≠ F(-h-k-l)*),
the descriptor can distinguish enantiomeric crystal structures.

The final :class:`ReflectionList` contains one entry per reciprocal-lattice
point in the sphere |G| ≤ 2·sin(θ)/λ_max.

Form-factor conventions
-----------------------
* ``"xray"``    — Waasmaier-Kirfel 1995 X-ray form factors
* ``"electron"``— Electron scattering factors
* ``"neutron"`` — Neutron coherent scattering lengths

Structure factor types
----------------------
* ``"F"``  — |F(hkl)|
* ``"F2"`` — |F(hkl)|²  (default)

Intensity normalisation
-----------------------
* ``"double_exponential"`` — fit an unbinned physically motivated envelope
    ``A * exp(-b*s^2 - c*s^4)`` to ``|F|²`` over all reflections, where
    ``s = sin(θ)/λ`` (default)
* ``"empirical"`` — estimate the resolution envelope of |F|² from adaptive
    bins in sin(θ)/λ, divide amplitudes by sqrt(envelope), then convert back to
    the requested output type
* ``"none"``      — use calculated intensities as-is

Intensity falloff
-----------------
* ``"debye_waller"`` — multiply amplitudes by an isotropic Debye-Waller factor
    with configurable average U_iso (default)
* ``"none"`` — do not apply an additional falloff

By default, isotropic and anisotropic displacement parameters are used as stored
in the :class:`cctbx.xray.structure`. Pass ``use_reported_adps=False`` to reset
all atoms to isotropic thermal motion with U_iso = 0.05 Å².
"""

from __future__ import annotations

import sys
import time
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from cctbx import xray as _cctbx_xray  # noqa: F401

# Eager import – must happen before pytest capture is active.
from iotbx import cif as _iotbx_cif  # noqa: F401
from numpy.typing import NDArray
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------


class FormFactorType(StrEnum):
    XRAY = "xray"
    ELECTRON = "electron"
    NEUTRON = "neutron"


class StructureFactorType(StrEnum):
    F = "F"
    F2 = "F2"


class IntensityNormalisation(StrEnum):
    NONE = "none"
    DOUBLE_EXPONENTIAL = "double_exponential"
    EMPIRICAL = "empirical"


class IntensityFalloff(StrEnum):
    NONE = "none"
    DEBYE_WALLER = "debye_waller"


# ---------------------------------------------------------------------------
# ReflectionList
# ---------------------------------------------------------------------------


class ReflectionList:
    """Container for a set of computed reflections.

    Attributes
    ----------
    hkl : (M, 3) int32
    q_vectors : (M, 3) float64  – Cartesian reciprocal-space vectors in Å⁻¹
        Using the crystallographic convention G = h·a* + k·b* + l·c*
        (no 2π factor), so |G| = 2sin(θ)/λ.
    q_magnitudes : (M,) float64 – |G| in Å⁻¹
    intensities : (M,) float64  – intensities |F|² by default, or amplitudes
        |F| only when ``structure_factor_type="F"`` is requested directly.
    """

    def __init__(
        self,
        hkl: NDArray[np.int32],
        q_vectors: NDArray[np.float64],
        q_magnitudes: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        self.hkl = hkl
        self.q_vectors = q_vectors
        self.q_magnitudes = q_magnitudes
        self.intensities = intensities

    def __len__(self) -> int:
        return int(self.hkl.shape[0])

    def __repr__(self) -> str:
        if len(self) == 0:
            return "ReflectionList(n_reflections=0)"
        return f"ReflectionList(n_reflections={len(self)}, q_max={self.q_magnitudes.max():.3f} Å⁻¹)"


# ---------------------------------------------------------------------------
# cctbx scattering tables
# ---------------------------------------------------------------------------

_CCTBX_TABLES: dict[FormFactorType, str] = {
    FormFactorType.XRAY: "wk1995",
    FormFactorType.ELECTRON: "electron",
    FormFactorType.NEUTRON: "neutron",
}

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_structure_factors(
    xrs: Any,
    *,
    sin_theta_over_lambda_max: float = 2.0,
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron"] = "xray",
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
    intensity_normalisation: IntensityNormalisation
    | Literal["none", "double_exponential", "empirical"] = "double_exponential",
    intensity_normalisation_n_bins: int = 6,
    intensity_normalisation_min_bin_size: int = 50,
    intensity_falloff: IntensityFalloff | Literal["none", "debye_waller"] = "debye_waller",
    intensity_falloff_u_iso: float = 0.05,
    use_reported_adps: bool = True,
    debug: bool = False,
) -> ReflectionList:
    """Compute structure factors and return a :class:`ReflectionList`.

    Parameters
    ----------
    xrs:
        A :class:`cctbx.xray.structure` (e.g. from :func:`~rinse_descriptor.load_cif`).
    sin_theta_over_lambda_max:
        Resolution cutoff; default 2.0 Å⁻¹ → |G| ≤ 4.0 Å⁻¹.
    form_factor_type:
        Scattering factor type. ``"xray"`` | ``"electron"`` | ``"neutron"``.
    structure_factor_type:
        Output structure factor type. ``"F2"`` | ``"F"``.
    intensity_normalisation:
        Resolution-envelope normalisation to apply before output conversion.
        ``"double_exponential"`` fits ``A * exp(-b*s^2 - c*s^4)`` to
        ``|F|²`` using all reflections (no binning), applies
        ``F' = F / sqrt(envelope)``, then returns either ``|F'|²`` or ``|F'|``.
        ``"empirical"`` estimates ⟨|F|²⟩ in adaptive sin(θ)/λ bins, applies
        ``F' = F / sqrt(envelope)``, then returns either ``|F'|²`` or ``|F'|``.
    intensity_normalisation_n_bins:
        Maximum number of adaptive bins for ``"empirical"`` normalisation.
    intensity_normalisation_min_bin_size:
        Minimum target reflections per adaptive bin.
    intensity_falloff:
        Amplitude falloff to apply after intensity normalisation.
        ``"debye_waller"`` multiplies amplitudes by
        ``exp(-8 * pi**2 * U_iso * s**2)``, where ``s = sin(theta)/lambda``.
    intensity_falloff_u_iso:
        Average isotropic displacement parameter in Å² for ``"debye_waller"``
        falloff. Must be non-negative.
    use_reported_adps:
        If *True* (default), use displacement parameters from the CIF. If
        *False*, all atoms are reset to isotropic U_iso = 0.01 Å².
    """
    ff_type = FormFactorType(form_factor_type)
    sf_type = StructureFactorType(structure_factor_type)
    intensity_norm = IntensityNormalisation(intensity_normalisation)
    falloff = IntensityFalloff(intensity_falloff)

    d_min = 1.0 / (2.0 * sin_theta_over_lambda_max)

    # Reciprocal-space vector matrix: rows = a*, b*, c* in Å⁻¹
    uc = xrs.unit_cell()
    orth = np.array(uc.orthogonalization_matrix(), dtype=np.float64).reshape(3, 3)
    recip = np.linalg.inv(orth.T).T  # rows = a*, b*, c*

    _t = time.perf_counter()
    hkl_arr, F_vals = _calc_cctbx(xrs, d_min, ff_type, use_reported_adps=use_reported_adps)
    if debug:
        print(
            f"[rinse_descriptor] sf: calculate F(hkl):"
            f"{(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"(M={hkl_arr.shape[0]}, ff={ff_type.value!r})",
            file=sys.stderr,
        )

    q_vectors = hkl_arr.astype(np.float64) @ recip
    q_magnitudes = np.linalg.norm(q_vectors, axis=1)

    _t = time.perf_counter()
    F2 = (F_vals * F_vals.conj()).real.astype(np.float64)

    # Remove (000) and any zero-vector reflections before estimating envelopes.
    mask = q_magnitudes > 1e-9
    hkl_arr = hkl_arr[mask].astype(np.int32)
    q_vectors = q_vectors[mask]
    q_magnitudes = q_magnitudes[mask]
    F2 = F2[mask]

    if intensity_norm == IntensityNormalisation.DOUBLE_EXPONENTIAL:
        envelope = _double_exponential_intensity_envelope(q_magnitudes, F2)
        F2 = F2 / envelope
    elif intensity_norm == IntensityNormalisation.EMPIRICAL:
        envelope = _empirical_intensity_envelope(
            q_magnitudes,
            F2,
            n_bins=intensity_normalisation_n_bins,
            min_bin_size=intensity_normalisation_min_bin_size,
        )
        F2 = F2 / envelope

    if falloff == IntensityFalloff.DEBYE_WALLER:
        window = _debye_waller_amplitude_window(
            q_magnitudes,
            u_iso=intensity_falloff_u_iso,
        )
        F2 = F2 * window * window

    if sf_type == StructureFactorType.F2:
        intensities = F2
    else:
        intensities = np.sqrt(np.maximum(F2, 0.0))
    if debug:
        print(
            f"[rinse_descriptor] sf: normalise:        "
            f"{(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"(type={sf_type.value!r}, intensity_normalisation={intensity_norm.value!r}, "
            f"intensity_falloff={falloff.value!r}, "
            f"intensity_falloff_u_iso={intensity_falloff_u_iso!r})",
            file=sys.stderr,
        )

    return ReflectionList(
        hkl=hkl_arr,
        q_vectors=q_vectors,
        q_magnitudes=q_magnitudes,
        intensities=intensities,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _calc_cctbx(
    xrs: Any,
    d_min: float,
    ff_type: FormFactorType,
    *,
    use_reported_adps: bool = True,
) -> tuple[NDArray[np.int32], NDArray[np.complex128]]:
    """Compute F(hkl) using cctbx and expand to the full reciprocal sphere.

    Strategy
    --------
    1. Deep-copy the structure and set the requested scattering table.
    2. Calculate structure factors for the Miller asymmetric unit with
       ``anomalous_flag=True`` → both Friedel mates (F(hkl) and F(-h-k-l))
       are treated as independent reflections.
    3. ``expand_to_p1()`` → all space-group equivalents; with anomalous=True
       this directly yields the full sphere.
    """
    xrs_calc = xrs.deep_copy_scatterers()

    if not use_reported_adps:
        # Reset every atom to isotropic U_iso = 0.01 Å², giving a consistent
        # baseline regardless of what (if anything) was reported in the CIF.
        xrs_calc.convert_to_isotropic()
        scs = xrs_calc.scatterers()
        for i in range(scs.size()):
            sc = scs[i]
            sc.u_iso = 0.01
            scs[i] = sc

    xrs_calc.scattering_type_registry(table=_CCTBX_TABLES[ff_type])

    fc_asym = xrs_calc.structure_factors(
        d_min=d_min, anomalous_flag=True, algorithm="direct"
    ).f_calc()

    fc_p1 = fc_asym.expand_to_p1()

    hkl = np.array(list(fc_p1.indices()), dtype=np.int32)
    F_vals = np.array(list(fc_p1.data()), dtype=np.complex128)

    return hkl, F_vals


def _empirical_intensity_envelope(
    q_magnitudes: NDArray[np.float64],
    intensities: NDArray[np.float64],
    *,
    n_bins: int = 6,
    min_bin_size: int = 50,
) -> NDArray[np.float64]:
    """Estimate <|F|²> as a smoothed function of sin(theta)/lambda.

    The raw intensities are first averaged in adaptive resolution bins, then a
    Gaussian kernel smoother is applied to the log bin means. Working in
    log-space keeps the envelope strictly positive while avoiding the kinks from
    piecewise-linear interpolation.
    """
    if q_magnitudes.shape != intensities.shape:
        raise ValueError("q_magnitudes and intensities must have the same shape")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if min_bin_size < 1:
        raise ValueError(f"min_bin_size must be >= 1, got {min_bin_size}")

    finite = np.isfinite(q_magnitudes) & np.isfinite(intensities) & (intensities > 0.0)
    if not np.any(finite):
        return np.ones_like(intensities, dtype=np.float64)

    s = 0.5 * q_magnitudes
    n_positive = int(finite.sum())
    adaptive_n_bins = min(n_bins, max(1, n_positive // min_bin_size))
    order = np.argsort(s[finite])
    finite_indices = np.flatnonzero(finite)[order]

    centers = np.empty(adaptive_n_bins, dtype=np.float64)
    means = np.empty(adaptive_n_bins, dtype=np.float64)
    counts = np.empty(adaptive_n_bins, dtype=np.float64)
    for i, bin_indices in enumerate(np.array_split(finite_indices, adaptive_n_bins)):
        centers[i] = float(np.mean(s[bin_indices]))
        means[i] = float(np.mean(intensities[bin_indices]))
        counts[i] = float(len(bin_indices))

    floor = np.finfo(np.float64).tiny
    if adaptive_n_bins == 1:
        envelope = np.full_like(intensities, means[0], dtype=np.float64)
    else:
        log_means = np.log(np.maximum(means, floor))
        center_diffs = np.diff(centers)
        bandwidth = max(
            float(np.median(center_diffs[center_diffs > 0.0]))
            if np.any(center_diffs > 0.0)
            else 0.0,
            float((centers[-1] - centers[0]) / adaptive_n_bins),
            float(np.spacing(1.0)),
        )

        # Add one support point at each edge by linear extrapolation in log-space,
        # then smooth against the augmented support with the Gaussian kernel.
        left_dx = max(centers[1] - centers[0], float(np.spacing(1.0)))
        right_dx = max(centers[-1] - centers[-2], float(np.spacing(1.0)))
        left_slope = (log_means[1] - log_means[0]) / left_dx
        right_slope = (log_means[-1] - log_means[-2]) / right_dx

        support_centers = np.concatenate(
            ([centers[0] - left_dx], centers, [centers[-1] + right_dx])
        )
        support_log_means = np.concatenate(
            (
                [log_means[0] - left_slope * left_dx],
                log_means,
                [log_means[-1] + right_slope * right_dx],
            )
        )
        support_counts = np.concatenate(([counts[0]], counts, [counts[-1]]))

        distances = (s[:, None] - support_centers[None, :]) / bandwidth
        weights = np.exp(-0.5 * distances * distances) * support_counts[None, :]
        smoothed_log_means = (weights @ support_log_means) / np.maximum(
            weights.sum(axis=1), np.finfo(np.float64).tiny
        )
        envelope = np.exp(smoothed_log_means)

    return np.maximum(envelope, floor)


def _double_exponential_intensity_envelope(
    q_magnitudes: NDArray[np.float64],
    intensities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit ``A * exp(-b*s^2 - c*s^4)`` to ``|F|²`` using all reflections.

    The model combines two physically motivated decay factors over
    ``s = sin(theta)/lambda``:

    - atomic form-factor falloff (captured by ``exp(-b*s^2)``)
    - Debye-Waller-like damping (captured by ``exp(-c*s^4)``)

    Fitting is done with nonlinear least squares in a positive parameterisation:
    ``A = exp(theta0)``, ``b = exp(theta1)``, ``c = exp(theta2)``.
    Residuals are evaluated in log-space to balance low/high intensity ranges.
    """
    if q_magnitudes.shape != intensities.shape:
        raise ValueError("q_magnitudes and intensities must have the same shape")

    floor = np.finfo(np.float64).tiny
    finite = np.isfinite(q_magnitudes) & np.isfinite(intensities) & (intensities > 0.0)
    if int(finite.sum()) < 3:
        return np.ones_like(intensities, dtype=np.float64)

    s = 0.5 * q_magnitudes
    s_fit = s[finite]
    y_fit = np.maximum(intensities[finite], floor)
    log_y_fit = np.log(y_fit)

    # Build a stable initial guess from the linearised model.
    s2 = s_fit * s_fit
    s4 = s2 * s2
    X = np.column_stack((np.ones_like(s2), s2, s4))
    coeffs, *_ = np.linalg.lstsq(X, log_y_fit, rcond=None)

    log_a0 = float(coeffs[0])
    b0 = max(-float(coeffs[1]), float(np.spacing(1.0)))
    c0 = max(-float(coeffs[2]), float(np.spacing(1.0)))
    theta0 = np.array([log_a0, np.log(b0), np.log(c0)], dtype=np.float64)

    def _residual(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        log_a, log_b, log_c = theta
        b = np.exp(log_b)
        c = np.exp(log_c)
        return (log_a - b * s2 - c * s4) - log_y_fit  # type: ignore[no-any-return]

    fit = least_squares(_residual, theta0, method="trf")
    theta = fit.x if fit.success else theta0
    log_a, log_b, log_c = map(float, theta)
    b = np.exp(log_b)
    c = np.exp(log_c)

    all_s2 = s * s
    all_s4 = all_s2 * all_s2
    log_envelope = log_a - b * all_s2 - c * all_s4
    envelope = np.exp(log_envelope)
    clipped_envelope = np.maximum(envelope, floor).astype(np.float64, copy=False)
    return clipped_envelope  # type: ignore[no-any-return]


def _debye_waller_amplitude_window(
    q_magnitudes: NDArray[np.float64],
    *,
    u_iso: float = 0.05,
) -> NDArray[np.float64]:
    """Isotropic Debye-Waller amplitude factor over s = sin(theta)/lambda."""
    if u_iso < 0.0:
        raise ValueError(f"u_iso must be >= 0, got {u_iso}")

    s = 0.5 * q_magnitudes
    window = np.exp(-8.0 * np.pi**2 * u_iso * s * s).astype(np.float64, copy=False)
    return window
