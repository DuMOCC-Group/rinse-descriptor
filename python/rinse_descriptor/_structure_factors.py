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

Displacement parameters are taken directly from the :class:`cctbx.xray.structure`
as parsed from the CIF file.
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
    intensities : (M,) float64  – chosen structure factor type (|F|² or |F|)
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
    """
    ff_type = FormFactorType(form_factor_type)
    sf_type = StructureFactorType(structure_factor_type)

    d_min = 1.0 / (2.0 * sin_theta_over_lambda_max)

    # Reciprocal-space vector matrix: rows = a*, b*, c* in Å⁻¹
    uc = xrs.unit_cell()
    orth = np.array(uc.orthogonalization_matrix(), dtype=np.float64).reshape(3, 3)
    recip = np.linalg.inv(orth.T).T  # rows = a*, b*, c*

    _t = time.perf_counter()
    hkl_arr, F_vals = _calc_cctbx(xrs, d_min, ff_type)
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
    if sf_type == StructureFactorType.F2:
        intensities = F2
    else:
        intensities = np.sqrt(np.maximum(F2, 0.0))
    if debug:
        print(
            f"[rinse_descriptor] sf: normalise:        "
            f"{(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"(type={sf_type.value!r})",
            file=sys.stderr,
        )

    # Remove (000) and any zero-vector reflections
    mask = q_magnitudes > 1e-9
    return ReflectionList(
        hkl=hkl_arr[mask].astype(np.int32),
        q_vectors=q_vectors[mask],
        q_magnitudes=q_magnitudes[mask],
        intensities=intensities[mask],
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _calc_cctbx(
    xrs: Any,
    d_min: float,
    ff_type: FormFactorType,
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
    xrs_calc.scattering_type_registry(table=_CCTBX_TABLES[ff_type])

    fc_asym = xrs_calc.structure_factors(
        d_min=d_min, anomalous_flag=True, algorithm="direct"
    ).f_calc()

    fc_p1 = fc_asym.expand_to_p1()

    hkl = np.array(list(fc_p1.indices()), dtype=np.int32)
    F_vals = np.array(list(fc_p1.data()), dtype=np.complex128)

    return hkl, F_vals
