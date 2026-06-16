"""Structure factor calculation via Gemmi (direct summation).

This module computes structure factors F(hkl) for a crystal using Gemmi's
built-in direct-summation structure factor calculators, then converts them to
intensities I(q) = |F(hkl)|² or |F(hkl)| according to the chosen structure
factor type.

The reciprocal-space grid is enumerated up to a user-specified resolution
(default sin(θ)/λ ≤ 2.0 Å⁻¹, i.e. |G| = 2sin(θ)/λ ≤ 4.0 Å⁻¹).

Form-factor conventions
-----------------------
* ``"xray"``    — IT92 X-ray form factors (Gemmi StructureFactorCalculatorX)
* ``"electron"``— C4322 electron scattering factors (Gemmi StructureFactorCalculatorE)
* ``"neutron"`` — Neutron92 coherent scattering lengths (Gemmi StructureFactorCalculatorN)
* ``"unity"``   — all atoms contribute 1+0j; useful for debugging

Structure factor types
-------------------------
* ``"F"``          — |F(hkl)|
* ``"F2"``         — |F(hkl)|²  (default)

B-factor handling
-----------------
* All B-factors default to 1.0 Å² (unit B) unless overridden.
* B_iso is converted internally to U_iso = B/(8π²) for Gemmi.
"""

from __future__ import annotations

import sys
import time
from enum import StrEnum
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._crystal import Crystal

# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------


class FormFactorType(StrEnum):
    XRAY = "xray"
    ELECTRON = "electron"
    NEUTRON = "neutron"
    UNITY = "unity"


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
# Main entry point
# ---------------------------------------------------------------------------


def compute_structure_factors(
    crystal: Crystal,
    *,
    sin_theta_over_lambda_max: float = 2.0,
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron", "unity"] = "xray",
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
    b_factors: NDArray[np.float64] | None = None,
    debug: bool = False,
) -> ReflectionList:
    """Compute structure factors and return a :class:`ReflectionList`.

    Parameters
    ----------
    crystal:
        Input structure (ASE-independent).
    sin_theta_over_lambda_max:
        Resolution cutoff; default 2.0 Å⁻¹ → |G| ≤ 4.0 Å⁻¹.
    form_factor_type:
        Scattering factor type. ``"xray"`` | ``"electron"`` | ``"neutron"`` | ``"unity"``.
    structure_factor_type:
        Output structure factor type. ``"F2"`` | ``"F"``.
    b_factors:
        Per-atom isotropic B-factors in Å². *None* → unit B-factors (1 Å²).
    """
    _gemmi_available = True
    gemmi_mod: Any = None
    try:
        import gemmi as _gemmi

        # Guard against mock/stub installs (e.g. micropip.add_mock_package):
        # verify the module exposes the expected API before trusting it.
        _gemmi.SmallStructure  # noqa: B018
        gemmi_mod = _gemmi
    except (ImportError, AttributeError):
        _gemmi_available = False

    ff_type = FormFactorType(form_factor_type)
    sf_type = StructureFactorType(structure_factor_type)
    _8pi2 = 8.0 * float(np.pi) ** 2

    # --- Determine B-factors -------------------------------------------------
    n_atoms = crystal.n_atoms
    if b_factors is not None:
        b_use = np.asarray(b_factors, dtype=np.float64)
        if b_use.shape != (n_atoms,):
            raise ValueError(f"b_factors must have shape ({n_atoms},), got {b_use.shape}")
    else:
        b_from_u = np.asarray(crystal.u_iso, dtype=np.float64) * _8pi2
        if crystal.u_aniso is not None and np.any(np.abs(crystal.u_aniso) > 0.0):
            u_eq_from_aniso = np.trace(crystal.u_aniso, axis1=1, axis2=2) / 3.0
            b_from_u = np.maximum(b_from_u, u_eq_from_aniso * _8pi2)
        b_use = b_from_u if np.any(b_from_u > 0.0) else np.ones(n_atoms, dtype=np.float64)

    # --- Build Gemmi SmallStructure (only needed for gemmi SF calculators) ---
    st = None
    if _gemmi_available and ff_type != FormFactorType.UNITY:
        _t = time.perf_counter()
        base_st = getattr(crystal, "gemmi_small_structure", None)
        if base_st is not None and b_factors is None:
            st = base_st
        else:
            st = _build_small_structure(crystal, b_use, gemmi_mod)
        if debug:
            _ms = (time.perf_counter() - _t) * 1e3
            print(
                f"[rinse_descriptor] sf: build structure:  {_ms:8.2f} ms  "
                f"({crystal.n_atoms} atoms)",
                file=sys.stderr,
            )

    # --- Enumerate hkl indices -----------------------------------------------
    q_max = 2.0 * sin_theta_over_lambda_max
    recip = np.asarray(np.linalg.inv(crystal.cell).T, dtype=np.float64)  # rows = a*, b*, c* in Å⁻¹
    _t = time.perf_counter()
    hkl_arr = _enumerate_hkl(recip, q_max)
    if debug:
        print(
            f"[rinse_descriptor] sf: enumerate hkl:   {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"({hkl_arr.shape[0]} candidates, q_max={q_max:.3f} Å⁻¹)",
            file=sys.stderr,
        )

    if hkl_arr.shape[0] == 0:
        empty3 = np.empty((0, 3), dtype=np.float64)
        return ReflectionList(
            np.empty((0, 3), dtype=np.int32),
            empty3,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    # --- Compute F(hkl): gemmi, pure-Python Gaussian, or unity --------------
    _t = time.perf_counter()
    if ff_type == FormFactorType.UNITY:
        F_vals = _calc_unity(crystal, hkl_arr, b_use, recip)
    elif not _gemmi_available:
        from ._pure_python import calc_sf_gauss

        F_vals = calc_sf_gauss(crystal, hkl_arr, recip)
    else:
        assert st is not None
        calc = _make_calculator(st.cell, ff_type, gemmi_mod)
        F_vals = _calc_gemmi(calc, st, hkl_arr)
    if debug:
        print(
            f"[rinse_descriptor] sf: calculate F(hkl):{(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"(M={hkl_arr.shape[0]}, ff={ff_type.value!r})",
            file=sys.stderr,
        )

    # --- Compute q-vectors and magnitudes ------------------------------------
    _t = time.perf_counter()
    q_vectors = hkl_arr.astype(np.float64) @ recip  # (M, 3)
    q_magnitudes = np.linalg.norm(q_vectors, axis=1)  # (M,)
    if debug:
        print(
            f"[rinse_descriptor] sf: q-vectors:        {(time.perf_counter() - _t) * 1e3:8.2f} ms",
            file=sys.stderr,
        )

    # --- Normalise intensities -----------------------------------------------
    _t = time.perf_counter()
    F2 = (F_vals * F_vals.conj()).real.astype(np.float64)
    if sf_type == StructureFactorType.F2:
        intensities = F2
    elif sf_type == StructureFactorType.F:
        intensities = np.sqrt(np.maximum(F2, 0.0))
    else:
        raise ValueError(f"Unknown structure factor type: {sf_type}")
    if debug:
        print(
            f"[rinse_descriptor] sf: normalise:        {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
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


def _build_small_structure(
    crystal: Crystal,
    b_factors: NDArray[np.float64],
    gemmi: Any,
) -> Any:
    """Construct a :class:`gemmi.SmallStructure` from a :class:`Crystal`."""
    cell_mat = crystal.cell
    a = float(np.linalg.norm(cell_mat[0]))
    b_len = float(np.linalg.norm(cell_mat[1]))
    c = float(np.linalg.norm(cell_mat[2]))
    alpha = float(
        np.degrees(np.arccos(np.clip(np.dot(cell_mat[1], cell_mat[2]) / (b_len * c), -1.0, 1.0)))
    )
    beta = float(
        np.degrees(np.arccos(np.clip(np.dot(cell_mat[0], cell_mat[2]) / (a * c), -1.0, 1.0)))
    )
    gamma = float(
        np.degrees(np.arccos(np.clip(np.dot(cell_mat[0], cell_mat[1]) / (a * b_len), -1.0, 1.0)))
    )

    st = gemmi.SmallStructure()
    st.cell = gemmi.UnitCell(a, b_len, c, alpha, beta, gamma)
    st.spacegroup_hm = "P 1"

    inv_cell = np.linalg.inv(cell_mat)
    frac_pos = crystal.positions @ inv_cell.T  # (N, 3) fractional
    assert crystal.occupancies is not None
    _8pi2 = 8.0 * float(np.pi) ** 2

    for i in range(crystal.n_atoms):
        Z = int(crystal.species[i])
        elem = gemmi.Element(Z)
        site = gemmi.SmallStructure.Site()
        site.element = elem
        site.label = f"{elem.name}{i}"
        site.fract = gemmi.Fractional(
            float(frac_pos[i, 0]), float(frac_pos[i, 1]), float(frac_pos[i, 2])
        )
        site.occ = float(crystal.occupancies[i])
        site.u_iso = float(b_factors[i]) / _8pi2
        st.sites.append(site)

    return st


def _enumerate_hkl(
    recip: NDArray[np.float64],
    q_max: float,
) -> NDArray[np.int32]:
    """Return (M, 3) int32 array of all (h,k,l) with |G| ≤ q_max."""
    a_star = float(np.linalg.norm(recip[0]))
    b_star = float(np.linalg.norm(recip[1]))
    c_star = float(np.linalg.norm(recip[2]))

    h_max = int(np.ceil(q_max / a_star)) + 1
    k_max = int(np.ceil(q_max / b_star)) + 1
    l_max_val = int(np.ceil(q_max / c_star)) + 1

    h_range = np.arange(-h_max, h_max + 1, dtype=np.int32)
    k_range = np.arange(-k_max, k_max + 1, dtype=np.int32)
    l_range = np.arange(-l_max_val, l_max_val + 1, dtype=np.int32)

    hh, kk, ll = np.meshgrid(h_range, k_range, l_range, indexing="ij")
    hkl_all = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1)

    q_vecs = hkl_all.astype(np.float64) @ recip
    q_mags = np.linalg.norm(q_vecs, axis=1)
    return cast(NDArray[np.int32], hkl_all[q_mags <= q_max + 1e-9])


def _make_calculator(cell: object, ff_type: FormFactorType, gemmi: Any) -> Any:
    """Return the appropriate Gemmi StructureFactorCalculator."""
    if ff_type == FormFactorType.XRAY:
        return gemmi.StructureFactorCalculatorX(cell)
    elif ff_type == FormFactorType.ELECTRON:
        return gemmi.StructureFactorCalculatorE(cell)
    elif ff_type == FormFactorType.NEUTRON:
        return gemmi.StructureFactorCalculatorN(cell)
    else:
        raise ValueError(f"No Gemmi calculator for form_factor_type={ff_type!r}")


def _calc_gemmi(
    calc: Any,
    st: Any,
    hkl: NDArray[np.int32],
) -> NDArray[np.complex128]:
    """Compute F(hkl) for each row in *hkl* using a Gemmi calculator."""
    F = np.empty(hkl.shape[0], dtype=np.complex128)
    for i in range(hkl.shape[0]):
        F[i] = calc.calculate_sf_from_small_structure(
            st, (int(hkl[i, 0]), int(hkl[i, 1]), int(hkl[i, 2]))
        )
    return F


def _calc_unity(
    crystal: Crystal,
    hkl: NDArray[np.int32],
    b_factors: NDArray[np.float64],
    recip: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Compute F(hkl) with f_j = 1 for all atoms (direct summation in Python)."""
    inv_cell = np.linalg.inv(crystal.cell)
    frac_pos = crystal.positions @ inv_cell.T  # (N, 3)
    occupancies = np.asarray(crystal.occupancies, dtype=np.float64)  # (N,)

    q_vecs = hkl.astype(np.float64) @ recip  # (M, 3)
    s_sq = (np.linalg.norm(q_vecs, axis=1) / 2.0) ** 2  # (M,)

    phases = 2.0 * np.pi * (hkl.astype(np.float64) @ frac_pos.T)  # (M, N)
    dw = np.exp(-b_factors[np.newaxis, :] * s_sq[:, np.newaxis])  # (M, N)
    weighted = occupancies[np.newaxis, :] * dw * np.exp(1j * phases)
    return cast(NDArray[np.complex128], weighted.sum(axis=1).astype(np.complex128))
