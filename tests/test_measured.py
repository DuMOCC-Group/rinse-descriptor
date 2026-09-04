"""Tests for the measured-data (model-free) descriptor path.

Measured reflections are synthesised from the NaCl model so that the
model-free path can be validated against the model-based path (they must
agree when the supplied intensities are the calculated ones).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rinse_descriptor import (
    MeasuredCompletenessError,
    RinseParams,
    descriptor,
    descriptor_from_hkl,
    descriptor_hash,
    load_crystal_symmetry,
    load_hkl,
)
from rinse_descriptor._structure_factors import compute_structure_factors

FIXTURES_DIR = Path(__file__).parent / "fixtures"
NACL_CIF = FIXTURES_DIR / "nacl.cif"
NACL_EXP_RES = FIXTURES_DIR / "nacl_exp.res"
NACL_EXP_HKL = FIXTURES_DIR / "nacl_exp_gener.hkl"


def _model_reflections(cif: Path, factor: float):
    """Full-sphere (hkl, intensity) from the model out to ``factor`` x q_max."""
    from rinse_descriptor import load_cif

    params = RinseParams()
    xrs = load_cif(cif)
    refls = compute_structure_factors(xrs, sin_theta_over_lambda_max=0.5 * factor * params.q_max)
    return refls.hkl.copy(), refls.intensities.copy()


def _positive_hemisphere(hkl: np.ndarray, intensity: np.ndarray):
    """Keep one Friedel mate of each pair (a hemisphere of reflections)."""
    keep = (
        (hkl[:, 2] > 0)
        | ((hkl[:, 2] == 0) & (hkl[:, 1] > 0))
        | ((hkl[:, 2] == 0) & (hkl[:, 1] == 0) & (hkl[:, 0] > 0))
    )
    return hkl[keep], intensity[keep]


# ---------------------------------------------------------------------------
# Three-way equivalence on real fixtures
# ---------------------------------------------------------------------------


class TestThreeWayEquivalence:
    """The same NaCl structure via three independent routes must agree.

    1. ``nacl.cif``           – model in its published space group (Fm-3m).
    2. ``nacl_exp.res``       – the same model expanded to P1.
    3. ``nacl_exp_gener.hkl`` – *measured* intensities (no model), with the
       cell/symmetry taken from ``nacl_exp.res``.
    """

    def test_all_three_hashes_agree(self) -> None:
        d_cif = descriptor(NACL_CIF)
        d_res_p1 = descriptor(NACL_EXP_RES)
        d_hkl = descriptor_from_hkl(NACL_EXP_HKL, NACL_EXP_RES)
        assert descriptor_hash(d_cif) == descriptor_hash(d_res_p1) == descriptor_hash(d_hkl)

    def test_model_routes_identical(self) -> None:
        # cif and its P1 expansion are the same electron density -> same descriptor.
        d_cif = descriptor(NACL_CIF)
        d_res_p1 = descriptor(NACL_EXP_RES)
        np.testing.assert_allclose(d_res_p1, d_cif, atol=1e-9)

    def test_measured_route_matches_model(self) -> None:
        # Measured intensities are rounded integers, so allow a small tolerance.
        d_cif = descriptor(NACL_CIF)
        d_hkl = descriptor_from_hkl(NACL_EXP_HKL, NACL_EXP_RES)
        cosine = float(d_cif @ d_hkl / (np.linalg.norm(d_cif) * np.linalg.norm(d_hkl)))
        assert cosine > 0.9999
        np.testing.assert_allclose(d_hkl, d_cif, atol=5e-3)


# ---------------------------------------------------------------------------
# Consistency with the model-based path
# ---------------------------------------------------------------------------


class TestMeasuredMatchesModel:
    def test_full_sphere_matches_model(self) -> None:
        hkl, intensity = _model_reflections(NACL_CIF, factor=1.2)
        x_meas = descriptor_from_hkl((hkl, intensity), NACL_CIF)
        x_model = descriptor(NACL_CIF)
        np.testing.assert_allclose(x_meas, x_model, atol=1e-10)

    def test_hemisphere_matches_full_sphere(self) -> None:
        hkl, intensity = _model_reflections(NACL_CIF, factor=1.2)
        hemi_hkl, hemi_I = _positive_hemisphere(hkl, intensity)
        assert hemi_hkl.shape[0] < hkl.shape[0]
        x_hemi = descriptor_from_hkl((hemi_hkl, hemi_I), NACL_CIF)
        x_full = descriptor_from_hkl((hkl, intensity), NACL_CIF)
        np.testing.assert_allclose(x_hemi, x_full, atol=1e-10)


# ---------------------------------------------------------------------------
# Completeness enforcement
# ---------------------------------------------------------------------------


class TestCompleteness:
    def test_truncated_sphere_raises(self) -> None:
        # Data only out to 1.0 x q_max leaves the 1.0-1.2 shell entirely missing.
        hkl, intensity = _model_reflections(NACL_CIF, factor=1.0)
        with pytest.raises(MeasuredCompletenessError, match="Incomplete reflection sphere"):
            descriptor_from_hkl((hkl, intensity), NACL_CIF)

    def test_relaxed_tolerance_allows_truncation(self) -> None:
        hkl, intensity = _model_reflections(NACL_CIF, factor=1.0)
        x = descriptor_from_hkl((hkl, intensity), NACL_CIF, max_missing_fraction=1.0)
        assert x.shape[0] == RinseParams().descriptor_length


# ---------------------------------------------------------------------------
# HKL file parsing
# ---------------------------------------------------------------------------


class TestLoadHkl:
    def test_free_format(self, tmp_path: Path) -> None:
        content = (
            "  1  0  0   10.50    0.30\n  2  0  0   20.00    0.40\n  0  0  0    0.00    0.00\n"
        )
        f = tmp_path / "free.hkl"
        f.write_text(content)
        hkl, intensity, sigma = load_hkl(f)
        assert hkl.shape == (2, 3)
        np.testing.assert_array_equal(hkl[0], [1, 0, 0])
        assert intensity[1] == pytest.approx(20.0)
        assert sigma[0] == pytest.approx(0.30)

    def test_fixed_width_hklf4(self, tmp_path: Path) -> None:
        # 3I4, 2F8.2 with indices running together (no separating spaces).
        content = "  -1  -1 -10  100.00    5.00\n   0   0   0    0.00    0.00\n"
        f = tmp_path / "fixed.hkl"
        f.write_text(content)
        hkl, intensity, _ = load_hkl(f)
        assert hkl.shape[0] == 1
        assert intensity[0] == pytest.approx(100.0)

    def test_terminator_stops_reading(self, tmp_path: Path) -> None:
        content = "1 0 0 10 1\n0 0 0 0 0\n5 5 5 99 9\n"
        f = tmp_path / "term.hkl"
        f.write_text(content)
        hkl, _, _ = load_hkl(f)
        assert hkl.shape[0] == 1

    def test_empty_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.hkl"
        f.write_text("\n# comment only\n")
        with pytest.raises(ValueError, match="no reflections"):
            load_hkl(f)


# ---------------------------------------------------------------------------
# Cell / symmetry loading
# ---------------------------------------------------------------------------


class TestLoadCrystalSymmetry:
    def test_from_cif(self) -> None:
        cs = load_crystal_symmetry(NACL_CIF)
        assert cs.space_group().type().lookup_symbol().startswith("F m -3 m")

    def test_from_cell_params(self) -> None:
        cs = load_crystal_symmetry((5.64, 5.64, 5.64, 90, 90, 90))
        assert cs.space_group().type().lookup_symbol().strip() == "P 1"

    def test_space_group_override(self) -> None:
        cs = load_crystal_symmetry((5.64, 5.64, 5.64, 90, 90, 90), space_group="F m -3 m")
        assert cs.space_group().type().lookup_symbol().startswith("F m -3 m")

    def test_cell_from_params_matches_file(self) -> None:
        hkl, intensity = _model_reflections(NACL_CIF, factor=1.2)
        x_file = descriptor_from_hkl((hkl, intensity), NACL_CIF)
        x_params = descriptor_from_hkl(
            (hkl, intensity),
            (5.6035, 5.6035, 5.6035, 90, 90, 90),
            space_group="F m -3 m",
        )
        np.testing.assert_allclose(x_file, x_params, atol=1e-10)
