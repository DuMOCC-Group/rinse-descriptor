"""Tests for the RINSE descriptor.

Structures tested:
  - NaCl (rocksalt)
  - Si (diamond cubic)
  - ylid
  - 1-Methylpiperazinium oxalate dihydrate

Properties verified:
  - Descriptor shape: 128 elements by default (flat 1-D), (8, 16) when flatten=False
  - Consistency: two calls with identical input return identical output
  - Non-negativity
  - Atom substitution sensitivity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rinse_descriptor import RinseParams, descriptor, descriptor_many, load_cif
from rinse_descriptor._descriptor import compute_power_spectrum
from rinse_descriptor._structure_factors import ReflectionList, compute_structure_factors

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def nacl():
    return load_cif(FIXTURES_DIR / "nacl.cif")


@pytest.fixture(scope="module")
def silicon():
    return load_cif(FIXTURES_DIR / "si.cif")


@pytest.fixture(scope="module")
def ylid():
    return load_cif(FIXTURES_DIR / "ylid.cif")


@pytest.fixture(scope="module")
def params() -> RinseParams:
    # Use small n_max/l_max for speed in unit tests; flatten=False for shape comparisons
    return RinseParams(n_max=8, l_max=8, sin_theta_over_lambda_max=1.0, flatten=False)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestDescriptorShape:
    def test_default_shape_nacl(self, nacl: object) -> None:
        x = descriptor(nacl)
        assert x.ndim == 1, f"Expected 1-D vector, got shape {x.shape}"
        assert x.shape[0] == RinseParams().descriptor_length

    def test_default_shape_silicon(self, silicon: object) -> None:
        x = descriptor(silicon)
        assert x.ndim == 1
        assert x.shape[0] == RinseParams().descriptor_length

    def test_default_shape_ylid(self, ylid: object) -> None:
        x = descriptor(ylid)
        assert x.ndim == 1
        assert x.shape[0] == RinseParams().descriptor_length

    def test_matrix_shape_with_flatten_false(self, nacl: object) -> None:
        params = RinseParams(flatten=False)
        x = descriptor(nacl, params=params)
        assert x.shape == params.descriptor_shape

    def test_custom_params_shape(self, nacl: object, params: RinseParams) -> None:
        # params fixture has flatten=False
        x = descriptor(nacl, params=params)
        assert x.shape == params.descriptor_shape

    def test_descriptor_many_shape(self, nacl: object, silicon: object, ylid: object) -> None:
        params = RinseParams(flatten=False)
        X = descriptor_many([nacl, silicon, ylid], params=params)
        assert X.shape == (3, 8, 16), f"Expected (3, 8, 16), got {X.shape}"

    def test_descriptor_many_flat_default(self, nacl: object, silicon: object) -> None:
        X = descriptor_many([nacl, silicon])
        default_len = RinseParams().descriptor_length
        assert X.shape == (2, default_len), f"Expected (2, {default_len}), got {X.shape}"

    def test_descriptor_accepts_cif_path(self) -> None:
        x = descriptor(FIXTURES_DIR / "nacl.cif")
        assert x.ndim == 1
        assert x.shape[0] == RinseParams().descriptor_length

    def test_descriptor_accepts_cif_string(self) -> None:
        x = descriptor(str(FIXTURES_DIR / "nacl.cif"))
        assert x.ndim == 1
        assert x.shape[0] == RinseParams().descriptor_length


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_result_twice(self, ylid: object, params: RinseParams) -> None:
        x1 = descriptor(ylid, params=params)
        x2 = descriptor(ylid, params=params)
        np.testing.assert_array_equal(x1, x2)

    def test_path_vs_xrs_same_result(self, nacl: object) -> None:
        x_xrs = descriptor(nacl)
        x_path = descriptor(FIXTURES_DIR / "nacl.cif")
        np.testing.assert_array_equal(x_xrs, x_path)


# ---------------------------------------------------------------------------
# Non-negativity (power spectrum is always ≥ 0)
# ---------------------------------------------------------------------------


class TestNonNegativity:
    @pytest.mark.parametrize("atoms_fixture", ["nacl", "silicon", "ylid"])
    def test_non_negative(self, request: pytest.FixtureRequest, atoms_fixture: str) -> None:
        atoms = request.getfixturevalue(atoms_fixture)
        x = descriptor(atoms)
        assert np.all(x >= -1e-12), f"Negative descriptor values found: min={x.min()}"


# ---------------------------------------------------------------------------
# Structure factor module smoke tests
# ---------------------------------------------------------------------------


class TestStructureFactors:
    def test_default_intensity_normalisation_is_double_exponential(self, ylid: object) -> None:
        assert RinseParams().intensity_normalisation == "double_exponential"
        assert RinseParams().intensity_falloff == "debye_waller"
        assert RinseParams().intensity_falloff_u_iso == 0.05
        assert RinseParams().use_reported_adps is True

        refls_default = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
        )
        refls_explicit = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            intensity_normalisation="double_exponential",
            intensity_falloff="debye_waller",
            intensity_falloff_u_iso=0.05,
        )
        np.testing.assert_allclose(refls_default.intensities, refls_explicit.intensities)

    def test_double_exponential_intensity_normalisation_is_finite(self, ylid: object) -> None:
        refls = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            intensity_normalisation="double_exponential",
            intensity_falloff="none",
        )
        assert np.all(np.isfinite(refls.intensities))
        assert np.all(refls.intensities >= 0.0)

    def test_debye_waller_intensity_falloff_suppresses_high_resolution(self, ylid: object) -> None:
        refls_no_falloff = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            intensity_normalisation="empirical",
            intensity_falloff="none",
        )
        refls_debye_waller = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            intensity_normalisation="empirical",
            intensity_falloff="debye_waller",
            intensity_falloff_u_iso=0.05,
        )

        s = 0.5 * refls_debye_waller.q_magnitudes
        high_s = s > np.quantile(s, 0.9)
        np.testing.assert_allclose(
            refls_debye_waller.intensities / refls_no_falloff.intensities,
            np.exp(-16.0 * np.pi**2 * 0.05 * s * s),
            rtol=1e-12,
        )
        assert np.mean(refls_debye_waller.intensities[high_s]) < np.mean(
            refls_no_falloff.intensities[high_s]
        )
        assert np.all(refls_debye_waller.intensities <= refls_no_falloff.intensities + 1e-12)

    def test_invalid_debye_waller_u_iso_rejected(self) -> None:
        with pytest.raises(ValueError, match="intensity_falloff_u_iso"):
            RinseParams(intensity_falloff_u_iso=-0.05)

    def test_reflection_count_positive(self, nacl: object) -> None:
        refls = compute_structure_factors(nacl, sin_theta_over_lambda_max=1.0)
        assert len(refls) > 0

    def test_q_magnitudes_within_cutoff(self, silicon: object) -> None:
        cutoff = 1.5
        refls = compute_structure_factors(silicon, sin_theta_over_lambda_max=cutoff)
        assert np.all(refls.q_magnitudes <= 2 * cutoff + 1e-6)

    def test_intensities_non_negative(self, ylid: object) -> None:
        refls = compute_structure_factors(ylid, structure_factor_type="F2")
        assert np.all(refls.intensities >= 0.0)

    @pytest.mark.parametrize("ff_type", ["xray", "electron", "neutron"])
    def test_form_factor_types(self, nacl: object, ff_type: str) -> None:
        refls = compute_structure_factors(
            nacl,
            sin_theta_over_lambda_max=1.0,
            form_factor_type=ff_type,
        )
        assert len(refls) > 0

    @pytest.mark.parametrize("sf_type", ["F2", "F"])
    def test_structure_factor_types(self, silicon: object, sf_type: str) -> None:
        refls = compute_structure_factors(
            silicon,
            sin_theta_over_lambda_max=1.0,
            structure_factor_type=sf_type,
        )
        assert refls.intensities.shape == (len(refls),)

    def test_empirical_intensity_normalisation_is_finite(self, ylid: object) -> None:
        refls = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            intensity_normalisation="empirical",
            intensity_falloff="none",
        )
        assert np.all(np.isfinite(refls.intensities))
        assert np.all(refls.intensities >= 0.0)

    def test_empirical_intensity_normalisation_flattens_dense_fixture(self, ylid: object) -> None:
        refls = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            intensity_normalisation="empirical",
            intensity_falloff="none",
        )
        s = 0.5 * refls.q_magnitudes
        order = np.argsort(s)
        bin_means = [
            float(np.mean(refls.intensities[idx]))
            for idx in np.array_split(order, 12)
            if len(idx) > 0
        ]
        np.testing.assert_allclose(bin_means, np.ones(len(bin_means)), rtol=0.35, atol=0.35)

    def test_empirical_f_output_squares_to_intensity_output(self, ylid: object) -> None:
        refls_f2 = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            structure_factor_type="F2",
            intensity_normalisation="empirical",
            intensity_falloff="debye_waller",
        )
        refls_f = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=0.6,
            structure_factor_type="F",
            intensity_normalisation="empirical",
            intensity_falloff="debye_waller",
        )

        np.testing.assert_array_equal(refls_f.hkl, refls_f2.hkl)
        np.testing.assert_allclose(refls_f.intensities**2, refls_f2.intensities, rtol=1e-12)


# ---------------------------------------------------------------------------
# CIF occupancy / ADP handling
# ---------------------------------------------------------------------------


class TestCifOccupancyAndAdp:
    def test_fractional_occupancy_affects_intensities(self, tmp_path: Path) -> None:
        cif_occ_1 = """data_occ1
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Si1 Si 0.0 0.0 0.0 1.0
"""
        cif_occ_half = cif_occ_1.replace(
            "  Si1 Si 0.0 0.0 0.0 1.0",
            "  Si1 Si 0.0 0.0 0.0 0.5",
        )

        p1 = tmp_path / "occ1.cif"
        p2 = tmp_path / "occ05.cif"
        p1.write_text(cif_occ_1)
        p2.write_text(cif_occ_half)

        xrs1 = load_cif(str(p1))
        xrs05 = load_cif(str(p2))

        r1 = compute_structure_factors(
            xrs1,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
            intensity_normalisation="none",
            use_reported_adps=False,
        )
        r05 = compute_structure_factors(
            xrs05,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
            intensity_normalisation="none",
            use_reported_adps=False,
        )

        # For a single atom at the origin, occupancy scales amplitudes linearly,
        # hence intensities scale quadratically.
        np.testing.assert_allclose(
            r05.intensities,
            0.25 * r1.intensities,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_anisotropic_displacement_parsed_from_cif(self, tmp_path: Path) -> None:
        cif_aniso = """data_aniso
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Si1 Si 0.1 0.2 0.3 1.0
loop_
    _atom_site_aniso_label
    _atom_site_aniso_U_11
    _atom_site_aniso_U_22
    _atom_site_aniso_U_33
    _atom_site_aniso_U_12
    _atom_site_aniso_U_13
    _atom_site_aniso_U_23
    Si1 0.010 0.020 0.030 0.001 0.002 0.003
"""
        path = tmp_path / "aniso.cif"
        path.write_text(cif_aniso)

        xrs = load_cif(str(path))
        # At least one scatterer should have anisotropic displacement parameters set
        sc = xrs.scatterers()[0]
        assert sc.flags.use_u_aniso(), "Expected anisotropic ADPs to be parsed from CIF"

        # Ensure structure-factor computation succeeds with parsed ADP metadata.
        refls = compute_structure_factors(
            xrs,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
        )
        assert len(refls) > 0

        refls_explicit = compute_structure_factors(
            xrs,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
            use_reported_adps=True,
        )
        np.testing.assert_allclose(refls.intensities, refls_explicit.intensities)

        refls_fixed_u = compute_structure_factors(
            xrs,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
            intensity_normalisation="none",
            use_reported_adps=False,
        )
        refls_reported_u = compute_structure_factors(
            xrs,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
            intensity_normalisation="none",
            use_reported_adps=True,
        )
        assert not np.allclose(refls_fixed_u.intensities, refls_reported_u.intensities)


# ---------------------------------------------------------------------------
# Additional invariance / sensitivity tests
# ---------------------------------------------------------------------------


class TestAdditionalInvariances:
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
        return float(np.dot(a_flat, b_flat) / (denom + 1e-20))

    def test_reflection_permutation_invariance(self, silicon: object) -> None:
        params = RinseParams(
            n_max=8,
            l_max=8,
            sin_theta_over_lambda_max=1.0,
            flatten=False,
        )
        reflections = compute_structure_factors(
            silicon,
            sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
            form_factor_type="xray",
            structure_factor_type="F2",
        )

        rng = np.random.default_rng(123)
        perm = rng.permutation(len(reflections))
        reflections_perm = ReflectionList(
            hkl=reflections.hkl[perm],
            q_vectors=reflections.q_vectors[perm],
            q_magnitudes=reflections.q_magnitudes[perm],
            intensities=reflections.intensities[perm],
        )

        p_ref = compute_power_spectrum(reflections, params=params)
        p_perm = compute_power_spectrum(reflections_perm, params=params)
        np.testing.assert_allclose(p_ref, p_perm, rtol=1e-12, atol=1e-12)

    def test_intensity_scaling_invariant(self, ylid: object) -> None:
        params = RinseParams(
            n_max=8,
            l_max=8,
            sin_theta_over_lambda_max=1.0,
            flatten=False,
            log1p=False,
        )
        reflections = compute_structure_factors(
            ylid,
            sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
            form_factor_type="xray",
            structure_factor_type="F2",
        )

        scale = 3.0
        reflections_scaled = ReflectionList(
            hkl=reflections.hkl.copy(),
            q_vectors=reflections.q_vectors.copy(),
            q_magnitudes=reflections.q_magnitudes.copy(),
            intensities=reflections.intensities * scale,
        )

        p_ref = compute_power_spectrum(reflections, params=params)
        p_scaled = compute_power_spectrum(reflections_scaled, params=params)
        assert np.allclose(p_scaled, p_ref, rtol=1e-4, atol=1e-8)

    def test_atom_substitution_changes_descriptor(self, nacl: object, silicon: object) -> None:
        params = RinseParams(
            n_max=8,
            l_max=8,
            sin_theta_over_lambda_max=1.0,
            flatten=False,
        )
        # NaCl and Si are chemically distinct: their xray descriptors must differ.
        x_nacl = descriptor(nacl, params=params)
        x_si = descriptor(silicon, params=params)
        assert not np.allclose(x_nacl, x_si, rtol=1e-4, atol=1e-6)
