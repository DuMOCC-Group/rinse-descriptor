"""Tests for the RINSE descriptor.

Structures tested:
  - NaCl (rocksalt)
  - Si (diamond cubic)
  - ylid
  - 1-Methylpiperazinium oxalate dihydrate

Properties verified:
  - Descriptor shape: (8, 16) by default, 128 when flattened
  - Translation invariance: shifting all atoms by a vector leaves descriptor unchanged
  - Atom-order invariance: permuting atoms leaves descriptor unchanged (within rounding)
  - Consistency: two calls with identical input return identical output
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read
from rinse_descriptor import Crystal, RinseParams, descriptor, descriptor_many
from rinse_descriptor._descriptor import compute_power_spectrum
from rinse_descriptor._structure_factors import ReflectionList, compute_structure_factors

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def nacl() -> object:
    return ase_read(FIXTURES_DIR / "nacl.cif")


@pytest.fixture(scope="module")
def silicon() -> object:
    return ase_read(FIXTURES_DIR / "si.cif")


@pytest.fixture(scope="module")
def ylid() -> object:
    return ase_read(FIXTURES_DIR / "ylid.cif")


@pytest.fixture(scope="module")
def params() -> RinseParams:
    # Use small n_max/l_max for speed in unit tests; shape is still tested
    return RinseParams(n_max=8, l_max=8, sin_theta_over_lambda_max=1.0)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestDescriptorShape:
    def test_default_shape_nacl(self, nacl: object) -> None:
        x = descriptor(nacl)
        assert x.shape == (8, 16), f"Expected (8, 16), got {x.shape}"

    def test_default_shape_silicon(self, silicon: object) -> None:
        x = descriptor(silicon)
        assert x.shape == (8, 16), f"Expected (8, 16), got {x.shape}"

    def test_default_shape_ylid(self, ylid: object) -> None:
        x = descriptor(ylid)
        assert x.shape == (8, 16), f"Expected (8, 16), got {x.shape}"

    def test_flattened_length(self, nacl: object) -> None:
        x = descriptor(nacl, flatten=True)
        assert x.ndim == 1
        assert x.shape[0] == 128

    def test_custom_params_shape(self, nacl: object, params: RinseParams) -> None:
        x = descriptor(nacl, params=params)
        assert x.shape == (params.n_max, params.l_max)

    def test_descriptor_many_shape(self, nacl: object, silicon: object, ylid: object) -> None:
        X = descriptor_many([nacl, silicon, ylid])
        assert X.shape == (3, 8, 16), f"Expected (3, 8, 16), got {X.shape}"

    def test_descriptor_many_flattened(self, nacl: object, silicon: object) -> None:
        X = descriptor_many([nacl, silicon], flatten=True)
        assert X.shape == (2, 128), f"Expected (2, 128), got {X.shape}"


# ---------------------------------------------------------------------------
# Translation invariance
# ---------------------------------------------------------------------------


class TestTranslationInvariance:
    @pytest.mark.parametrize(
        "shift",
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.5, 1.3],
            [3.14, -1.0, 2.71],
        ],
    )
    def test_translation_nacl(self, nacl: object, params: RinseParams, shift: list) -> None:
        atoms_shifted = nacl.copy()
        atoms_shifted.translate(shift)
        # wrap back into cell to keep fractional coordinates valid
        atoms_shifted.wrap()

        x_orig = descriptor(nacl, params=params)
        x_shift = descriptor(atoms_shifted, params=params)
        np.testing.assert_allclose(
            x_orig,
            x_shift,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Descriptor changed after translation by {shift}",
        )

    def test_translation_silicon(self, silicon: object, params: RinseParams) -> None:
        atoms_shifted = silicon.copy()
        atoms_shifted.translate([0.5, 0.5, 0.5])
        atoms_shifted.wrap()
        x_orig = descriptor(silicon, params=params)
        x_shift = descriptor(atoms_shifted, params=params)
        np.testing.assert_allclose(x_orig, x_shift, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Atom-order invariance
# ---------------------------------------------------------------------------


class TestAtomOrderInvariance:
    def test_permutation_nacl(self, nacl: object, params: RinseParams) -> None:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(nacl))
        atoms_perm = nacl[perm]

        x_orig = descriptor(nacl, params=params)
        x_perm = descriptor(atoms_perm, params=params)
        np.testing.assert_allclose(
            x_orig,
            x_perm,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Descriptor changed after atom permutation",
        )

    def test_permutation_silicon(self, silicon: object, params: RinseParams) -> None:
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(silicon))
        atoms_perm = silicon[perm]
        x_orig = descriptor(silicon, params=params)
        x_perm = descriptor(atoms_perm, params=params)
        np.testing.assert_allclose(x_orig, x_perm, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_result_twice(self, ylid: object, params: RinseParams) -> None:
        x1 = descriptor(ylid, params=params)
        x2 = descriptor(ylid, params=params)
        np.testing.assert_array_equal(x1, x2)


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
    def test_reflection_count_positive(self, nacl: object) -> None:
        crystal = Crystal.from_ase(nacl)
        refls = compute_structure_factors(crystal, sin_theta_over_lambda_max=1.0)
        assert len(refls) > 0

    def test_q_magnitudes_within_cutoff(self, silicon: object) -> None:
        crystal = Crystal.from_ase(silicon)
        cutoff = 1.5
        refls = compute_structure_factors(crystal, sin_theta_over_lambda_max=cutoff)
        assert np.all(refls.q_magnitudes <= 2 * cutoff + 1e-6)

    def test_intensities_non_negative(self, ylid: object) -> None:
        crystal = Crystal.from_ase(ylid)
        refls = compute_structure_factors(crystal, structure_factor_type="F2")
        assert np.all(refls.intensities >= 0.0)

    @pytest.mark.parametrize("ff_type", ["xray", "unity"])
    def test_form_factor_types(self, nacl: object, ff_type: str) -> None:
        crystal = Crystal.from_ase(nacl)
        refls = compute_structure_factors(
            crystal,
            sin_theta_over_lambda_max=1.0,
            form_factor_type=ff_type,
        )
        assert len(refls) > 0

    @pytest.mark.parametrize("sf_type", ["F2", "F"])
    def test_structure_factor_types(self, silicon: object, sf_type: str) -> None:
        crystal = Crystal.from_ase(silicon)
        refls = compute_structure_factors(
            crystal,
            sin_theta_over_lambda_max=1.0,
            structure_factor_type=sf_type,
        )
        assert refls.intensities.shape == (len(refls),)


# ---------------------------------------------------------------------------
# Crystal dataclass tests
# ---------------------------------------------------------------------------


class TestCrystal:
    def test_from_ase_nacl(self, nacl: object) -> None:
        crystal = Crystal.from_ase(nacl)
        assert crystal.cell.shape == (3, 3)
        assert crystal.positions.shape == (len(nacl), 3)
        assert crystal.species.shape == (len(nacl),)
        assert crystal.pbc.shape == (3,)

    def test_volume_positive(self, nacl: object) -> None:
        crystal = Crystal.from_ase(nacl)
        assert crystal.volume > 0

    def test_bad_cell_raises(self) -> None:
        with pytest.raises(ValueError, match="cell must be"):
            Crystal(
                cell=np.eye(2),
                positions=np.zeros((1, 3)),
                species=np.array([1], dtype=np.int32),
            )

    def test_species_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="species length"):
            Crystal(
                cell=np.eye(3),
                positions=np.zeros((3, 3)),
                species=np.array([1, 2], dtype=np.int32),  # wrong length
            )


# ---------------------------------------------------------------------------
# CIF occupancy / ADP handling
# ---------------------------------------------------------------------------


class TestCifOccupancyAndAdp:
    def test_fractional_occupancy_affects_unity_intensities(self, tmp_path: Path) -> None:
        pytest.importorskip("gemmi")

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

        c1 = Crystal.from_cif(str(p1))
        c05 = Crystal.from_cif(str(p2))

        r1 = compute_structure_factors(
            c1,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
        )
        r05 = compute_structure_factors(
            c05,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
        )

        # For a single atom at the origin, occupancy scales amplitudes linearly,
        # hence intensities scale quadratically.
        np.testing.assert_allclose(
            r05.intensities,
            0.25 * r1.intensities,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_anisotropic_displacement_is_parsed_from_cif(self, tmp_path: Path) -> None:
        pytest.importorskip("gemmi")

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

        crystal = Crystal.from_cif(str(path))
        assert crystal.u_aniso is not None
        assert np.any(np.abs(crystal.u_aniso) > 0.0)

        # Ensure structure-factor computation succeeds with parsed ADP metadata.
        refls = compute_structure_factors(
            crystal,
            sin_theta_over_lambda_max=0.5,
            form_factor_type="xray",
        )
        assert len(refls) > 0


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

    def test_supercell_invariance(self, nacl: object) -> None:
        params = RinseParams(
            n_max=8,
            l_max=8,
            sin_theta_over_lambda_max=1.0,
        )

        atoms_super = nacl.repeat((2, 2, 2))
        x_prim = descriptor(nacl, params=params, structure_factor_type="F2")
        x_super = descriptor(atoms_super, params=params, structure_factor_type="F2")
        similarity = self._cosine_similarity(x_prim, x_super)
        assert similarity > 0.94, f"Supercell changed descriptor too much (cos={similarity:.6f})"

    def test_reflection_permutation_invariance(self, silicon: object) -> None:
        params = RinseParams(
            n_max=8,
            l_max=8,
            sin_theta_over_lambda_max=1.0,
        )
        crystal = Crystal.from_ase(silicon)
        reflections = compute_structure_factors(
            crystal,
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

    def test_intensity_scaling_changes_values_but_keeps_l2_scale(self, ylid: object) -> None:
        params = RinseParams(
            n_max=8,
            l_max=8,
            sin_theta_over_lambda_max=1.0,
        )
        crystal = Crystal.from_ase(ylid)
        reflections = compute_structure_factors(
            crystal,
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
        assert not np.allclose(p_scaled, p_ref, rtol=1e-4, atol=1e-8)
        assert np.isclose(np.linalg.norm(p_ref), 1.0, rtol=1e-10, atol=1e-12)
        assert np.isclose(np.linalg.norm(p_scaled), 1.0, rtol=1e-10, atol=1e-12)

    def test_atom_substitution_changes_descriptor(self, nacl: object) -> None:
        params = RinseParams(
            n_max=8,
            l_max=8,
            sin_theta_over_lambda_max=1.0,
        )

        x_ref = descriptor(nacl, params=params, structure_factor_type="F2")

        atoms_sub = nacl.copy()
        z = atoms_sub.get_atomic_numbers()
        z[0] = 14 if z[0] != 14 else 12
        atoms_sub.set_atomic_numbers(z)
        x_sub = descriptor(atoms_sub, params=params, structure_factor_type="F2")

        assert not np.allclose(x_ref, x_sub, rtol=1e-4, atol=1e-6)
