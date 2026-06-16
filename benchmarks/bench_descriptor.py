"""Benchmarks for the RINSE descriptor (pytest-benchmark)."""

from __future__ import annotations

import pytest
from ase.build import bulk, make_supercell
from rinse import RinseParams, descriptor


@pytest.fixture(scope="module")
def nacl_unit():
    return bulk("NaCl", "rocksalt", a=5.6402)


@pytest.fixture(scope="module")
def nacl_supercell():
    atoms = bulk("NaCl", "rocksalt", a=5.6402)
    return make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])


@pytest.fixture(scope="module")
def params_default():
    return RinseParams()


@pytest.fixture(scope="module")
def params_small():
    return RinseParams(n_max=8, l_max=8, sin_theta_over_lambda_max=1.0)


def test_benchmark_nacl_unit(benchmark, nacl_unit, params_default):
    benchmark(descriptor, nacl_unit, params=params_default)


def test_benchmark_nacl_supercell(benchmark, nacl_supercell, params_default):
    benchmark(descriptor, nacl_supercell, params=params_default)


def test_benchmark_small_params(benchmark, nacl_unit, params_small):
    benchmark(descriptor, nacl_unit, params=params_small)
