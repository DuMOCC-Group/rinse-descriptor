"""RINSE – Reciprocal-space INvariant Spectral Embedding.

Quick start
-----------
>>> from ase.build import bulk
>>> from rinse_descriptor import descriptor, RinseParams
>>> atoms = bulk("NaCl", "rocksalt", a=5.64)
>>> x = descriptor(atoms)           # 1-D vector by default
>>> x.shape == (RinseParams().descriptor_length,)
True

>>> X = descriptor_many([atoms, atoms])
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._crystal import Crystal
from ._descriptor import (
    RinseParams,
    compute_power_spectrum,
    normalise_power_spectrum,
    power_spectrum_to_vector,
)
from ._hash import DEFAULT_HASH_WORDS, descriptor_hash, hash_to_bits
from ._structure_factors import (
    FormFactorType,
    ReflectionList,
    StructureFactorType,
    compute_structure_factors,
)

__version__ = "0.1.0"
__all__ = [
    "Crystal",
    "RinseParams",
    "FormFactorType",
    "StructureFactorType",
    "ReflectionList",
    "compute_structure_factors",
    "compute_power_spectrum",
    "normalise_power_spectrum",
    "descriptor",
    "descriptor_many",
    "descriptor_hash",
    "hash_to_bits",
    "DEFAULT_HASH_WORDS",
]


def descriptor(
    atoms: object,
    *,
    params: RinseParams | None = None,
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron", "unity"] = "xray",
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
    b_factors: NDArray[np.float64] | None = None,
    debug: bool = False,
) -> NDArray[np.float64]:
    """Compute the RINSE descriptor for a single structure.

    Parameters
    ----------
    atoms:
        An :class:`ase.Atoms` object, a :class:`~rinse_descriptor.Crystal`, or a path
        to a CIF file (str / :class:`pathlib.Path`).
    params:
        Descriptor hyper-parameters.  Uses :class:`RinseParams` defaults if *None*
        (see :class:`RinseParams` for current values).
        ``params.log1p`` and ``params.l2`` control post-processing normalisation.
        ``params.flatten`` controls whether the output is a 1-D vector (default *True*)
        or the 2-D ``(n_max, n_l_levels)`` matrix.
    form_factor_type:
        ``"xray"`` | ``"electron"`` | ``"neutron"`` | ``"unity"``.
    structure_factor_type:
        ``"F2"`` (default) | ``"F"``.
    b_factors:
        Per-atom isotropic B-factors in Å².  *None* → unit B-factors (1 Å²).

    Returns
    -------
    ndarray of shape ``(n_max * n_l_levels,)`` [default, flatten=True] or
    ``(n_max, n_l_levels)`` [flatten=False].
    """
    import pathlib

    t0 = time.perf_counter()

    if debug:
        if isinstance(atoms, (str, pathlib.Path)):
            print(f"[rinse_descriptor] input: CIF {atoms}", file=sys.stderr)
        elif isinstance(atoms, Crystal):
            print(f"[rinse_descriptor] input: Crystal ({atoms.n_atoms} atoms)", file=sys.stderr)
        else:
            print(f"[rinse_descriptor] input: {type(atoms).__name__}", file=sys.stderr)

    _t = time.perf_counter()
    crystal = _to_crystal(atoms)
    if debug:
        print(
            f"[rinse_descriptor]   load structure:     {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"({crystal.n_atoms} atoms)",
            file=sys.stderr,
        )

    if params is None:
        params = RinseParams()

    _t = time.perf_counter()
    reflections = compute_structure_factors(
        crystal,
        sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
        form_factor_type=form_factor_type,
        structure_factor_type=structure_factor_type,
        b_factors=b_factors,
        debug=debug,
    )
    if debug:
        print(
            f"[rinse_descriptor]   structure factors:  {(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"({len(reflections)} reflections)",
            file=sys.stderr,
        )

    _t = time.perf_counter()
    P = compute_power_spectrum(reflections, params=params, debug=debug)
    if debug:
        print(
            f"[rinse_descriptor]   power spectrum:     {(time.perf_counter() - _t) * 1e3:8.2f} ms",
            file=sys.stderr,
        )
        print(
            f"[rinse_descriptor]   TOTAL:              {(time.perf_counter() - t0) * 1e3:8.2f} ms",
            file=sys.stderr,
        )

    return power_spectrum_to_vector(P) if params.flatten else P


def descriptor_many(
    structures: Sequence[object],
    *,
    params: RinseParams | None = None,
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron", "unity"] = "xray",
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
) -> NDArray[np.float64]:
    """Compute the RINSE descriptor for a list of structures.

    Parameters
    ----------
    structures:
        Iterable of :class:`ase.Atoms`, :class:`~rinse_descriptor.Crystal`, or CIF paths.
    params:
        Shared descriptor hyper-parameters.
        ``params.log1p`` and ``params.l2`` control post-processing normalisation.
    form_factor_type, structure_factor_type:
        Passed to :func:`descriptor`.
    flatten:
        If *True*, return shape (N, n_max*n_l_levels); otherwise (N, n_max, n_l_levels).

    Returns
    -------
    ndarray of shape (N, n_max*n_l_levels) [default, flatten=True] or
    (N, n_max, n_l_levels) [flatten=False].
    """
    results = [
        descriptor(
            s,
            params=params,
            form_factor_type=form_factor_type,
            structure_factor_type=structure_factor_type,
        )
        for s in structures
    ]
    return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _to_crystal(atoms: object) -> Crystal:
    """Accept ase.Atoms, Crystal, str (CIF path), or pathlib.Path."""
    import pathlib

    if isinstance(atoms, Crystal):
        return atoms
    if isinstance(atoms, (str, pathlib.Path)):
        return Crystal.from_cif(str(atoms))
    # Assume ASE Atoms (duck-typed)
    try:
        return Crystal.from_ase(atoms)
    except TypeError as exc:
        raise TypeError(
            f"Expected ase.Atoms, rinse_descriptor.Crystal, or a CIF file path, got {type(atoms)}"
        ) from exc
