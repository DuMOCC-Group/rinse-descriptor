"""RINSE – Reciprocal-space INvariant Spectral Embedding.

Quick start
-----------
>>> from rinse_descriptor import load_cif, descriptor, RinseParams
>>> xrs = load_cif("nacl.cif")
>>> x = descriptor(xrs)             # 1-D vector by default
>>> x.shape == (RinseParams().descriptor_length,)
True

Alternatively pass a CIF path directly::

    >>> x = descriptor("nacl.cif")

>>> X = descriptor_many([xrs, xrs])
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._crystal import load_cif
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
    "load_cif",
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
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron"] = "xray",
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
    debug: bool = False,
) -> NDArray[np.float64]:
    """Compute the RINSE descriptor for a single structure.

    Parameters
    ----------
    atoms:
        A :class:`cctbx.xray.structure` or a path to a CIF file
        (``str`` / :class:`pathlib.Path`).
    params:
        Descriptor hyper-parameters.  Uses :class:`RinseParams` defaults if *None*.
        ``params.log1p`` and ``params.l2`` control post-processing normalisation.
        ``params.flatten`` controls whether the output is a 1-D vector (default
        *True*) or the 2-D ``(n_max, n_l_levels)`` matrix.
    form_factor_type:
        ``"xray"`` | ``"electron"`` | ``"neutron"``.
    structure_factor_type:
        ``"F2"`` (default) | ``"F"``.

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
        else:
            print(f"[rinse_descriptor] input: {type(atoms).__name__}", file=sys.stderr)

    _t = time.perf_counter()
    xrs = _to_xrs(atoms)
    if debug:
        print(
            f"[rinse_descriptor]   load structure:     "
            f"{(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"({xrs.scatterers().size()} scatterers in asym unit)",
            file=sys.stderr,
        )

    if params is None:
        params = RinseParams()

    _t = time.perf_counter()
    reflections = compute_structure_factors(
        xrs,
        sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
        form_factor_type=form_factor_type,
        structure_factor_type=structure_factor_type,
        debug=debug,
    )
    if debug:
        print(
            f"[rinse_descriptor]   structure factors:  "
            f"{(time.perf_counter() - _t) * 1e3:8.2f} ms  "
            f"({len(reflections)} reflections)",
            file=sys.stderr,
        )

    _t = time.perf_counter()
    P = compute_power_spectrum(reflections, params=params, debug=debug)
    if debug:
        print(
            f"[rinse_descriptor]   power spectrum:     "
            f"{(time.perf_counter() - _t) * 1e3:8.2f} ms",
            file=sys.stderr,
        )
        print(
            f"[rinse_descriptor]   TOTAL:              "
            f"{(time.perf_counter() - t0) * 1e3:8.2f} ms",
            file=sys.stderr,
        )

    return power_spectrum_to_vector(P) if params.flatten else P


def descriptor_many(
    structures: Sequence[object],
    *,
    params: RinseParams | None = None,
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron"] = "xray",
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
) -> NDArray[np.float64]:
    """Compute the RINSE descriptor for a list of structures.

    Parameters
    ----------
    structures:
        Iterable of :class:`cctbx.xray.structure` objects or CIF paths.
    params:
        Shared descriptor hyper-parameters.
    form_factor_type, structure_factor_type:
        Passed to :func:`descriptor`.

    Returns
    -------
    ndarray of shape (N, descriptor_length) [default, flatten=True] or
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


def _to_xrs(atoms: object) -> Any:
    """Accept a cctbx xray.structure, str (CIF path), or pathlib.Path."""
    import pathlib

    if isinstance(atoms, (str, pathlib.Path)):
        return load_cif(atoms)
    # Accept any cctbx xray.structure (duck-typing; avoids a hard import of the C type)
    if hasattr(atoms, "structure_factors") and hasattr(atoms, "scatterers"):
        return atoms
    raise TypeError(
        f"Expected a cctbx xray.structure or a CIF file path, got {type(atoms)}"
    )
