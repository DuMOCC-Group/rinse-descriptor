"""RINSE – Reciprocal-space INvariant Spectral Embedding.

Quick start
-----------
>>> from rinse_descriptor import load_cif, descriptor, RinseParams
>>> xrs = load_cif("nacl.cif")
>>> x = descriptor(xrs)             # 1-D vector by default
>>> x.shape == (RinseParams().descriptor_length,)
True

Alternatively pass a structure path directly::

    >>> x = descriptor("nacl.cif")

>>> X = descriptor_many([xrs, xrs])
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cctbx_import_patch import patch_cctbx_imports

patch_cctbx_imports()

from ._crystal import load_cif, load_res, load_structure  # noqa: E402
from ._descriptor import (  # noqa: E402
    RinseParams,
    compute_power_spectrum,
    normalise_power_spectrum,
    power_spectrum_to_vector,
)
from ._hash import DEFAULT_HASH_WORDS, descriptor_hash, hash_to_bits  # noqa: E402
from ._measured import (  # noqa: E402
    MeasuredCompletenessError,
    load_crystal_symmetry,
    load_hkl,
    reflections_from_measured,
)
from ._structure_factors import (  # noqa: E402
    FormFactorType,
    ReflectionList,
    StructureFactorType,
    compute_structure_factors,
)

__version__ = "0.1.0"


class _Unset:
    """Sentinel: argument not supplied; fall back to the value from ``params``."""


_UNSET = _Unset()

__all__ = [
    "load_cif",
    "load_res",
    "load_structure",
    "RinseParams",
    "FormFactorType",
    "StructureFactorType",
    "ReflectionList",
    "compute_structure_factors",
    "compute_power_spectrum",
    "normalise_power_spectrum",
    "descriptor",
    "descriptor_many",
    "descriptor_from_hkl",
    "load_hkl",
    "load_crystal_symmetry",
    "reflections_from_measured",
    "MeasuredCompletenessError",
    "descriptor_hash",
    "hash_to_bits",
    "DEFAULT_HASH_WORDS",
]


def descriptor(
    atoms: object,
    *,
    params: RinseParams | None = None,
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron"] = "xray",
    set_fixed_uiso: float | None | _Unset = _UNSET,
    debug: bool = False,
) -> NDArray[np.float64]:
    """Compute the RINSE descriptor for a single structure.

    Parameters
    ----------
    atoms:
        A :class:`cctbx.xray.structure` or a path to a supported structure file
        (``.cif``, ``.res``, ``.ins``).
    params:
        Descriptor hyper-parameters.  Uses :class:`RinseParams` defaults if *None*.
        ``params.log1p`` and ``params.l2`` control post-processing normalisation.
        ``params.flatten`` controls whether the output is a 1-D vector (default
        *True*) or the 2-D ``(n_max, n_l_levels)`` matrix.
    form_factor_type:
        ``"xray"`` | ``"electron"`` | ``"neutron"``.
    set_fixed_uiso:
        If *None*, use the displacement parameters from the CIF/RES file. If a
        float, discard the reported ADPs and set all atoms to isotropic
        U_iso = ``set_fixed_uiso`` Å². Uses ``params.set_fixed_uiso`` when not
        supplied.

    Notes
    -----
    The descriptor is always weighted by intensities ``I = |F|²``.  The
    resolution-dependent intensity envelope is removed at the power-spectrum
    level via ``params.monopole_normalisation``.

    Returns
    -------
    ndarray of shape ``(n_max * n_l_levels,)`` [default, flatten=True] or
    ``(n_max, n_l_levels)`` [flatten=False].
    """
    import pathlib

    t0 = time.perf_counter()

    if debug:
        if isinstance(atoms, (str, pathlib.Path)):
            print(f"[rinse_descriptor] input: file {atoms}", file=sys.stderr)
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
    if isinstance(set_fixed_uiso, _Unset):
        set_fixed_uiso = params.set_fixed_uiso

    _t = time.perf_counter()
    reflections = compute_structure_factors(
        xrs,
        sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
        form_factor_type=form_factor_type,
        structure_factor_type="F2",
        set_fixed_uiso=set_fixed_uiso,
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
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron"] = "xray",
    set_fixed_uiso: float | None | _Unset = _UNSET,
) -> NDArray[np.float64]:
    """Compute the RINSE descriptor for a list of structures.

    Parameters
    ----------
    structures:
        Iterable of :class:`cctbx.xray.structure` objects or supported
        structure-file paths (``.cif``, ``.res``, ``.ins``).
    params:
        Shared descriptor hyper-parameters.
    form_factor_type:
        Passed to :func:`descriptor`.
    set_fixed_uiso:
        If *None*, use the displacement parameters from the CIF/RES file. If a
        float, discard the reported ADPs and set all atoms to isotropic
        U_iso = ``set_fixed_uiso`` Å². Uses ``params.set_fixed_uiso`` when not
        supplied.

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
            set_fixed_uiso=set_fixed_uiso,
        )
        for s in structures
    ]
    return np.stack(results, axis=0)


def descriptor_from_hkl(
    hkl: str | os.PathLike[str] | tuple[ArrayLike, ArrayLike],
    cell: str | os.PathLike[str] | Sequence[float],
    *,
    params: RinseParams | None = None,
    space_group: str | None = None,
    max_missing_fraction: float = 0.01,
    debug: bool = False,
) -> NDArray[np.float64]:
    """Compute the RINSE descriptor from a *measured* reflection list.

    This is the model-free alternative to :func:`descriptor`: the descriptor is
    built directly from experimental intensities rather than from calculated
    structure factors.

    Parameters
    ----------
    hkl:
        Path to a SHELX-style ``.hkl`` file (records ``h k l I σ``), or a
        ``(hkl, intensity)`` tuple of array-likes with shapes ``(M, 3)`` and
        ``(M,)`` (intensity is ``|F|²``).
    cell:
        Path to a ``.cif``/``.res``/``.ins`` file providing the unit cell and
        space group, or a sequence of six cell parameters ``(a, b, c, α, β, γ)``.
    params:
        Descriptor hyper-parameters.  Uses :class:`RinseParams` defaults if
        *None*.  ``params.qmax_factor`` sets how far beyond ``q_max`` the
        supplied sphere must extend (default 1.2×).
    space_group:
        Hermann–Mauguin symbol overriding the space group from *cell* (defaults
        to ``P 1`` when *cell* is a bare cell).
    max_missing_fraction:
        Maximum tolerated fraction of missing (non-absent) reflections before a
        :class:`MeasuredCompletenessError` is raised.  Default 0.01 (1 %).

    Returns
    -------
    ndarray of shape ``(n_max * n_l_levels,)`` [default] or
    ``(n_max, n_l_levels)`` when ``params.flatten=False``.
    """
    if params is None:
        params = RinseParams()

    if isinstance(hkl, (str, os.PathLike)):
        indices, intensities, _sigma = load_hkl(hkl)
    else:
        raw_indices, raw_intensities = hkl  # (hkl, intensity) array-likes
        indices = np.asarray(raw_indices)
        intensities = np.asarray(raw_intensities)

    symmetry = load_crystal_symmetry(cell, space_group=space_group)

    reflections = reflections_from_measured(
        indices,
        intensities,
        symmetry,
        params=params,
        max_missing_fraction=max_missing_fraction,
        debug=debug,
    )

    P = compute_power_spectrum(reflections, params=params, debug=debug)
    return power_spectrum_to_vector(P) if params.flatten else P


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _to_xrs(atoms: object) -> Any:
    """Accept a cctbx xray.structure, str path, or pathlib.Path."""
    import pathlib

    if isinstance(atoms, (str, pathlib.Path)):
        return load_structure(atoms)
    # Accept any cctbx xray.structure (duck-typing; avoids a hard import of the C type)
    if hasattr(atoms, "structure_factors") and hasattr(atoms, "scatterers"):
        return atoms
    raise TypeError(
        f"Expected a cctbx xray.structure or a supported structure-file path, got {type(atoms)}"
    )
