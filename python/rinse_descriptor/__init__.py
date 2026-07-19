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
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

from ._cctbx_import_patch import patch_cctbx_imports
from ._crystal import load_cif

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._descriptor import RinseParams
    from ._structure_factors import FormFactorType, StructureFactorType

patch_cctbx_imports()

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

_LAZY_EXPORTS = {
    "DEFAULT_HASH_WORDS": ("._hash", "DEFAULT_HASH_WORDS"),
    "FormFactorType": ("._structure_factors", "FormFactorType"),
    "ReflectionList": ("._structure_factors", "ReflectionList"),
    "RinseParams": ("._descriptor", "RinseParams"),
    "StructureFactorType": ("._structure_factors", "StructureFactorType"),
    "compute_power_spectrum": ("._descriptor", "compute_power_spectrum"),
    "compute_structure_factors": ("._structure_factors", "compute_structure_factors"),
    "descriptor_hash": ("._hash", "descriptor_hash"),
    "hash_to_bits": ("._hash", "hash_to_bits"),
    "normalise_power_spectrum": ("._descriptor", "normalise_power_spectrum"),
    "power_spectrum_to_vector": ("._descriptor", "power_spectrum_to_vector"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def descriptor(
    atoms: object,
    *,
    params: RinseParams | None = None,
    form_factor_type: FormFactorType | Literal["xray", "electron", "neutron"] = "xray",
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
    use_reported_adps: bool = False,
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
    use_reported_adps:
        If *True*, use displacement parameters from the CIF.  Default *False*:
        all atoms are reset to isotropic U_iso = 0.01 Å².

    Returns
    -------
    ndarray of shape ``(n_max * n_l_levels,)`` [default, flatten=True] or
    ``(n_max, n_l_levels)`` [flatten=False].
    """
    import pathlib

    from ._descriptor import RinseParams, compute_power_spectrum, power_spectrum_to_vector
    from ._structure_factors import compute_structure_factors

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
        use_reported_adps=use_reported_adps,
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
    structure_factor_type: StructureFactorType | Literal["F", "F2"] = "F2",
    use_reported_adps: bool = False,
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
    use_reported_adps:
        If *True*, use displacement parameters from the CIF.  Default *False*:
        all atoms are reset to isotropic U_iso = 0.01 Å².

    Returns
    -------
    ndarray of shape (N, descriptor_length) [default, flatten=True] or
    (N, n_max, n_l_levels) [flatten=False].
    """
    import numpy as np

    results = [
        descriptor(
            s,
            params=params,
            form_factor_type=form_factor_type,
            structure_factor_type=structure_factor_type,
            use_reported_adps=use_reported_adps,
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
    raise TypeError(f"Expected a cctbx xray.structure or a CIF file path, got {type(atoms)}")
