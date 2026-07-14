"""CIF loading via cctbx – returns a :class:`cctbx.xray.structure`.

The public entry point is :func:`load_cif`.  The ``iotbx.cif`` import is done
at module level to pre-load cctbx's C extensions before pytest activates its
I/O captures (avoids a libstdc++ segfault on Python 3.14).
"""

from __future__ import annotations

import os
from typing import IO, Any, cast

# Eager import – must happen before pytest capture is active.
from iotbx import cif as _iotbx_cif  # noqa: F401


def load_cif(path: str | os.PathLike[str] | IO[bytes] | IO[str]) -> Any:
    """Load a CIF file and return a :class:`cctbx.xray.structure`.

    Parameters
    ----------
    path:
        A file path (``str`` or :class:`pathlib.Path`), an open binary or
        text file-like object, or any object with a ``read()`` method.

    The asymmetric unit is stored with its original space group; it is
    expanded to P1 lazily during structure-factor calculation.
    """
    from iotbx import cif as iotbx_cif

    if hasattr(path, "read"):
        reader = cast(IO[bytes], path)
        raw = reader.read()
        cif_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        cif_reader = iotbx_cif.reader(input_string=cif_text)
    else:
        cif_reader = iotbx_cif.reader(file_path=str(path))

    model = cif_reader.build_crystal_structures()
    if not model:
        raise ValueError(f"CIF {path!r} did not yield any crystal structures")

    xrs = next(iter(model.values()))
    if xrs.scatterers().size() == 0:
        raise ValueError(f"CIF {path!r} does not contain any atomic sites")

    return xrs
