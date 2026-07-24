"""Crystal-structure loading via cctbx.

Public entry points:
- :func:`load_cif` for CIF files
- :func:`load_res` for SHELX INS/RES files
- :func:`load_structure` for extension-based dispatch

The iotbx/cctbx imports are delayed until function call time and guarded by
``patch_cctbx_imports`` to avoid known platform-specific import-time issues.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import IO, Any, cast

from ._cctbx_import_patch import patch_cctbx_imports

# Matches an _atom_site loop that has column headers but no data rows.
# Such CIFs cause iotbx.cif to fail silently or segfault during parsing.
_EMPTY_ATOM_SITE_LOOP_RE = re.compile(
    r"loop_"
    r"(?:\s+_atom_site_\S+)+"  # one or more _atom_site_* column names
    r"(?:\s|#[^\n]*)+"  # only whitespace/comments follow (no data)
    r"(?=_(?!atom_site_)|loop_|\Z)",  # next is a non-atom_site key, new loop, or EOF
    re.IGNORECASE,
)


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
    patch_cctbx_imports()

    from iotbx import cif as iotbx_cif

    cif_text = _read_text(path)

    if _EMPTY_ATOM_SITE_LOOP_RE.search(cif_text):
        raise ValueError(f"CIF {path!r} has an _atom_site loop with no data rows")

    cif_reader = iotbx_cif.reader(input_string=cif_text)
    model = cif_reader.build_crystal_structures()
    if not model:
        raise ValueError(f"CIF {path!r} did not yield any crystal structures")

    xrs = next(iter(model.values()))
    if xrs.scatterers().size() == 0:
        raise ValueError(f"CIF {path!r} does not contain any atomic sites")

    return xrs


def load_res(path: str | os.PathLike[str] | IO[bytes] | IO[str]) -> Any:
    """Load a SHELX INS/RES file and return a :class:`cctbx.xray.structure`.

    Parameters
    ----------
    path:
        A file path (``str`` or :class:`pathlib.Path`), an open binary or
        text file-like object, or any object with a ``read()`` method.
    """
    patch_cctbx_imports()

    from cctbx import xray

    try:
        if hasattr(path, "read"):
            text = _read_text(path)
            from io import StringIO

            xrs = xray.structure.from_shelx(file=StringIO(text), strictly_shelxl=False)
        else:
            xrs = xray.structure.from_shelx(filename=str(path), strictly_shelxl=False)
    except Exception as exc:
        raise ValueError(f"SHELX file {path!r} could not be parsed") from exc

    if xrs.scatterers().size() == 0:
        raise ValueError(f"SHELX file {path!r} does not contain any atomic sites")

    return xrs


def load_structure(path: str | os.PathLike[str] | IO[bytes] | IO[str]) -> Any:
    """Load a crystal structure from CIF or SHELX format.

    For filesystem paths, the parser is selected by extension:
    - ``.cif`` -> :func:`load_cif`
    - ``.res``/``.ins`` -> :func:`load_res`

    File-like objects default to CIF parsing because no extension is available.
    """
    if hasattr(path, "read"):
        return load_cif(path)

    suffix = Path(str(path)).suffix.lower()
    if suffix == ".cif":
        return load_cif(path)
    if suffix in {".res", ".ins"}:
        return load_res(path)

    raise ValueError(
        f"Unsupported structure file extension for {path!r}. "
        "Supported extensions are: .cif, .res, .ins"
    )


def _read_text(path: str | os.PathLike[str] | IO[bytes] | IO[str]) -> str:
    if hasattr(path, "read"):
        reader = cast(IO[bytes], path)
        raw = reader.read()
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    with open(str(path), encoding="utf-8", errors="replace") as f:
        return f.read()
