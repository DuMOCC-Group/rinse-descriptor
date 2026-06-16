"""Pure-Python fallbacks for CIF parsing and structure factor calculation.

Used when gemmi is unavailable (e.g., WebAssembly / Emscripten platforms).

Form factors are approximated by single Gaussians normalised to atomic number::

    f_j(s) = Z_j · exp(-(B_GAUSS + 8π²·u_iso_j) · s²),   s = |G|/2

Anisotropic displacement parameters are ignored.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._crystal import Crystal

# ---------------------------------------------------------------------------
# Form-factor parameter
# ---------------------------------------------------------------------------

#: Single-Gaussian width shared by all elements (Å²).
_B_GAUSS: float = 2.0

# ---------------------------------------------------------------------------
# Element table
# ---------------------------------------------------------------------------

_SYMBOL_TO_Z: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
}


def _label_to_Z(label: str) -> int:
    """Guess atomic number from a CIF atom-label or type-symbol string."""
    # Strip trailing oxidation state suffixes like "Fe3+", "O2-", "Na+"
    label = re.sub(r"\d*[+-]$", "", label.strip())
    # Try 2-char then 1-char element symbol at the start of the label
    for length in (2, 1):
        sym = label[:length]
        if not sym:
            continue
        sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
        if sym in _SYMBOL_TO_Z:
            return _SYMBOL_TO_Z[sym]
    raise ValueError(f"Cannot determine element from CIF label {label!r}")


# ---------------------------------------------------------------------------
# CIF tokenizer
# ---------------------------------------------------------------------------


def _tokenize_cif(text: str) -> list[str]:
    """Tokenise CIF text into a flat list of string tokens."""
    tokens: list[str] = []
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]

        # Semicolon text block: ';' must be the very first character of the line
        if line.startswith(";"):
            idx += 1
            block_lines: list[str] = []
            while idx < len(lines) and not lines[idx].startswith(";"):
                block_lines.append(lines[idx])
                idx += 1
            tokens.append("\n".join(block_lines))
            idx += 1  # skip closing ';'
            continue

        # Skip blank lines and whole-line comments
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue

        # Tokenise character-by-character
        pos = 0
        while pos < len(line):
            # Skip whitespace
            while pos < len(line) and line[pos] in " \t":
                pos += 1
            if pos >= len(line):
                break
            c = line[pos]
            if c == "#":
                break  # inline comment – discard rest of line
            if c in "\"'":
                # Quoted string
                q = c
                pos += 1
                start = pos
                while pos < len(line) and line[pos] != q:
                    pos += 1
                tokens.append(line[start:pos])
                pos += 1  # skip closing quote
            else:
                # Bare token (ends at whitespace or '#')
                start = pos
                while pos < len(line) and line[pos] not in " \t#":
                    pos += 1
                tokens.append(line[start:pos])

        idx += 1
    return tokens


# ---------------------------------------------------------------------------
# CIF parser
# ---------------------------------------------------------------------------


def _parse_cif(text: str) -> dict[str, object]:
    """Parse the first data block of a CIF file.

    Returns a flat dict: lowercased key → ``str`` (singleton key-value pair)
    or ``list[str]`` (loop column).
    """
    tokens = _tokenize_cif(text)
    data: dict[str, object] = {}
    in_block = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower()

        if low.startswith("data_"):
            if in_block:
                break  # stop at second data block
            in_block = True
            i += 1
            continue

        if not in_block:
            i += 1
            continue

        if low == "loop_":
            i += 1
            # Collect column headers
            headers: list[str] = []
            while i < len(tokens) and tokens[i].startswith("_"):
                headers.append(tokens[i].lower())
                i += 1
            if not headers:
                continue
            # Collect data values (stop at next key / loop_ / data_)
            flat: list[str] = []
            while i < len(tokens):
                t = tokens[i]
                tl = t.lower()
                if t.startswith("_") or tl == "loop_" or tl.startswith("data_"):
                    break
                flat.append(t)
                i += 1
            n = len(headers)
            n_rows = len(flat) // n
            for j, h in enumerate(headers):
                data[h] = [flat[row * n + j] for row in range(n_rows)]
            continue

        if tok.startswith("_"):
            key = low
            i += 1
            if i < len(tokens):
                t = tokens[i]
                tl = t.lower()
                if not t.startswith("_") and tl != "loop_" and not tl.startswith("data_"):
                    data[key] = t
                    i += 1
            continue

        i += 1  # skip stray tokens outside any key context

    return data


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _to_float(s: object, default: float = 0.0) -> float:
    """Parse a CIF numeric string, stripping parenthesised uncertainty."""
    if not isinstance(s, str):
        return default
    s = s.strip()
    if s in (".", "?", ""):
        return default
    s = re.sub(r"\([^)]*\)", "", s)
    try:
        return float(s)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Unit-cell matrix
# ---------------------------------------------------------------------------


def _cell_matrix(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Build a (3, 3) cell matrix (rows = lattice vectors **a**, **b**, **c**).

    Convention: **a** along *x*, **b** in the *xy* plane.
    """
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array(
        [[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Symmetry-operation parser
# ---------------------------------------------------------------------------


def _parse_symop_part(s: str) -> tuple[float, float, float, float]:
    """Parse one component of a symmetry-operation expression.

    Returns ``(cx, cy, cz, translation)`` — the row of the rotation matrix
    and the corresponding translation element.

    Examples::

        "x"       → (1, 0, 0, 0)
        "-y+1/2"  → (0, -1, 0, 0.5)
        "1/2-x"   → (-1, 0, 0, 0.5)
        "z-1/4"   → (0, 0, 1, -0.25)
    """
    cx = cy = cz = t = 0.0
    s = s.strip().replace(" ", "").lower()
    pos = 0
    n = len(s)
    while pos < n:
        # Optional sign
        sign = 1.0
        if s[pos] in "+-":
            sign = -1.0 if s[pos] == "-" else 1.0
            pos += 1
            if pos >= n:
                break
        # Optional numeric coefficient / fractional translation
        m = re.match(r"(\d+)(?:/(\d+))?", s[pos:])
        coeff: float | None = None
        if m:
            coeff = float(m.group(1))
            if m.group(2):
                coeff /= float(m.group(2))
            pos += len(m.group(0))
        # Optional variable x/y/z
        if pos < n and s[pos] in "xyz":
            if coeff is None:
                coeff = 1.0
            var = s[pos]
            pos += 1
            if var == "x":
                cx += sign * coeff
            elif var == "y":
                cy += sign * coeff
            else:
                cz += sign * coeff
        else:
            # Pure numeric → translation contribution
            if coeff is not None:
                t += sign * coeff

    return cx, cy, cz, t


def _parse_symop(
    op_str: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Parse a CIF symmetry-operation string into ``(rot, trans)``.

    ``rot`` is a (3, 3) float64 rotation matrix; ``trans`` is a (3,) float64
    translation vector (fractional coordinates).
    """
    op_str = op_str.strip().strip("'\"")
    parts = op_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated parts, got: {op_str!r}")
    rot = np.zeros((3, 3), dtype=np.float64)
    trans = np.zeros(3, dtype=np.float64)
    for i, part in enumerate(parts):
        cx, cy, cz, t_val = _parse_symop_part(part)
        rot[i] = [cx, cy, cz]
        trans[i] = t_val
    return rot, trans


# ---------------------------------------------------------------------------
# CIF → Crystal (pure-Python)
# ---------------------------------------------------------------------------


def crystal_from_cif_pure(path: str) -> Crystal:
    """Load a :class:`~rinse_descriptor.Crystal` from a CIF file using only
    the Python standard library and NumPy (no gemmi required).

    Symmetry operations are read directly from the CIF block
    (``_symmetry_equiv_pos_as_xyz`` or ``_space_group_symop_operation_xyz``).
    Falls back to P 1 (identity only) when neither key is present.
    """
    from ._crystal import Crystal

    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    d = _parse_cif(text)

    # ── Cell parameters ───────────────────────────────────────────────────────
    a = _to_float(d.get("_cell_length_a"))
    b = _to_float(d.get("_cell_length_b"))
    c = _to_float(d.get("_cell_length_c"))
    al = math.radians(_to_float(d.get("_cell_angle_alpha"), 90.0))
    be = math.radians(_to_float(d.get("_cell_angle_beta"), 90.0))
    ga = math.radians(_to_float(d.get("_cell_angle_gamma"), 90.0))
    cell = _cell_matrix(a, b, c, al, be, ga)

    # ── Symmetry operations ───────────────────────────────────────────────────
    ops: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for key in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
        val = d.get(key)
        candidates: list[str] = (
            val if isinstance(val, list) else ([val] if isinstance(val, str) else [])
        )
        for s in candidates:
            try:
                ops.append(_parse_symop(s))
            except (ValueError, ZeroDivisionError):
                pass
        if ops:
            break  # use first key that yielded operations

    if not ops:
        # Default: space group P 1 (identity only)
        ops = [(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))]

    # ── Atom sites ────────────────────────────────────────────────────────────
    # Prefer _atom_site_type_symbol (plain element symbol) over _atom_site_label
    type_col: list[str] = []
    for key in ("_atom_site_type_symbol", "_atom_site_label"):
        v = d.get(key)
        if isinstance(v, list) and v:
            type_col = v
            break

    label_col: list[str] = []
    lv = d.get("_atom_site_label")
    if isinstance(lv, list):
        label_col = lv

    frac_x_raw = d.get("_atom_site_fract_x", [])
    frac_y_raw = d.get("_atom_site_fract_y", [])
    frac_z_raw = d.get("_atom_site_fract_z", [])
    if not isinstance(frac_x_raw, list) or not frac_x_raw:
        raise ValueError(f"No fractional coordinates found in CIF {path!r}")
    frac_x = cast(list[str], frac_x_raw)
    frac_y = cast(list[str], frac_y_raw)
    frac_z = cast(list[str], frac_z_raw)

    occ_col: list[str] | None = None
    ov = d.get("_atom_site_occupancy")
    if isinstance(ov, list):
        occ_col = ov

    u_iso_col: list[str] | None = None
    for key in ("_atom_site_u_iso_or_equiv", "_atom_site_uiso_or_equiv"):
        uv = d.get(key)
        if isinstance(uv, list):
            u_iso_col = uv
            break

    b_iso_col: list[str] | None = None
    for key in ("_atom_site_b_iso_or_equiv", "_atom_site_biso_or_equiv"):
        bv = d.get(key)
        if isinstance(bv, list):
            b_iso_col = bv
            break

    _8pi2 = 8.0 * math.pi**2
    TOL = 1e-4
    n_asym = len(frac_x)

    exp_frac: list[NDArray[np.float64]] = []
    exp_Z: list[int] = []
    exp_occ: list[float] = []
    exp_u: list[float] = []

    for idx in range(n_asym):
        # Determine atomic number
        Z = 0
        for candidate in (
            type_col[idx] if type_col else "",
            label_col[idx] if label_col else "",
        ):
            if candidate:
                try:
                    Z = _label_to_Z(candidate)
                    break
                except ValueError:
                    pass
        if Z == 0:
            Z = 6  # last-resort fallback: carbon

        xyz0 = np.array(
            [
                _to_float(frac_x[idx]),
                _to_float(frac_y[idx]),
                _to_float(frac_z[idx]),
            ]
        )
        occ = _to_float(occ_col[idx], 1.0) if occ_col is not None else 1.0
        if u_iso_col is not None:
            u_iso = _to_float(u_iso_col[idx])
        elif b_iso_col is not None:
            u_iso = _to_float(b_iso_col[idx]) / _8pi2
        else:
            u_iso = 0.0

        seen: list[list[float]] = []
        for rot, trans in ops:
            xyz = (rot @ xyz0 + trans) % 1.0
            xyzl: list[float] = xyz.tolist()
            if any(
                abs(xyzl[0] - s[0]) < TOL
                and abs(xyzl[1] - s[1]) < TOL
                and abs(xyzl[2] - s[2]) < TOL
                for s in seen
            ):
                continue
            seen.append(xyzl)
            exp_frac.append(xyz)
            exp_Z.append(Z)
            exp_occ.append(occ)
            exp_u.append(u_iso)

    if not exp_frac:
        raise ValueError(f"CIF {path!r} yielded no atoms after symmetry expansion")

    frac_arr = np.array(exp_frac, dtype=np.float64)
    positions = frac_arr @ cell  # fractional → Cartesian (Å)

    return Crystal(
        cell=cell,
        positions=positions,
        species=np.array(exp_Z, dtype=np.int32),
        occupancies=np.array(exp_occ, dtype=np.float64),
        u_iso=np.array(exp_u, dtype=np.float64),
        pbc=np.array([True, True, True], dtype=np.bool_),
    )


# ---------------------------------------------------------------------------
# Single-Gaussian structure-factor calculation
# ---------------------------------------------------------------------------


def calc_sf_gauss(
    crystal: Crystal,
    hkl: NDArray[np.int32],
    recip: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""Compute F(hkl) using single-Gaussian form factors.

    .. math::

        f_j(s) = Z_j \cdot \exp\!\bigl(-(B_{\text{gauss}} + 8\pi^2 u_{\text{iso},j})\,s^2\bigr),
        \qquad s = |\mathbf{G}|/2

    where :math:`B_{\text{gauss}}` = :data:`_B_GAUSS` Å² is a fixed width
    shared by all elements.  Anisotropic displacement parameters are ignored.
    """
    inv_cell = np.linalg.inv(crystal.cell)
    frac_pos = crystal.positions @ inv_cell.T  # (N, 3) fractional
    Z_vals = crystal.species.astype(np.float64)  # (N,)
    occ = np.asarray(crystal.occupancies, dtype=np.float64)  # (N,)
    u_iso = np.asarray(crystal.u_iso, dtype=np.float64)  # (N,)
    B_tot = _B_GAUSS + 8.0 * math.pi**2 * u_iso  # (N,) Å²

    q_vecs = hkl.astype(np.float64) @ recip  # (M, 3) Å⁻¹
    s_sq = (np.linalg.norm(q_vecs, axis=1) / 2.0) ** 2  # (M,) Å⁻²

    # Form factors f_j(s) – shape (M, N)
    ff = Z_vals[np.newaxis, :] * np.exp(-B_tot[np.newaxis, :] * s_sq[:, np.newaxis])
    # Phases φ = 2π (h x_j + k y_j + l z_j) – shape (M, N)
    phases = 2.0 * math.pi * (hkl.astype(np.float64) @ frac_pos.T)

    F = (occ[np.newaxis, :] * ff * np.exp(1j * phases)).sum(axis=1)
    return cast(NDArray[np.complex128], F.astype(np.complex128))
