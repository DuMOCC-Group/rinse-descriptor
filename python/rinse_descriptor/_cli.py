"""Command-line interface for the RINSE descriptor.

Usage examples
--------------
    rinse nacl.cif
    rinse nacl.res
    rinse nacl.cif --n-max 12 --l-max 48 --l-min 0
    rinse nacl.cif --log1p --no-l2
    rinse nacl.cif --hash
    rinse nacl.cif --hash --hash-words 8
    rinse nacl.cif --output-format json
    rinse file1.cif file2.res --hash
    rinse file1.cif file2.res --no-flatten --output-format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np


def _make_parser() -> argparse.ArgumentParser:
    from . import DEFAULT_HASH_WORDS
    from ._descriptor import RinseParams

    defaults = RinseParams()

    p = argparse.ArgumentParser(
        prog="rinse",
        description="Compute the RINSE descriptor for one or more structure files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional
    p.add_argument(
        "input_files",
        nargs="+",
        metavar="INPUT",
        help="Path(s) to input file(s): a structure (.cif, .res, .ins) or a "
        "measured reflection list (.hkl). .hkl inputs require --cell.",
    )

    # --- Measured-data (model-free) path ---
    meas = p.add_argument_group("measured reflections (.hkl)")
    meas.add_argument(
        "--cell",
        metavar="FILE_OR_PARAMS",
        default=None,
        help="Unit cell + space group for .hkl inputs: a path to a "
        ".cif/.res/.ins file, or six comma-separated cell parameters "
        "'a,b,c,alpha,beta,gamma'.",
    )
    meas.add_argument(
        "--space-group",
        metavar="HM",
        default=None,
        dest="space_group",
        help="Hermann-Mauguin space-group symbol overriding the one in --cell.",
    )
    meas.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.01,
        metavar="F",
        dest="max_missing_fraction",
        help="Maximum tolerated fraction of missing reflections for .hkl inputs.",
    )

    # --- RinseParams ---
    p.add_argument(
        "--n-max",
        type=int,
        default=defaults.n_max,
        metavar="N",
        help="Number of radial basis functions.",
    )
    p.add_argument(
        "--l-max",
        type=int,
        default=defaults.l_max,
        metavar="L",
        help="Maximum ℓ value (exclusive).",
    )
    p.add_argument(
        "--l-min",
        type=int,
        default=defaults.l_min,
        metavar="L",
        help="Minimum ℓ value (inclusive).",
    )
    p.add_argument(
        "--radial-basis",
        default=defaults.radial_basis,
        choices=["cv_gaussian", "lin_gaussian"],
        help="Radial basis type.",
    )
    p.add_argument(
        "--radial-scale",
        type=float,
        default=defaults.radial_scale,
        metavar="S",
        dest="radial_scale",
        help=(
            "Per-shell scale factor in Å⁻¹.  Sets the distance between radial "
            "shells; the resolution cutoff is derived from this and --n-max."
        ),
    )
    p.add_argument(
        "--include-odd-l",
        action="store_true",
        default=defaults.include_odd_l,
        help="Include odd-ℓ spherical harmonics.",
    )

    # Normalisation
    norm = p.add_argument_group("normalisation")
    norm.add_argument(
        "--log1p",
        dest="log1p",
        action="store_true",
        help="Enable log1p compression.",
    )
    norm.add_argument(
        "--no-l2",
        dest="l2",
        action="store_false",
        help="Disable L2 normalisation.",
    )
    norm.add_argument(
        "--no-monopole-normalisation",
        dest="monopole_normalisation",
        action="store_false",
        help=(
            "Disable monopole (ℓ=0) normalisation, which by default divides each "
            "radial level's angular power by its ℓ=0 power to remove the "
            "resolution-dependent intensity envelope."
        ),
    )
    p.set_defaults(
        log1p=defaults.log1p, l2=defaults.l2, monopole_normalisation=defaults.monopole_normalisation
    )

    # Structure factors
    sf = p.add_argument_group("structure factors")
    sf.add_argument(
        "--form-factor",
        default="xray",
        choices=["xray", "electron", "neutron"],
        dest="form_factor_type",
        help="Atomic form factor type.",
    )

    # Hash
    h = p.add_argument_group("locality-sensitive hash")
    h.add_argument(
        "--hash",
        action="store_true",
        help="Print a proquint hash of the descriptor.",
    )
    h.add_argument(
        "--hash-words",
        type=int,
        default=DEFAULT_HASH_WORDS,
        metavar="W",
        help="Number of 16-bit proquint words in the hash.",
    )

    # Output
    p.add_argument(
        "--no-flatten",
        dest="flatten",
        action="store_false",
        help="Output a 2-D matrix instead of a flat 1-D vector.",
    )
    p.set_defaults(flatten=True)
    p.add_argument(
        "--output-format",
        choices=["text", "json", "npy"],
        default="text",
        dest="output_format",
        help=(
            "Output format. 'text' prints whitespace-separated values, "
            "'json' emits a JSON object, 'npy' writes .npy files alongside each input."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (rinse-descriptor {_get_version()})",
    )

    return p


def _get_version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:
        return "unknown"


def _parse_cell(cell: str) -> str | list[float]:
    """Interpret the --cell value as six cell params or a file path."""
    parts = [tok for tok in cell.replace(",", " ").split() if tok]
    if len(parts) == 6:
        try:
            return [float(tok) for tok in parts]
        except ValueError:
            pass
    return cell


def _compute_one(
    structure_path: str,
    params: object,
    form_factor_type: Literal["xray", "electron", "neutron"],
    want_hash: bool,
    hash_words: int,
    *,
    cell: str | None = None,
    space_group: str | None = None,
    max_missing_fraction: float = 0.01,
) -> tuple[object, str | None]:
    """Return (array, hash_str|None).  Raises on error."""
    from . import RinseParams, descriptor, descriptor_from_hkl, descriptor_hash

    assert isinstance(params, RinseParams)
    if Path(structure_path).suffix.lower() == ".hkl":
        if cell is None:
            raise ValueError("a .hkl input requires --cell (a .cif/.res/.ins file or cell params)")
        vec = descriptor_from_hkl(
            structure_path,
            _parse_cell(cell),
            params=params,
            space_group=space_group,
            max_missing_fraction=max_missing_fraction,
        )
    else:
        vec = descriptor(
            structure_path,
            params=params,
            form_factor_type=form_factor_type,
        )
    h = descriptor_hash(vec.ravel(), n_words=hash_words) if want_hash else None
    return vec, h


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    from . import RinseParams

    try:
        params = RinseParams(
            n_max=args.n_max,
            l_max=args.l_max,
            l_min=args.l_min,
            include_odd_l=args.include_odd_l,
            radial_scale=args.radial_scale,
            radial_basis=args.radial_basis,
            monopole_normalisation=args.monopole_normalisation,
            log1p=args.log1p,
            l2=args.l2,
            flatten=args.flatten,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 1  # unreachable, but satisfies type checker

    results: list[dict[str, object]] = []
    errors: list[str] = []

    for input_file in args.input_files:
        path = Path(input_file)
        if not path.exists():
            errors.append(f"{input_file}: file not found")
            continue
        try:
            vec, h = _compute_one(
                str(path),
                params,
                form_factor_type=cast(
                    Literal["xray", "electron", "neutron"], args.form_factor_type
                ),
                want_hash=args.hash,
                hash_words=args.hash_words,
                cell=args.cell,
                space_group=args.space_group,
                max_missing_fraction=args.max_missing_fraction,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{input_file}: {exc}")
            continue

        results.append({"file": input_file, "vector": vec, "hash": h})

    if args.output_format == "json":
        out: list[dict[str, object]] = []
        for r in results:
            entry: dict[str, object] = {
                "file": r["file"],
                "shape": list(np.asarray(r["vector"]).shape),
                "vector": np.asarray(r["vector"]).ravel().tolist(),
            }
            if r["hash"] is not None:
                entry["hash"] = r["hash"]
            out.append(entry)
        print(json.dumps(out, indent=2))

    elif args.output_format == "npy":
        for r in results:
            out_path = Path(str(r["file"])).with_suffix(".npy")
            np.save(out_path, np.asarray(r["vector"]))
            msg = f"Saved {out_path}"
            if r["hash"] is not None:
                msg += f"  hash={r['hash']}"
            print(msg)

    else:  # text
        for r in results:
            header = f"# {r['file']}  shape={np.asarray(r['vector']).shape}"
            if r["hash"] is not None:
                header += f"  hash={r['hash']}"
            print(header)
            arr = np.asarray(r["vector"])
            if arr.ndim == 1:
                print(" ".join(f"{v:.6g}" for v in arr))
            else:
                for row in arr:
                    print(" ".join(f"{v:.6g}" for v in row))

    for msg in errors:
        print(f"ERROR: {msg}", file=sys.stderr)

    return 1 if errors and not results else (2 if errors else 0)


if __name__ == "__main__":
    sys.exit(main())
