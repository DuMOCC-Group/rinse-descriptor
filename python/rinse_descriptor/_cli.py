"""Command-line interface for the RINSE descriptor.

Usage examples
--------------
    rinse nacl.cif
    rinse nacl.cif --n-max 12 --l-max 48 --l-min 0
    rinse nacl.cif --log1p --no-l2
    rinse nacl.cif --hash
    rinse nacl.cif --hash --hash-words 8
    rinse nacl.cif --output-format json
    rinse file1.cif file2.cif --hash
    rinse file1.cif file2.cif --no-flatten --output-format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _make_parser() -> argparse.ArgumentParser:
    from . import DEFAULT_HASH_WORDS
    from ._descriptor import RinseParams

    defaults = RinseParams()

    p = argparse.ArgumentParser(
        prog="rinse",
        description="Compute the RINSE descriptor for one or more CIF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional
    p.add_argument(
        "cif",
        nargs="+",
        metavar="CIF",
        help="Path(s) to input CIF file(s).",
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
        choices=["smooth_shells_nl", "smooth_shells_cw", "chebyshev", "bessel"],
        help="Radial basis type.",
    )
    p.add_argument(
        "--stol",
        type=float,
        default=defaults.sin_theta_over_lambda_max,
        metavar="STOL",
        dest="sin_theta_over_lambda_max",
        help="sin(θ)/λ resolution cutoff in Å⁻¹.",
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
    p.set_defaults(log1p=defaults.log1p, l2=defaults.l2)

    # Structure factors
    sf = p.add_argument_group("structure factors")
    sf.add_argument(
        "--form-factor",
        default="xray",
        choices=["xray", "electron", "neutron", "unity"],
        dest="form_factor_type",
        help="Atomic form factor type.",
    )
    sf.add_argument(
        "--sf-type",
        default="F2",
        choices=["F2", "F"],
        dest="structure_factor_type",
        help="Structure factor type (intensity F² or amplitude |F|).",
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
            "'json' emits a JSON object, 'npy' writes .npy files alongside each CIF."
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


def _compute_one(
    cif_path: str,
    params: object,
    form_factor_type: str,
    structure_factor_type: str,
    want_hash: bool,
    hash_words: int,
) -> tuple[object, str | None]:
    """Return (array, hash_str|None).  Raises on error."""
    from typing import Literal, cast

    from . import RinseParams, descriptor, descriptor_hash
    from ._structure_factors import FormFactorType, StructureFactorType

    assert isinstance(params, RinseParams)
    vec = descriptor(
        cif_path,
        params=params,
        form_factor_type=cast(
            "Literal['xray','electron','neutron','unity'] | FormFactorType",
            form_factor_type,
        ),
        structure_factor_type=cast(
            "Literal['F','F2'] | StructureFactorType",
            structure_factor_type,
        ),
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
            sin_theta_over_lambda_max=args.sin_theta_over_lambda_max,
            radial_basis=args.radial_basis,
            log1p=args.log1p,
            l2=args.l2,
            flatten=args.flatten,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 1  # unreachable, but satisfies type checker

    results: list[dict[str, object]] = []
    errors: list[str] = []

    for cif in args.cif:
        path = Path(cif)
        if not path.exists():
            errors.append(f"{cif}: file not found")
            continue
        try:
            vec, h = _compute_one(
                str(path),
                params,
                form_factor_type=args.form_factor_type,
                structure_factor_type=args.structure_factor_type,
                want_hash=args.hash,
                hash_words=args.hash_words,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cif}: {exc}")
            continue

        results.append({"file": cif, "vector": vec, "hash": h})

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
