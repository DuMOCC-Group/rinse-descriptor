"""Compute RINSE descriptor hashes for Materials Project structures.

This script queries the Materials Project summary endpoint, retrieves structures,
computes RINSE descriptors and hashes, and writes results incrementally.

Supports chunking for parallel processing and resumption from an existing
descriptor pickle.

Outputs:
    - mp_hashes_chunk_N.csv: CSV file with material_id and hash columns
    - mp_descriptors_chunk_N.pkl: Pickle file with material_ids and descriptors
    - stdout: Tab-separated material_id and hash for each processed structure

Usage:
    uv run tools/compute_mp_hashes.py
    uv run tools/compute_mp_hashes.py 10 0
    uv run tools/compute_mp_hashes.py --material-id mp-149
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

from libtbx.utils import Sorry
from rinse_descriptor import (
    RinseParams,
    compute_power_spectrum,
    compute_structure_factors,
    descriptor_hash,
    load_cif,
    power_spectrum_to_vector,
)

_RECOVERABLE_INPUT_ERRORS = (
    Sorry,
    ValueError,
    RuntimeError,
    OSError,
    NameError,
    AttributeError,
    KeyError,
)


def _print_timings(t_acc: dict[str, float], n: int) -> None:
    total = t_acc["total"]
    accounted = (
        t_acc["fetch"]
        + t_acc["to_cif"]
        + t_acc["load_cif"]
        + t_acc["struct_factors"]
        + t_acc["power_spectrum"]
        + t_acc["hash"]
    )
    other = max(total - accounted, 0.0)
    print(
        f"  avg timings over {n} structures (ms):  "
        f"fetch={t_acc['fetch'] / n * 1e3:.1f}  "
        f"to_cif={t_acc['to_cif'] / n * 1e3:.1f}  "
        f"load_cif={t_acc['load_cif'] / n * 1e3:.1f}  "
        f"struct_factors={t_acc['struct_factors'] / n * 1e3:.1f}  "
        f"power_spectrum={t_acc['power_spectrum'] / n * 1e3:.1f}  "
        f"hash={t_acc['hash'] / n * 1e3:.1f}  "
        f"other={other / n * 1e3:.1f}  "
        f"total={total / n * 1e3:.1f}",
        file=sys.stderr,
    )


def _doc_get(doc: Any, key: str) -> Any:
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)


def _dataset_columns(docs: Any, fields: list[str]) -> dict[str, list[Any]]:
    """Extract requested columns from MP API responses without row-wise iteration."""
    if hasattr(docs, "delta_table"):
        table = docs.delta_table
        if hasattr(table, "to_pyarrow_table"):
            return table.to_pyarrow_table(columns=fields).to_pydict()

    if hasattr(docs, "pyarrow_dataset"):
        table = docs.pyarrow_dataset.to_table(columns=fields)
        return table.to_pydict()

    out: dict[str, list[Any]] = {field: [] for field in fields}
    for doc in docs:
        for field in fields:
            out[field].append(_doc_get(doc, field))
    return out


def _resolve_api_key(cli_key: str | None) -> str:
    key = cli_key or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise ValueError(
            "Materials Project API key missing. Pass --api-key or set MP_API_KEY."
        )
    return key


def _fetch_all_material_ids(mpr: Any, chunk_size: int) -> list[str]:
    docs = mpr.materials.summary.search(
        all_fields=False,
        fields=["material_id"],
        chunk_size=chunk_size,
    )
    columns = _dataset_columns(docs, ["material_id"])
    material_ids: list[str] = []
    for mid in columns.get("material_id", []):
        if mid is not None:
            material_ids.append(str(mid))
    material_ids.sort()
    return material_ids


def _fetch_structure_docs(mpr: Any, material_ids: list[str]) -> list[tuple[str, Any]]:
    if not material_ids:
        return []
    docs = mpr.materials.summary.search(
        material_ids=material_ids,
        fields=["material_id", "structure"],
        chunk_size=max(len(material_ids), 1),
    )
    columns = _dataset_columns(docs, ["material_id", "structure"])
    out: list[tuple[str, Any]] = []
    mids = columns.get("material_id", [])
    structures = columns.get("structure", [])
    for mid, structure in zip(mids, structures, strict=False):
        if mid is not None:
            out.append((str(mid), structure))
    return out


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _process_single(material_id: str, api_key: str) -> None:
    from mp_api.client import MPRester  # pyright: ignore[reportMissingImports]
    from pymatgen.io.cif import CifWriter  # pyright: ignore[reportMissingImports]

    params = RinseParams()
    t_total_start = time.perf_counter()

    with MPRester(api_key) as mpr:
        docs = _fetch_structure_docs(mpr, [material_id])

    if not docs or docs[0][1] is None:
        print(f"Material id {material_id!r} not found or has no structure", file=sys.stderr)
        sys.exit(1)

    _, structure = docs[0]
    t0 = time.perf_counter()
    try:
        cif_text = str(CifWriter(structure))
        xrs = load_cif(StringIO(cif_text))
    except _RECOVERABLE_INPUT_ERRORS as exc:
        print(f"Error processing {material_id}: {exc}", file=sys.stderr)
        sys.exit(1)

    t_load = time.perf_counter()
    reflections = compute_structure_factors(
        xrs,
        sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
    )
    t_sf = time.perf_counter()
    P = compute_power_spectrum(reflections, params=params)
    desc = power_spectrum_to_vector(P)
    t_ps = time.perf_counter()
    hash_str = descriptor_hash(desc)
    t_hash = time.perf_counter()

    total = time.perf_counter() - t_total_start
    accounted = (t_load - t0) + (t_sf - t_load) + (t_ps - t_sf) + (t_hash - t_ps)
    other = max(total - accounted, 0.0)

    print(f"{material_id}\t{hash_str}")
    print(
        f"  to_cif+load={(t_load - t0) * 1e3:.1f}ms  "
        f"struct_factors={(t_sf - t_load) * 1e3:.1f}ms  "
        f"power_spectrum={(t_ps - t_sf) * 1e3:.1f}ms  "
        f"hash={(t_hash - t_ps) * 1e3:.1f}ms  "
        f"other={other * 1e3:.1f}ms  "
        f"total={total * 1e3:.1f}ms",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RINSE descriptor hashes for Materials Project structures"
    )
    parser.add_argument(
        "num_chunks",
        type=int,
        nargs="?",
        default=1,
        help="Total number of chunks to split the MP dataset into (default: 1)",
    )
    parser.add_argument(
        "chunk_id",
        type=int,
        nargs="?",
        default=0,
        help="Which chunk to process (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Materials Project API key (prefer env var MP_API_KEY).",
    )
    parser.add_argument(
        "--material-id",
        type=str,
        default=None,
        help="Process a single Materials Project id and print its hash, then exit.",
    )
    parser.add_argument(
        "--id-chunk-size",
        type=int,
        default=5000,
        help="Page size used while listing all material ids.",
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=200,
        help="Number of material ids fetched per structure-query batch.",
    )
    parser.add_argument(
        "--max-materials",
        type=int,
        default=None,
        help="Optional cap on selected material ids (useful for smoke tests).",
    )
    args = parser.parse_args()

    if args.num_chunks < 1:
        print("Error: num_chunks must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
        print(
            f"Error: chunk_id ({args.chunk_id}) must be in [0, {args.num_chunks - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.id_chunk_size < 1 or args.fetch_batch_size < 1:
        print("Error: id chunk sizes must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.max_materials is not None and args.max_materials < 1:
        print("Error: max-materials must be >= 1", file=sys.stderr)
        sys.exit(1)

    try:
        api_key = _resolve_api_key(args.api_key)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.material_id is not None:
        _process_single(args.material_id, api_key)
        return

    from mp_api.client import MPRester  # pyright: ignore[reportMissingImports]
    from pymatgen.io.cif import CifWriter  # pyright: ignore[reportMissingImports]

    chunk_suffix = f"_chunk_{args.chunk_id}" if args.num_chunks > 1 else ""
    pickle_file = Path(f"mp_descriptors{chunk_suffix}.pkl")
    csv_file = Path(f"mp_hashes{chunk_suffix}.csv")
    run_t0 = time.perf_counter()

    if pickle_file.exists():
        print("Loading existing descriptors from pickle file...", file=sys.stderr)
        with open(pickle_file, "rb") as f:
            existing_ids, existing_descriptors = pickle.load(f)
            material_ids_done = list(existing_ids)
            descriptors = list(existing_descriptors)
        processed_ids = set(material_ids_done)
        print(f"Loaded {len(material_ids_done)} existing descriptors", file=sys.stderr)
        resuming = True
    else:
        material_ids_done = []
        descriptors = []
        processed_ids = set()
        resuming = False

    params = RinseParams()
    t_acc = {
        "fetch": 0.0,
        "to_cif": 0.0,
        "load_cif": 0.0,
        "struct_factors": 0.0,
        "power_spectrum": 0.0,
        "hash": 0.0,
        "total": 0.0,
    }
    attempted = 0
    processed_new = 0
    errors = 0

    with MPRester(api_key) as mpr, open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["material_id", "hash"])

        if resuming:
            print("Writing existing hashes to CSV...", file=sys.stderr)
            for material_id, desc in zip(material_ids_done, descriptors, strict=True):
                writer.writerow([material_id, descriptor_hash(desc)])
            csvfile.flush()

        print("Fetching Materials Project id list...", file=sys.stderr)
        all_material_ids = _fetch_all_material_ids(mpr, args.id_chunk_size)
        print(f"Fetched {len(all_material_ids)} material ids", file=sys.stderr)

        selected_ids = [
            mid
            for idx, mid in enumerate(all_material_ids)
            if idx % args.num_chunks == args.chunk_id
        ]
        if args.max_materials is not None:
            selected_ids = selected_ids[: args.max_materials]
        print(
            f"Processing {len(selected_ids)} ids in chunk {args.chunk_id}/{args.num_chunks - 1}",
            file=sys.stderr,
        )

        for batch_ids in _chunks(selected_ids, args.fetch_batch_size):
            t_fetch = time.perf_counter()
            docs = _fetch_structure_docs(mpr, batch_ids)
            t_acc["fetch"] += time.perf_counter() - t_fetch
            if len(docs) < len(batch_ids):
                errors += len(batch_ids) - len(docs)

            for mid, structure in docs:
                attempted += 1
                if attempted % 100 == 0:
                    print(
                        f"Checked {attempted} selected structures, "
                        f"added {processed_new} new ({errors} errors)...",
                        file=sys.stderr,
                    )
                    _print_timings(t_acc, attempted)

                if mid in processed_ids:
                    continue

                if structure is None:
                    errors += 1
                    continue

                t_entry_start = time.perf_counter()
                try:
                    t0 = time.perf_counter()
                    cif_text = str(CifWriter(structure))
                    t_acc["to_cif"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    xrs = load_cif(StringIO(cif_text))
                    t_acc["load_cif"] += time.perf_counter() - t0

                    if xrs.scatterers().size() == 0:
                        continue

                    t0 = time.perf_counter()
                    reflections = compute_structure_factors(
                        xrs,
                        sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
                    )
                    t_acc["struct_factors"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    P = compute_power_spectrum(reflections, params=params)
                    desc = power_spectrum_to_vector(P)
                    t_acc["power_spectrum"] += time.perf_counter() - t0

                    material_ids_done.append(mid)
                    descriptors.append(desc)
                    processed_ids.add(mid)
                    processed_new += 1

                    t0 = time.perf_counter()
                    hash_str = descriptor_hash(desc)
                    t_acc["hash"] += time.perf_counter() - t0

                    writer.writerow([mid, hash_str])
                    print(f"{mid}\t{hash_str}")

                    if processed_new % 100 == 0:
                        csvfile.flush()
                        with open(pickle_file, "wb") as f:
                            pickle.dump((material_ids_done, descriptors), f)

                except _RECOVERABLE_INPUT_ERRORS as exc:
                    errors += 1
                    print(f"Error processing {mid}: {exc}", file=sys.stderr)
                finally:
                    t_acc["total"] += time.perf_counter() - t_entry_start

    with open(pickle_file, "wb") as f:
        pickle.dump((material_ids_done, descriptors), f)

    if attempted > 0:
        print("\nAverage timings per selected structure:", file=sys.stderr)
        _print_timings(t_acc, attempted)

    print(
        f"\nComplete! Checked {attempted} selected structures, "
        f"added {processed_new} new ({errors} errors).",
        file=sys.stderr,
    )
    print(f"Results saved to {csv_file}", file=sys.stderr)
    print(
        f"Descriptors saved to {pickle_file} ({len(material_ids_done)} total structures)",
        file=sys.stderr,
    )

    run_total = time.perf_counter() - run_t0
    if attempted > 0:
        print(
            f"Run wall-clock time: {run_total:.1f}s "
            f"({run_total / attempted * 1e3:.1f}ms per attempted structure)",
            file=sys.stderr,
        )
    else:
        print(f"Run wall-clock time: {run_total:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
