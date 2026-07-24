"""Compute RINSE descriptor hashes for CIF structures stored in a COD ZIP archive.

This script processes CIFs directly from a .zip file without extracting them to disk.
It mirrors the descriptor pipeline used by ``compute_csd_hashes.py`` while using COD
file names (without extension) as identifiers.

Supports chunking for parallel processing on HPC clusters.

Outputs:
    - cod_hashes_chunk_N.csv: CSV file with cod_id and hash columns
    - cod_descriptors_chunk_N.pkl: Pickle file with cod_ids and high-dimensional descriptors
    - stdout: Tab-separated cod_id and hash for each processed structure

Usage:
    python compute_cod_hashes.py
    python compute_cod_hashes.py 10 0
    python compute_cod_hashes.py --zip-path C:/path/to/cod-cifs-mysql.zip
    python compute_cod_hashes.py --cod-id 1000001
"""

import argparse
import csv
import pickle
import sys
import time
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

from libtbx.utils import Sorry
from rinse_descriptor import (
    RinseParams,
    compute_power_spectrum,
    compute_structure_factors,
    descriptor_hash,
    load_cif,
    power_spectrum_to_vector,
)


def _print_timings(t_acc: dict, n: int) -> None:
    total = t_acc["total"]
    accounted = (
        t_acc["cif_string"]
        + t_acc["load_cif"]
        + t_acc["struct_factors"]
        + t_acc["power_spectrum"]
        + t_acc["hash"]
    )
    other = max(total - accounted, 0.0)
    print(
        f"  avg timings over {n} structures (ms):  "
        f"cif_string={t_acc['cif_string'] / n * 1e3:.1f}  "
        f"load_cif={t_acc['load_cif'] / n * 1e3:.1f}  "
        f"struct_factors={t_acc['struct_factors'] / n * 1e3:.1f}  "
        f"power_spectrum={t_acc['power_spectrum'] / n * 1e3:.1f}  "
        f"hash={t_acc['hash'] / n * 1e3:.1f}  "
        f"other={other / n * 1e3:.1f}  "
        f"total={total / n * 1e3:.1f}",
        file=sys.stderr,
    )


def _iter_cif_members(zip_file: ZipFile):
    for member in zip_file.infolist():
        if member.is_dir():
            continue
        if member.filename.lower().endswith(".cif"):
            yield member


def _cod_id_from_member_name(filename: str) -> str:
    return Path(filename).stem


def _decode_cif_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _process_single(cod_id: str, zip_path: Path) -> None:
    """Compute and print the descriptor hash for one COD CIF id in the ZIP file."""
    t_total_start = time.perf_counter()
    with ZipFile(zip_path) as zf:
        target_member = None
        for member in _iter_cif_members(zf):
            if _cod_id_from_member_name(member.filename) == cod_id:
                target_member = member
                break

        if target_member is None:
            print(f"COD id {cod_id!r} not found in {zip_path}", file=sys.stderr)
            sys.exit(1)

        with zf.open(target_member) as handle:
            cif_string = _decode_cif_bytes(handle.read())

    params = RinseParams()
    t0 = time.perf_counter()
    try:
        xrs = load_cif(StringIO(cif_string))
    except (Sorry, ValueError, RuntimeError, OSError, NameError) as exc:
        print(f"Error processing {cod_id}: {exc}", file=sys.stderr)
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
    print(f"{cod_id}\t{hash_str}")
    print(
        f"  load_cif={(t_load - t0) * 1e3:.1f}ms  "
        f"struct_factors={(t_sf - t_load) * 1e3:.1f}ms  "
        f"power_spectrum={(t_ps - t_sf) * 1e3:.1f}ms  "
        f"hash={(t_hash - t_ps) * 1e3:.1f}ms  "
        f"other={other * 1e3:.1f}ms  "
        f"total={total * 1e3:.1f}ms",
        file=sys.stderr,
    )


def main():
    """Process all COD structures and compute their RINSE descriptor hashes."""
    parser = argparse.ArgumentParser(
        description="Compute RINSE descriptor hashes for COD structures"
    )
    parser.add_argument(
        "num_chunks",
        type=int,
        nargs="?",
        default=1,
        help="Total number of chunks to split the COD into (default: 1)",
    )
    parser.add_argument(
        "chunk_id",
        type=int,
        nargs="?",
        default=0,
        help="Which chunk to process (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(r"C:\Users\Tom\Downloads\cod-cifs-mysql.zip"),
        help=(
            "Path to COD CIF zip archive "
            "(default: C:/Users/Tom/Downloads/cod-cifs-mysql.zip)"
        ),
    )
    parser.add_argument(
        "--cod-id",
        type=str,
        default=None,
        help="Process a single COD id (filename stem) and print its hash, then exit",
    )
    args = parser.parse_args()

    if args.cod_id is not None:
        _process_single(args.cod_id, args.zip_path)
        return

    if args.chunk_id >= args.num_chunks:
        print(
            f"Error: chunk_id ({args.chunk_id}) must be less than num_chunks ({args.num_chunks})",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.zip_path.exists():
        print(f"Error: ZIP archive not found: {args.zip_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening COD ZIP archive: {args.zip_path}", file=sys.stderr)

    # Determine output file names based on chunking
    if args.num_chunks > 1:
        chunk_suffix = f"_chunk_{args.chunk_id}"
        print(f"Processing chunk {args.chunk_id} of {args.num_chunks}...", file=sys.stderr)
    else:
        chunk_suffix = ""

    run_t0 = time.perf_counter()

    pickle_file = Path(f"cod_descriptors{chunk_suffix}.pkl")
    csv_file = Path(f"cod_hashes{chunk_suffix}.csv")

    # Storage for descriptors - load existing data if available
    resuming = False
    if pickle_file.exists():
        print("Loading existing descriptors from pickle file...", file=sys.stderr)
        with open(pickle_file, "rb") as f:
            existing_refcodes, existing_descriptors = pickle.load(f)
            refcodes = list(existing_refcodes)
            descriptors = list(existing_descriptors)
        processed_refcodes = set(refcodes)
        print(f"Loaded {len(refcodes)} existing descriptors", file=sys.stderr)
        resuming = True
    else:
        refcodes = []
        descriptors = []
        processed_refcodes = set()

    # Open CSV file for writing
    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cod_id", "hash"])

        # If resuming, write existing hashes to CSV first
        if resuming:
            print("Writing existing hashes to CSV...", file=sys.stderr)
            for refcode, desc in zip(refcodes, descriptors, strict=True):
                hash_str = descriptor_hash(desc)
                writer.writerow([refcode, hash_str])
            csvfile.flush()

        # Progress counter
        count = 0
        new_count = 0
        errors = 0
        skipped_chunk = 0

        # Per-step timing accumulators (seconds)
        params = RinseParams()
        t_acc = {
            "cif_string": 0.0,
            "load_cif": 0.0,
            "struct_factors": 0.0,
            "power_spectrum": 0.0,
            "hash": 0.0,
            "total": 0.0,
        }
        t_count = 0

        # Iterate through all CIF members in ZIP without extracting files
        print("Scanning ZIP members...", file=sys.stderr)
        with ZipFile(args.zip_path) as zf:
            cif_members = list(_iter_cif_members(zf))
            print(f"Found {len(cif_members)} CIF files in ZIP", file=sys.stderr)
            print("Processing structures...", file=sys.stderr)

            for entry_idx, member in enumerate(cif_members):
                refcode = _cod_id_from_member_name(member.filename)
                count += 1

                # Determine if this entry belongs to our chunk
                if args.num_chunks > 1:
                    if entry_idx % args.num_chunks != args.chunk_id:
                        skipped_chunk += 1
                        continue

                if count % 100 == 0:
                    if args.num_chunks > 1:
                        print(
                            f"Checked {count} structures (chunk: {count - skipped_chunk}), "
                            f"added {new_count} new ({errors} errors)...",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"Checked {count} structures, added {new_count} new "
                            f"({errors} errors)...",
                            file=sys.stderr,
                        )
                    if t_count > 0:
                        _print_timings(t_acc, t_count)

                # Skip if already processed
                if refcode in processed_refcodes:
                    continue

                t_entry_start = time.perf_counter()
                try:
                    # Read CIF string directly from ZIP member
                    _t = time.perf_counter()
                    with zf.open(member) as handle:
                        cif_string = _decode_cif_bytes(handle.read())
                    t_acc["cif_string"] += time.perf_counter() - _t

                    # Create xray.structure from CIF string
                    _t = time.perf_counter()
                    try:
                        xrs = load_cif(StringIO(cif_string))
                    except (Sorry, ValueError, RuntimeError, OSError, NameError):
                        continue
                    t_acc["load_cif"] += time.perf_counter() - _t

                    n_atoms = xrs.scatterers().size()
                    if n_atoms == 0:
                        continue

                    # Compute structure factors
                    _t = time.perf_counter()
                    reflections = compute_structure_factors(
                        xrs,
                        sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
                    )
                    t_acc["struct_factors"] += time.perf_counter() - _t

                    # Compute power spectrum
                    _t = time.perf_counter()
                    P = compute_power_spectrum(reflections, params=params)
                    desc = power_spectrum_to_vector(P)
                    t_acc["power_spectrum"] += time.perf_counter() - _t

                    # Store descriptor
                    refcodes.append(refcode)
                    descriptors.append(desc)
                    processed_refcodes.add(refcode)
                    new_count += 1
                    t_count += 1

                    # Compute hash
                    _t = time.perf_counter()
                    hash_str = descriptor_hash(desc)
                    t_acc["hash"] += time.perf_counter() - _t

                    # Write to CSV and print to stdout
                    writer.writerow([refcode, hash_str])
                    print(f"{refcode}\t{hash_str}")

                    # Flush periodically to save progress
                    if new_count % 100 == 0:
                        csvfile.flush()
                        with open(pickle_file, "wb") as f:
                            pickle.dump((refcodes, descriptors), f)

                except (Sorry, ValueError, RuntimeError, OSError, NameError) as e:
                    errors += 1
                    print(f"Error processing {refcode}: {e}", file=sys.stderr)
                finally:
                    t_acc["total"] += time.perf_counter() - t_entry_start
                    t_count += 1

        if t_count > 0:
            print("\nAverage timings per structure:", file=sys.stderr)
            _print_timings(t_acc, t_count)

        if args.num_chunks > 1:
            print(
                f"\nComplete! Checked {count} structures ({count - skipped_chunk} in chunk), "
                f"added {new_count} new ({errors} errors).",
                file=sys.stderr,
            )
        else:
            print(
                f"\nComplete! Checked {count} structures, added {new_count} new ({errors} errors).",
                file=sys.stderr,
            )
        print(f"Results saved to {csv_file}", file=sys.stderr)

    # Final save of descriptors
    with open(pickle_file, "wb") as f:
        pickle.dump((refcodes, descriptors), f)
    print(
        f"Descriptors saved to {pickle_file} ({len(refcodes)} total structures)",
        file=sys.stderr,
    )

    run_total = time.perf_counter() - run_t0
    if t_count > 0:
        print(
            f"Run wall-clock time: {run_total:.1f}s "
            f"({run_total / t_count * 1e3:.1f}ms per attempted structure)",
            file=sys.stderr,
        )
    else:
        print(f"Run wall-clock time: {run_total:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
