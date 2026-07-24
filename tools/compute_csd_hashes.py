"""Compute RINSE descriptor hashes for all structures in the CSD.

Supports chunking for parallel processing on HPC clusters.

Outputs:
    - csd_hashes_chunk_N.csv: CSV file with refcode and hash columns
    - csd_descriptors_chunk_N.pkl: Pickle file with refcodes and high-dimensional descriptors
    - stdout: Tab-separated refcode and hash for each structure

Usage:
    python compute_csd_hashes.py                    # Process all structures
    python compute_csd_hashes.py 10 0               # Process chunk 0 of 10
    python compute_csd_hashes.py 10 1               # Process chunk 1 of 10
    python compute_csd_hashes.py --refcode AABHTZ  # Single refcode
"""

import argparse
import csv
import os
import pickle
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path

from ccdc.io import CrystalWriter, EntryReader
from rinse_descriptor import (
    RinseParams,
    compute_power_spectrum,
    compute_structure_factors,
    descriptor_hash,
    load_res,
    power_spectrum_to_vector,
)


def _print_timings(t_acc: dict, n: int) -> None:
    total = sum(t_acc.values())
    print(
        f"  avg timings over {n} structures (ms):  "
        f"res_string={t_acc['res_string'] / n * 1e3:.1f}  "
        f"load_res={t_acc['load_res'] / n * 1e3:.1f}  "
        f"struct_factors={t_acc['struct_factors'] / n * 1e3:.1f}  "
        f"power_spectrum={t_acc['power_spectrum'] / n * 1e3:.1f}  "
        f"hash={t_acc['hash'] / n * 1e3:.1f}  "
        f"total={total / n * 1e3:.1f}",
        file=sys.stderr,
    )


def _entry_to_res_string(entry: object) -> str:
    """Convert a CSD entry to SHELX RES text using the CSD writer API."""
    fd, tmp_name = tempfile.mkstemp(suffix=".res")
    os.close(fd)
    Path(tmp_name).unlink(missing_ok=True)

    try:
        with CrystalWriter(tmp_name, format="res") as writer:
            writer.write(entry.crystal)

        res_string = Path(tmp_name).read_text(encoding="utf-8", errors="replace")
        if not res_string.rstrip().endswith("END"):
            res_string = f"{res_string.rstrip()}\nEND\n"
        return res_string
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _process_single(refcode: str) -> None:
    """Compute and print the descriptor hash for one CSD refcode."""
    reader = EntryReader("CSD")
    entry = reader.entry(refcode)
    params = RinseParams()
    t0 = time.perf_counter()
    res_string = _entry_to_res_string(entry)
    t_res = time.perf_counter()
    xrs = load_res(StringIO(res_string))
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
    print(f"{refcode}\t{hash_str}")
    print(
        f"  res_string={(t_res - t0) * 1e3:.1f}ms  "
        f"load_res={(t_load - t_res) * 1e3:.1f}ms  "
        f"struct_factors={(t_sf - t_load) * 1e3:.1f}ms  "
        f"power_spectrum={(t_ps - t_sf) * 1e3:.1f}ms  "
        f"hash={(t_hash - t_ps) * 1e3:.1f}ms  "
        f"total={(t_hash - t0) * 1e3:.1f}ms",
        file=sys.stderr,
    )


def main():
    """Process all CSD structures and compute their RINSE descriptor hashes."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute RINSE descriptor hashes for CSD structures"
    )
    parser.add_argument(
        "num_chunks",
        type=int,
        nargs="?",
        default=1,
        help="Total number of chunks to split the CSD into (default: 1)",
    )
    parser.add_argument(
        "chunk_id",
        type=int,
        nargs="?",
        default=0,
        help="Which chunk to process (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--refcode",
        type=str,
        default=None,
        help="Process a single refcode and print its hash, then exit",
    )
    args = parser.parse_args()

    if args.refcode is not None:
        _process_single(args.refcode)
        return

    if args.chunk_id >= args.num_chunks:
        print(
            f"Error: chunk_id ({args.chunk_id}) must be less than num_chunks ({args.num_chunks})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Open CSD reader
    print("Opening CSD database...", file=sys.stderr)
    reader = EntryReader("CSD")

    # Determine output file names based on chunking
    if args.num_chunks > 1:
        chunk_suffix = f"_chunk_{args.chunk_id}"
        print(f"Processing chunk {args.chunk_id} of {args.num_chunks}...", file=sys.stderr)
    else:
        chunk_suffix = ""

    pickle_file = Path(f"csd_descriptors{chunk_suffix}.pkl")
    csv_file = Path(f"csd_hashes{chunk_suffix}.csv")

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
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["refcode", "hash"])

        # If resuming, write existing hashes to CSV first
        if resuming:
            print("Writing existing hashes to CSV...", file=sys.stderr)
            for refcode, desc in zip(refcodes, descriptors):
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
            "res_string": 0.0,
            "load_res": 0.0,
            "struct_factors": 0.0,
            "power_spectrum": 0.0,
            "hash": 0.0,
        }
        t_count = 0

        # Iterate through all structures
        print("Processing structures...", file=sys.stderr)
        for entry_idx, entry in enumerate(reader):
            refcode = entry.identifier
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
                        f"Checked {count} structures, added {new_count} new ({errors} errors)...",
                        file=sys.stderr,
                    )
                if t_count > 0:
                    _print_timings(t_acc, t_count)

            # Skip if already processed
            if refcode in processed_refcodes:
                continue

            try:
                # Build RES string from CSD entry
                _t = time.perf_counter()
                res_string = _entry_to_res_string(entry)
                t_acc["res_string"] += time.perf_counter() - _t

                # Create xray.structure from RES string
                _t = time.perf_counter()
                try:
                    xrs = load_res(StringIO(res_string))
                except Exception:
                    continue
                t_acc["load_res"] += time.perf_counter() - _t

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

            except Exception as e:
                errors += 1
                print(f"Error processing {refcode}: {e}", file=sys.stderr)
                continue

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


if __name__ == "__main__":
    main()
