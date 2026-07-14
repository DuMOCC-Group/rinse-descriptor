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
"""

import argparse
import csv
import pickle
import sys
from io import StringIO
from pathlib import Path

from ccdc.io import EntryReader
from rinse_descriptor import descriptor, descriptor_hash, load_cif


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
    args = parser.parse_args()

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
        print(
            f"Processing chunk {args.chunk_id} of {args.num_chunks}...", file=sys.stderr
        )
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

            if count % 1000 == 0:
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

            # Skip if already processed
            if refcode in processed_refcodes:
                continue

            try:
                # Get CIF string
                cif_string = entry.to_string(format="cif")

                # Create xray.structure from CIF string
                try:
                    xrs = load_cif(StringIO(cif_string))
                except Exception:
                    continue

                n_atoms = xrs.scatterers().size()
                if n_atoms == 0 or n_atoms > 200:
                    continue

                # Compute descriptor
                desc = descriptor(xrs)

                # Store descriptor
                refcodes.append(refcode)
                descriptors.append(desc)
                processed_refcodes.add(refcode)
                new_count += 1

                # Compute hash (single word)
                hash_str = descriptor_hash(desc)

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

