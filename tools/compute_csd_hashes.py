"""Compute RINSE descriptor hashes for all structures in the CSD.

Outputs:
    - csd_hashes.csv: CSV file with refcode and hash columns
    - csd_descriptors.pkl: Pickle file with refcodes and high-dimensional descriptors
    - stdout: Tab-separated refcode and hash for each structure

Usage:
    python compute_csd_hashes.py
"""

import csv
import pickle
import sys
from io import StringIO
from pathlib import Path

from ccdc.io import EntryReader
from rinse_descriptor import Crystal, descriptor, descriptor_hash


def main():
    """Process all CSD structures and compute their RINSE descriptor hashes."""
    # Open CSD reader
    print("Opening CSD database...", file=sys.stderr)
    reader = EntryReader("CSD")

    # Storage for descriptors - load existing data if available
    pickle_file = Path("csd_descriptors.pkl")
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
    with open("csd_hashes.csv", "w", newline="") as csvfile:
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

        # Iterate through all structures
        print("Processing structures...", file=sys.stderr)
        for entry in reader:
            refcode = entry.identifier
            count += 1

            if count % 1000 == 0:
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

                # Create Crystal from CIF string
                try:
                    crystal = Crystal.from_cif(StringIO(cif_string))
                except Exception:
                    continue

                if len(crystal.positions) == 0 or len(crystal.positions) > 200:
                    continue

                # Compute descriptor
                desc = descriptor(crystal)

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
                    with open("csd_descriptors.pkl", "wb") as f:
                        pickle.dump((refcodes, descriptors), f)

            except Exception as e:
                errors += 1
                print(f"Error processing {refcode}: {e}", file=sys.stderr)
                continue

        print(
            f"\nComplete! Checked {count} structures, added {new_count} new ({errors} errors).",
            file=sys.stderr,
        )
        print("Results saved to csd_hashes.csv", file=sys.stderr)

    # Final save of descriptors
    with open("csd_descriptors.pkl", "wb") as f:
        pickle.dump((refcodes, descriptors), f)
    print(
        f"Descriptors saved to csd_descriptors.pkl ({len(refcodes)} total structures)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
