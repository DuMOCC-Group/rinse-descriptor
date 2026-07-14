"""Merge chunk files produced by parallel compute_csd_hashes.py runs.

Usage:
    python merge_chunks.py <num_chunks>
    python merge_chunks.py 10  # Merge 10 chunk files
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge CSD hash chunk files")
    parser.add_argument("num_chunks", type=int, help="Total number of chunk files to merge")
    args = parser.parse_args()

    all_refcodes = []
    all_descriptors = []
    all_hashes = {}

    print(f"Merging {args.num_chunks} chunk files...")

    # Load all chunk files
    for chunk_id in range(args.num_chunks):
        pickle_file = Path(f"csd_descriptors_chunk_{chunk_id}.pkl")
        csv_file = Path(f"csd_hashes_chunk_{chunk_id}.csv")

        if not pickle_file.exists():
            print(f"Warning: {pickle_file} not found, skipping...", file=sys.stderr)
            continue

        print(f"Loading chunk {chunk_id}...")

        # Load descriptors
        with open(pickle_file, "rb") as f:
            refcodes, descriptors = pickle.load(f)
            all_refcodes.extend(refcodes)
            all_descriptors.extend(descriptors)

        # Load hashes from CSV
        if csv_file.exists():
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_hashes[row["refcode"]] = row["hash"]

        print(f"  Loaded {len(refcodes)} structures from chunk {chunk_id}")

    print(f"\nTotal structures: {len(all_refcodes)}")

    # Save merged pickle
    print("Saving merged csd_descriptors.pkl...")
    with open("csd_descriptors.pkl", "wb") as f:
        pickle.dump((all_refcodes, all_descriptors), f)

    # Save merged CSV
    print("Saving merged csd_hashes.csv...")
    with open("csd_hashes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["refcode", "hash"])
        for refcode in all_refcodes:
            if refcode in all_hashes:
                writer.writerow([refcode, all_hashes[refcode]])

    print(f"Done! Merged {len(all_refcodes)} structures.")


if __name__ == "__main__":
    main()
