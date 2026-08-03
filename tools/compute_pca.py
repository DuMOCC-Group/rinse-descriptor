"""Compute PCA on RINSE descriptors from pickle file.

Loads high-dimensional descriptors from a pickle file, performs PCA, and saves
the principal component vectors to a JSON file for package distribution.

Two pickle layouts are accepted:

    * a training-set pickle written by ``build_training_set.py`` and populated
      by ``compute_descriptors.py`` (a list of record dicts, each with a
      ``descriptor`` field); records without a valid descriptor are skipped.
    * the legacy ``(refcodes, descriptors)`` two-tuple written by
      ``compute_csd_hashes.py``.

Outputs:
    - ../python/rinse_descriptor/data/pca_components.json (default)

Usage:
    python compute_pca.py
    python compute_pca.py --n-components 50  # Specify number of components
    python compute_pca.py --output custom_path.json  # Custom output location
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def _load_descriptors(input_file: Path) -> tuple[list, np.ndarray]:
    """Load ``(labels, descriptors)`` from either supported pickle layout.

    Records/entries without a usable descriptor (missing or ``None``) are
    skipped.  Returns the list of labels and a 2-D float array of descriptors.
    """
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    # Training-set layout: list of record dicts with a "descriptor" field.
    if isinstance(data, list) and (not data or isinstance(data[0], dict)):
        labels: list = []
        descriptors: list = []
        skipped = 0
        for record in data:
            desc = record.get("descriptor")
            if desc is None:
                skipped += 1
                continue
            labels.append(record.get("id", len(labels)))
            descriptors.append(np.asarray(desc, dtype=np.float64))
        if skipped:
            print(f"Skipped {skipped} record(s) without a descriptor.")
        return labels, np.asarray(descriptors, dtype=np.float64)

    # Legacy layout: (refcodes, descriptors) two-tuple.
    labels, descriptors = data
    return list(labels), np.asarray(descriptors, dtype=np.float64)


def main():
    """Load descriptors, perform PCA, and save results."""
    # Default output path is in the package data directory
    default_output = (
        Path(__file__).parent.parent
        / "python"
        / "rinse_descriptor"
        / "data"
        / "pca_components.json"
    )

    parser = argparse.ArgumentParser(description="Perform PCA on RINSE descriptors")
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of principal components (default: min(n_samples, n_features))",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="csd_descriptors.pkl",
        help="Input pickle file (default: csd_descriptors.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help=f"Output JSON file (default: {default_output})",
    )
    args = parser.parse_args()

    # Load descriptors
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file {args.input} not found")
        return

    print(f"Loading descriptors from {args.input}...")
    refcodes, descriptors = _load_descriptors(input_file)

    if len(descriptors) == 0:
        print("Error: no descriptors found in input file")
        return

    print(f"Loaded {len(refcodes)} descriptors with shape {descriptors.shape}")

    # Perform PCA
    print(f"Performing PCA with n_components={args.n_components}...")
    pca = PCA(n_components=args.n_components)
    pca.fit(descriptors)

    print(f"PCA complete: {pca.n_components_} components")
    print(f"Explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")
    print(
        f"Cumulative explained variance (first 10): {np.cumsum(pca.explained_variance_ratio_[:10])}"
    )

    # Prepare output data
    output_data = {
        "n_components": int(pca.n_components_),
        "n_features": int(descriptors.shape[1]),
        "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "mean": pca.mean_.tolist(),
    }

    # Save to JSON
    print(f"Saving PCA results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Done! PCA results saved to {args.output}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")


if __name__ == "__main__":
    main()
