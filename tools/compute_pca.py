"""Compute PCA on RINSE descriptors from pickle file.

Loads high-dimensional descriptors from csd_descriptors.pkl, performs PCA,
and saves the principal component vectors to a JSON file for package distribution.

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
    with open(input_file, "rb") as f:
        refcodes, descriptors = pickle.load(f)

    # Convert to numpy array if not already
    descriptors = np.array(descriptors)
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
