# RINSE Descriptor Tools

This directory contains scripts for computing and updating the PCA-based hash model used by the RINSE descriptor package.

## Scripts

### `compute_csd_hashes.py`

Computes RINSE descriptor hashes for all structures in the Cambridge Structural Database (CSD).

**Outputs:**
- `csd_hashes.csv`: CSV file with refcode and hash columns
- `csd_descriptors.pkl`: Pickle file with refcodes and high-dimensional descriptors

**Usage:**
```bash
python compute_csd_hashes.py
```

**Notes:**
- Requires access to the CSD via the `ccdc` Python API
- The script supports resumption: if interrupted, it will skip already-processed structures
- Descriptors are saved incrementally every 100 structures

### `compute_pca.py`

Performs PCA on the collected descriptors and saves the principal components to the package data directory.

**Outputs:**
- `../python/rinse_descriptor/data/pca_components.json`: PCA model for distribution

**Usage:**
```bash
# Use all components
python compute_pca.py

# Specify number of components (must be at least 16 for single-word hashes)
python compute_pca.py --n-components 50

# Custom input/output paths
python compute_pca.py --input custom_descriptors.pkl --output custom_output.json
```

**Output format:**
The JSON file contains:
- `components`: PCA component vectors (n_components × n_features)
- `mean`: Mean vector for centering
- `explained_variance`: Variance explained by each component
- `explained_variance_ratio`: Fraction of variance explained
- `singular_values`: Singular values from SVD
- `n_components`, `n_samples`, `n_features`: Model metadata

## Workflow

To update the PCA model distributed with the package:

1. **Collect descriptors:**
   ```bash
   python compute_csd_hashes.py
   ```
   This creates `csd_descriptors.pkl` with high-dimensional descriptors from the CSD.

2. **Compute PCA:**
   ```bash
   python compute_pca.py
   ```
   This generates `../python/rinse_descriptor/data/pca_components.json` that will be bundled with the package.

3. **Test the hash function:**
   ```python
   from rinse_descriptor import descriptor, descriptor_hash, Crystal
   
   crystal = Crystal.from_cif("test.cif")
   desc = descriptor(crystal)
   hash_str = descriptor_hash(desc)
   print(hash_str)  # e.g., "lusab-babad"
   ```

4. **Build and distribute:**
   The PCA components in `python/rinse_descriptor/data/` will automatically be included in the package wheel.

## Requirements

- `ccdc` (for CSD access)
- `scikit-learn` (for PCA)
- All rinse-descriptor dependencies

The PCA model uses the first 16 principal components by default (for 1-word hashes), but more components can be stored for multi-word hashes if needed.
