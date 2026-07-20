# RINSE Descriptor Tools

This directory contains scripts for computing and updating the PCA-based hash model used by the RINSE descriptor package.

## Scripts

### `compute_csd_hashes.py`

Computes RINSE descriptor hashes for all structures in the Cambridge Structural Database (CSD).

**Outputs:**
- `csd_hashes.csv`: CSV file with refcode and hash columns (or `csd_hashes_chunk_N.csv` for chunks)
- `csd_descriptors.pkl`: Pickle file with refcodes and high-dimensional descriptors (or `csd_descriptors_chunk_N.pkl`)

**Usage:**
```bash
# Process all structures sequentially
python compute_csd_hashes.py

# Parallel processing: split into 10 chunks, process chunk 0
python compute_csd_hashes.py 10 0

# Process chunk 5 of 10
python compute_csd_hashes.py 10 5
```

**Notes:**
- Requires access to the CSD via the `ccdc` Python API
- The script supports resumption: if interrupted, it will skip already-processed structures
- Descriptors are saved incrementally every 100 structures
- Chunking distributes entries by index: chunk N processes entries where `index % num_chunks == N`

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

### `converge_csd_hashes.py`

Runs `compute_csd_hashes.py` and `compute_pca.py` on progressively larger
prefixes of a chunked CSD split until hashes for a fixed test chunk stop
changing.

**Default workflow:**
- Split the CSD into `1000` chunks
- Use chunk `0` as the fixed convergence subset
- Refit PCA after each additional chunk
- Stop when the chunk-0 hashes are unchanged for one evaluation round

**Usage:**
```bash
# Default convergence run: 1000 chunks, test chunk 0
uv run tools/converge_csd_hashes.py

# Require two consecutive unchanged rounds before accepting convergence
uv run tools/converge_csd_hashes.py --stable-rounds 2

# Limit the run while testing the workflow
uv run tools/converge_csd_hashes.py --max-chunks 25 --work-dir tmp/convergence
```

**Outputs:**
- `convergence_runs/csd_descriptors_chunk_N.pkl`: per-chunk descriptor pickles
- `convergence_runs/csd_descriptors_prefix_N.pkl`: cumulative descriptor pickles used for PCA
- `convergence_runs/pca_components_prefix_N.json`: PCA model after each evaluation round
- `convergence_runs/convergence_history.csv`: per-round convergence summary
- `../python/rinse_descriptor/data/pca_components.json`: copied only after convergence by default

### `merge_chunks.py`

Merges chunk files produced by parallel runs of `compute_csd_hashes.py`.

**Usage:** (sequential):**
   ```bash
   python compute_csd_hashes.py
   ```

1. **Collect descriptors:**
   ```bash
   # Submit array job with 10 parallel workers
   sbatch submit_parallel.sh
   
   # After all jobs complete, merge results
   python merge_chunks.py 10rge 10 chunk files into single files
```

**Outputs:**
- `csd_descriptors.pkl`: Merged descriptors from all chunks
- `csd_hashes.csv`: Merged hashes from all chunks

### `submit_parallel.sh`

Example SLURM batch script for parallel processing on HPC clusters.

**Configuration:**
- Edit `--array=0-9` to set number of parallel jobs
- Edit `NUM_CHUNKS=10` to match array size
- Adjust memory, time, and CPU requirements as needed

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

   Or, to estimate when the hash assignments have stabilised before updating the
   bundled PCA model:
   ```bash
   uv run tools/converge_csd_hashes.py
   ```

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
