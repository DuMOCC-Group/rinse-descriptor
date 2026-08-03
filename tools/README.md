# RINSE Descriptor Tools

This directory contains scripts for computing and updating the PCA-based hash model used by the RINSE descriptor package.

## Scripts

### `compute_csd_hashes.py`

Computes RINSE descriptor hashes for all structures in the Cambridge Structural Database (CSD).

Descriptor computation is parallelised across worker processes (`--jobs`, default: all CPUs). CSD access (entry iteration and SHELX RES export) runs serially in the main process while the crystallography runs in the pool. Hashes are **2 proquint words by default** (`--n-words`). Every 100 new structures a per-position letter histogram is printed to stderr as a quick uniformity check.

**Outputs:**
- `csd_hashes.csv`: CSV file with refcode and hash columns (or `csd_hashes_chunk_N.csv` for chunks)
- `csd_descriptors.pkl`: Pickle file with refcodes and high-dimensional descriptors (or `csd_descriptors_chunk_N.pkl`)

**Usage:**
```bash
# Process all structures using all CPUs
python compute_csd_hashes.py

# Limit the number of worker processes
python compute_csd_hashes.py --jobs 8

# One-word hashes instead of the default two
python compute_csd_hashes.py --n-words 1

# Distribute across machines: split into 10 chunks, process chunk 0
python compute_csd_hashes.py 10 0

# Process chunk 5 of 10
python compute_csd_hashes.py 10 5
```

**Notes:**
- Requires access to the CSD via the `ccdc` Python API
- The script supports resumption: if interrupted, it will skip already-processed structures
- Descriptors are saved incrementally every 100 structures
- `--jobs` parallelises the compute within one run; index chunking (`num_chunks chunk_id`) distributes work across separate processes/machines (chunk N processes entries where `index % num_chunks == N`)

### `compute_descriptors.py`

Computes RINSE descriptors for every structure in a training-set pickle produced by `build_training_set.py` and writes them back into the same pickle under a `descriptor` key.

**Inputs:**
- A training-set pickle (list of record dicts, each with a `text` structure string and a `format` of `"cif"` or `"res"`).

**Outputs:**
- The input pickle, updated in place: each record gains a `descriptor` field (a float64 vector, or `None` with a `descriptor_error` message on failure).

**Usage:**
```bash
# Compute descriptors for the training split
uv run tools/compute_descriptors.py --input training_set_full.pkl

# Checkpoint more frequently
uv run tools/compute_descriptors.py --input training_set_full.pkl --checkpoint-every 200

# Re-attempt records that failed on a previous run
uv run tools/compute_descriptors.py --input training_set_full.pkl --retry-failed
```

**Notes:**
- Resumable: records that already carry a `descriptor` key are skipped, so re-running the same command continues where it left off.
- Progress (processed/total, percent, rate, elapsed, ETA) is printed as a live-updating line on stderr.
- The pickle is checkpointed to disk atomically every `--checkpoint-every` new descriptors and on Ctrl-C.

### `compute_pca.py`

**Inputs:**
Accepts either pickle layout:
- a training-set pickle from `build_training_set.py` populated by `compute_descriptors.py` (a list of record dicts each with a `descriptor` field; records without a descriptor are skipped)
- the legacy `(refcodes, descriptors)` two-tuple from `compute_csd_hashes.py`

**Outputs:**
- `../python/rinse_descriptor/data/pca_components.json`: PCA model for distribution

**Usage:**
```bash
# Use all components
python compute_pca.py

# Specify number of components (must be at least 16 for single-word hashes)
python compute_pca.py --n-components 50

# Custom input/output paths
python compute_pca.py --input training_set_full.pkl --output custom_output.json
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

### `compute_mp_hashes.py`

Computes RINSE descriptor hashes for structures from the Materials Project API.

**Outputs:**
- `mp_hashes.csv` (or `mp_hashes_chunk_N.csv` for chunks): CSV with `material_id,hash`
- `mp_descriptors.pkl` (or `mp_descriptors_chunk_N.pkl`): Pickle with material ids and descriptors

**Usage:**
```bash
# Set API key in environment (recommended)
$env:MP_API_KEY="<your-key>"
uv run tools/compute_mp_hashes.py

# Parallel processing chunk 0 of 10
uv run tools/compute_mp_hashes.py 10 0

# Single material id
uv run tools/compute_mp_hashes.py --material-id mp-149
```

**Notes:**
- Requires `mp-api` and `pymatgen`
- Supports resumption from existing `mp_descriptors*.pkl`
- Avoid hardcoding API keys in scripts

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
