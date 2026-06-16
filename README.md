# RINSE – Reciprocal-space INvariant Spectral Embedding

RINSE computes rotationally invariant descriptors of crystalline materials
by projecting the intensity-weighted reciprocal lattice onto a combined radial
and angular basis (analogous to SOAP, but working entirely in reciprocal space).

## Descriptor formulation

For a crystal, all reflections $\mathbf{G}_{\mathrm{hkl}}$ within a resolution cutoff
(default $\sin{\theta}/\lambda \leq 0.6 Å^{-1}$ , i.e. $\mathbf{G} \leq 0.6 Å^{-1}$) are enumerated.
Each reflection is assigned an intensity $\mathrm{I}(\mathbf{G}) = \lvert\mathrm{F}(\mathbf{G})\rvert^{2}$ from the
structure factor calculated via Gemmi (direct summation, IT92 X-ray form
factors by default).

The expansion coefficients are:

$$
A_{nlm} = Σ_\mathbf{G}  I(\mathbf{G}) · R_n(\mathbf{G}) · Y_l^m(\hat{\mathbf{G}})
$$

and the rotationally invariant power spectrum is:

$$
p_{nl} = Σ_m \lvert A_{nlm}\rvert^{2}
$$

Because the intensity field is centrosymmetric if anomalous dispersion is not considered, only even *l*
contributes. Default parameters give a **8 × 16 = 128-element** descriptor:

| Axis | Values | Count |
|------|--------|-------|
| Radial (*n*) | 0, 1, …, 7 | 8 |
| Angular (*l*) | 0, 2, 4, …, 30 (even only) | 16 |

## Installation

### From source (requires `uv`)

```bash
# Clone
git clone https://github.com/DuMOCC-Group/rinse-descriptor.git
cd rinse-descriptor

# Install uv (if not already available)
curl -Ls https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

### From PyPI (once published)

```bash
pip install rinse-descriptor
# or
uv add rinse-descriptor
```

## Quick start

### From an ASE Atoms object

```python
from ase.build import bulk
from rinse_descriptor import descriptor, descriptor_many, RinseParams

# Single structure → (8, 16) matrix
atoms = bulk("NaCl", "rocksalt", a=5.64)
x = descriptor(atoms)
print(x.shape)  # (8, 16)

# Flatten to 1-D feature vector
x_vec = descriptor(atoms, flatten=True)
print(x_vec.shape)  # (128,)

# Batch of structures → (N, 8, 16)
structures = [bulk("Si", "diamond", a=5.43), bulk("Cu", "fcc", a=3.62)]
X = descriptor_many(structures)
print(X.shape)  # (2, 8, 16)
```

### From a CIF file

```python
from rinse_descriptor import descriptor

x = descriptor("mystructure.cif")
print(x.shape)  # (8, 16)
```

### Custom parameters

```python
from rinse_descriptor import RinseParams, descriptor

params = RinseParams(
    n_max=8,                       # radial basis order (n = 0 … 7)
    l_max=16,                       # angular levels (gives l = 0,2,...,30)
    sin_theta_over_lambda_max=0.6,  # resolution cutoff in Å⁻¹
    radial_basis="chebyshev",       # or "bessel" / "smooth_shells_cw" / "smooth_shells_nl"
)
x = descriptor(atoms, params=params)
```

### Form factors and structure factor type

```python
from rinse_descriptor import descriptor

# Electron scattering factors, intensities
x = descriptor(atoms, form_factor_type="electron", structure_factor_type="F2")

# Neutron scattering lengths, amplitudes
x = descriptor(atoms, form_factor_type="neutron", structure_factor_type="F")
```

Available `form_factor_type` values: `"xray"` (default), `"electron"`, `"neutron"`, `"unity"`.

Available `structure_factor_type` values: `"F2"` (default), `"F"`.

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run benchmarks
uv run pytest benchmarks/ --benchmark-only -v

# Lint / format
uv run ruff check python/ tests/
uv run ruff format python/ tests/
uv run mypy python/rinse_descriptor/
```

### Pre-commit hooks (recommended)

This repository includes a `.pre-commit-config.yaml` that runs:

- On commit: `ruff check --fix` and `ruff format` for staged Python files.
- On push: full `ruff check`, `mypy`, and `pytest`.

Install and run once:

```bash
uv sync --group dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pre-commit run --all-files
```

## Project structure

```
rinse-descriptor/
├── python/rinse_descriptor/  # Python package
│   ├── __init__.py        # Public API: descriptor(), descriptor_many()
│   ├── _crystal.py        # Crystal dataclass (ASE/Gemmi-independent)
│   ├── _structure_factors.py  # Gemmi structure factor calculation
│   ├── _radial_basis.py   # Chebyshev / Bessel / smooth-shell radial bases
│   └── _descriptor.py     # Power spectrum computation
├── tests/                 # pytest test suite
├── benchmarks/            # pytest-benchmark suite
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Future: `rinse_descriptor.diffraction` submodule

The package is designed to support a future `rinse_descriptor.diffraction` submodule that will provide:

- Indexed diffraction patterns (hkl, d-spacing, intensity)
- Unindexed powder diffraction patterns (2θ or *d*-spacing profiles)
- Reciprocal-space descriptors beyond the power spectrum (e.g. bispectrum)

The `Crystal`, `ReflectionList`, and `RinseParams` types are designed to be
shared between the core descriptor and the diffraction submodule.

## License

MIT
