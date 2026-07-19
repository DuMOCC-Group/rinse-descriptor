# RINSE – Reciprocal-space INvariant Spectral Embedding

RINSE computes rotationally invariant descriptors of crystalline materials
by projecting the intensity-weighted reciprocal lattice onto a combined radial
and angular basis (analogous to SOAP, but working entirely in reciprocal space).

## Descriptor formulation

For a crystal, all reflections $\mathbf{G}_{\mathrm{hkl}}$ within a resolution cutoff
(default $\sin{\theta}/\lambda \leq 0.6 Å^{-1}$, i.e. $|\mathbf{G}| \leq 1.2 Å^{-1}$) are enumerated.
Each reflection is assigned an intensity $\mathrm{I}(\mathbf{G}) = \lvert\mathrm{F}(\mathbf{G})\rvert^{2}$ from the
structure factor calculated via cctbx direct summation with Waasmaier-Kirfel
X-ray form factors by default. The descriptor is always weighted by intensities,
not amplitudes. Isotropic and anisotropic displacement parameters are read from
the CIF by default; empirical reciprocal-space intensity normalisation then
removes the mean resolution envelope, followed by an isotropic Debye-Waller
falloff that softly damps high-resolution reflections.

The expansion coefficients are:

$$
A_{nlm} = Σ_\mathbf{G}  I(\mathbf{G}) · R_n(|\mathbf{G}|) · Y_l^m(\hat{\mathbf{G}})
$$

and the rotationally invariant power spectrum is:

$$
p_{nl} = Σ_m \lvert A_{nlm}\rvert^{2}
$$

Because the intensity field is centrosymmetric if anomalous dispersion is not considered, only even *l*
contributes. By default, RINSE also drops the monopole (*l* = 0) and quadrupole
(*l* = 2) terms. Default parameters give a **8 × 16 = 128-element** descriptor:

| Axis | Values | Count |
|------|--------|-------|
| Radial (*n*) | 0, 1, …, 7 | 8 |
| Angular (*l*) | 4, 6, 8, …, 34 (even only) | 16 |

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

# Run the demo.py notebook
uv run marimo edit demo.py
```

### From PyPI

```bash
pip install rinse-descriptor
# or
uv add rinse-descriptor
```

## Quick start

### From a CIF file

```python
from rinse_descriptor import RinseParams, descriptor, descriptor_many

# Single structure → 1-D feature vector
x = descriptor("mystructure.cif")
print(x.shape)  # (128,)

# Return the 2-D power-spectrum matrix instead
params = RinseParams(flatten=False)
x_mat = descriptor("mystructure.cif", params=params)
print(x_mat.shape)  # (8, 16)

# Batch of structures → (N, 128)
structures = ["structure_1.cif", "structure_2.cif"]
X = descriptor_many(structures)
print(X.shape)  # (2, 128)
```

### From a loaded cctbx structure

```python
from rinse_descriptor import descriptor, load_cif

xrs = load_cif("mystructure.cif")
x = descriptor(xrs)
print(x.shape)  # (128,)
```

### Custom parameters

```python
from rinse_descriptor import RinseParams, descriptor

params = RinseParams(
    n_max=8,                       # radial basis order (n = 0 … 7)
    l_max=36,                       # angular levels (gives l = 4,6,...,34 by default)
    sin_theta_over_lambda_max=0.6,  # resolution cutoff in Å⁻¹
    radial_basis="chebyshev",       # or "bessel" / "smooth_shells_cw" / "smooth_shells_nl"
    intensity_normalisation="none",  # optional: disable the default empirical envelope removal
    intensity_falloff="none",        # optional: disable the default Debye-Waller falloff
)
x = descriptor("mystructure.cif", params=params)
```

### Form factors, intensity normalisation, and falloff

```python
from rinse_descriptor import descriptor

# Electron scattering factors; descriptor weights are still intensities I = |F|²
x = descriptor("mystructure.cif", form_factor_type="electron")

# Empirical envelope removal and Debye-Waller falloff are on by default.
# Set intensity_normalisation="none" and/or intensity_falloff="none" to opt out.
x_norm = descriptor("mystructure.cif")
```

Available `form_factor_type` values: `"xray"` (default), `"electron"`, `"neutron"`.

Available `intensity_normalisation` values: `"empirical"` (default), `"none"`.

Available `intensity_falloff` values: `"debye_waller"` (default), `"none"`.
For `"debye_waller"`, `intensity_falloff_u_iso` sets the average isotropic
displacement parameter used in the amplitude factor
`exp(-8 * pi**2 * U_iso * (sin(theta)/lambda)**2)`; the default is `0.01` Å².

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
│   ├── _crystal.py        # CIF loading into cctbx xray.structure objects
│   ├── _structure_factors.py  # cctbx structure factor calculation
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

The `ReflectionList` and `RinseParams` types are designed to be shared between
the core descriptor and the diffraction submodule.

## License

MIT
