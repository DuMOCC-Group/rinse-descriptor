# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.9",
#     "matplotlib",
#     "numpy",
#     "rinse-descriptor==1.0.7",
# ]
# ///
"""RINSE Descriptor Demo – marimo notebook.

Run with:
    uv run marimo edit demo.py
or in read-only mode:
    uv run marimo run demo.py
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
async def _():
    import sys

    import marimo as mo

    if sys.platform == "emscripten":
        import micropip

        # gemmi has no pure-Python wheel; stub it so micropip can resolve
        # rinse-descriptor's dependencies. Our code detects the stub at
        # runtime and falls back to the built-in pure-Python implementation.
        micropip.add_mock_package("gemmi", "0.7.0")
    return (mo,)


@app.cell(hide_code=True)
def _():
    import dataclasses

    import matplotlib.pyplot as plt
    import numpy as np
    from rinse_descriptor import (
        DEFAULT_HASH_WORDS,
        Crystal,
        RinseParams,
        descriptor,
        descriptor_hash,
    )
    from rinse_descriptor._descriptor import compute_power_spectrum, power_spectrum_to_vector
    from rinse_descriptor._structure_factors import compute_structure_factors

    return (
        Crystal,
        DEFAULT_HASH_WORDS,
        RinseParams,
        compute_power_spectrum,
        compute_structure_factors,
        dataclasses,
        descriptor,
        descriptor_hash,
        np,
        plt,
        power_spectrum_to_vector,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RINSE Descriptor Explorer

    **Reciprocal-space INvariant Spectral Embedding*

    Computes the intensity-weighted reciprocal-space power spectrum for any
    crystal structure. The descriptor is by default a **8 × 16 matrix** (128 elements)
    indexed by radial order *n* and even angular
    level *ℓ* ∈ {0, 2, 4, …, 62}.

    Upload a CIF file, then adjust
    the parameters below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Structure input
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cif_upload = mo.ui.file(
        filetypes=[".cif"],
        label="Upload CIF file",
    )
    cif_upload
    return (cif_upload,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Descriptor parameters
    Radial basis has a significant impact on the descriptor.
    The smooth_shells_nl seems to work well, equally weighting
    all parts of reciprocal space in a predictable way.
    smooth_shells_cw (with high n_max) can be used to generate
    a quasi powder pattern at each multipole level. Chebyshev
    and Bessel (0th order) functions are included because they
    are well behaved but seem to give less useful descriptors.

    X-ray form factors really the only sensible option.
    Others are included just for comparison.

    l_min and l_max set the number of spherical harmonic levels.
    Keep l_max low, as the basis functions are relatively slow
    to compute. You may want to set l_min to 4 to remove l_0
    (monopole) and l_3 (quadrupole levels). Monopoles (if
    present) dominate the descriptor as they have no negative
    regions. Quadrupoles vanish in cubic symmetry crystals,
    so contribute no information. "Include odd l" is an
    option to demonstrate that the odd levels (dipole,
    octopole...) are always zero because of Friedel's law, if
    the structure factor calculations neglect anomalous
    dispersion (they do).

    sin_theta_over_lambda_max controls how far out strucure
    factors are calculated. This is generally the slowest step.
    0.6 is atomic resolution, and this should be more than enough.
    It may be justfied to reduce this.

    log1p compression reduces the dynamic range of the descriptor.
    This is generally required when monopoles are included.
    l2 normalisation simply puts all descriptor vectors on a
    common scale. This should be left on.
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT_HASH_WORDS, RinseParams, dataclasses, mo):
    _defaults = dataclasses.asdict(RinseParams())

    n_max_slider = mo.ui.slider(
        start=2,
        stop=64,
        step=2,
        value=_defaults["n_max"],
        label="n_max  (radial basis functions, n = 0 … n_max−1)",
        show_value=True,
    )
    l_max_slider = mo.ui.slider(
        start=2,
        stop=64,
        step=2,
        value=_defaults["l_max"],
        label="l_max  (max ℓ exclusive, ℓ = l_min, l_min+2, …, l_max−2)",
        show_value=True,
    )
    l_min_slider = mo.ui.slider(
        start=0,
        stop=8,
        step=2,
        value=_defaults["l_min"],
        label="l_min  (first angular level included, ℓ ≥ l_min)",
        show_value=True,
    )
    stol_slider = mo.ui.slider(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=_defaults["sin_theta_over_lambda_max"],
        label="sin(θ)/λ_max  (Å⁻¹)  →  |G|_max = 2 × this value",
        show_value=True,
    )
    basis_dd = mo.ui.dropdown(
        options=["chebyshev", "bessel", "smooth_shells_cw", "smooth_shells_nl"],
        value="smooth_shells_nl",  # demo default; library default is "chebyshev"
        label="Radial basis",
    )
    ff_dd = mo.ui.dropdown(
        options=["xray", "electron", "neutron", "unity"],
        value="xray",
        label="Form factor type",
    )
    norm_dd = mo.ui.dropdown(
        options=["F2", "F"],
        value="F2",
        label="Structure factor type",
    )
    log1p_compression_cb = mo.ui.checkbox(value=_defaults["log1p"], label="log1p compression")
    l2_normalisation_cb = mo.ui.checkbox(value=_defaults["l2"], label="l2 normalisation")
    include_odd_l_cb = mo.ui.checkbox(value=_defaults["include_odd_l"], label="include odd ℓ")
    n_words_slider = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=DEFAULT_HASH_WORDS,
        label="hash words  (each word = 16 bits)",
        show_value=True,
    )
    mo.hstack(
        [
            mo.vstack([n_max_slider, l_max_slider, l_min_slider, stol_slider], gap="0.6rem"),
            mo.vstack(
                [
                    basis_dd,
                    ff_dd,
                    norm_dd,
                    log1p_compression_cb,
                    l2_normalisation_cb,
                    include_odd_l_cb,
                    n_words_slider,
                ],
                gap="0.6rem",
            ),
        ],
        gap="3rem",
        align="start",
    )
    return (
        basis_dd,
        ff_dd,
        include_odd_l_cb,
        l2_normalisation_cb,
        l_max_slider,
        l_min_slider,
        log1p_compression_cb,
        n_max_slider,
        n_words_slider,
        norm_dd,
        stol_slider,
    )


@app.cell(hide_code=True)
def _(
    Crystal,
    RinseParams,
    basis_dd,
    cif_upload,
    compute_power_spectrum,
    compute_structure_factors,
    ff_dd,
    include_odd_l_cb,
    l2_normalisation_cb,
    l_max_slider,
    l_min_slider,
    log1p_compression_cb,
    n_max_slider,
    norm_dd,
    power_spectrum_to_vector,
    stol_slider,
):
    import os
    import tempfile

    _crystal = None
    _struct_label = ""
    _error = None

    try:
        if cif_upload.value:
            _f = cif_upload.value[0]
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as _tmp:
                _tmp.write(_f.contents)
                _tmp_path = _tmp.name
            try:
                _crystal = Crystal.from_cif(_tmp_path)
                _struct_label = _f.name
            finally:
                os.unlink(_tmp_path)
    except Exception as _e:
        _error = str(_e)

    _params = None
    _refls = None
    _P = None
    _vec = None

    if _crystal is not None and _error is None:
        try:
            _params = RinseParams(
                n_max=n_max_slider.value,
                l_max=l_max_slider.value,
                l_min=l_min_slider.value,
                include_odd_l=include_odd_l_cb.value,
                sin_theta_over_lambda_max=stol_slider.value,
                radial_basis=basis_dd.value,
                log1p=log1p_compression_cb.value,
                l2=l2_normalisation_cb.value,
                flatten=False,
            )
            _refls = compute_structure_factors(
                _crystal,
                sin_theta_over_lambda_max=stol_slider.value,
                form_factor_type=ff_dd.value,
                structure_factor_type=norm_dd.value,
                debug=True,
            )
            _P = compute_power_spectrum(
                _refls,
                params=_params,
                debug=True,
            )
            _vec = power_spectrum_to_vector(_P)
        except Exception as _e:
            _error = str(_e)

    crystal = _crystal
    params = _params
    P = _P
    vec = _vec
    compute_error = _error
    if compute_error:
        print(compute_error)
    return P, compute_error, crystal, os, params, tempfile, vec


@app.cell(hide_code=True)
def _(P, compute_error, mo, plt):
    if compute_error or P is None:
        mo.stop(True)

    _fig, _axes = plt.subplots(2, 1, figsize=(12, 6))
    _fig.suptitle("RINSE Descriptor", fontsize=13, fontweight="bold")

    # Panel 1 – heatmap
    _ax = _axes[0]
    _im = _ax.imshow(P, origin="lower", aspect="auto", cmap="inferno", interpolation="nearest")
    _ax.set_xlabel("Angular level k  (ℓ = 2k)", fontsize=9)
    _ax.set_ylabel("Radial order n", fontsize=9)
    _ax.set_title("Descriptor matrix  p(n, ℓ)", fontsize=10)
    _n_l = P.shape[1]
    _kticks = list(range(0, _n_l, max(1, _n_l // 8)))
    _ax.set_xticks(_kticks)
    _ax.set_xticklabels([str(2 * k) for k in _kticks], fontsize=7)
    plt.colorbar(_im, ax=_ax, shrink=0.85)

    # Panel 2 – radial & angular profiles
    _ax2 = _axes[1]
    _radial = P.sum(axis=1)
    _angular = P.sum(axis=0)
    _l_vals = [2 * k for k in range(P.shape[1])]
    _ax2b = _ax2.twinx()
    _ax2.bar(range(P.shape[0]), _radial, color="#4477AA", alpha=0.7, label="Radial Σ_ℓ p(n,ℓ)")
    _ax2b.plot(_l_vals, _angular, "o-", color="#EE6677", ms=4, lw=1.5, label="Angular Σ_n p(n,ℓ)")
    _ax2.set_xlabel("n  /  ℓ", fontsize=9)
    _ax2.set_ylabel("Σ_ℓ p(n, ℓ)", fontsize=9, color="#4477AA")
    _ax2b.set_ylabel("Σ_n p(n, ℓ)", fontsize=9, color="#EE6677")
    _ax2.set_title("Radial (bars) and angular (line) profiles", fontsize=10)
    _lines = _ax2.get_legend_handles_labels()[0] + _ax2b.get_legend_handles_labels()[0]
    _lbls = _ax2.get_legend_handles_labels()[1] + _ax2b.get_legend_handles_labels()[1]
    _ax2.legend(_lines, _lbls, fontsize=7, loc="upper right")

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(P, compute_error, mo, np, plt, vec):
    if compute_error or P is None:
        mo.stop(True)

    _vec = vec
    _fig2, _ax = plt.subplots(figsize=(12, 2.5))
    _ax.fill_between(range(len(_vec)), _vec, alpha=0.7, color="#66CCEE")
    _ax.set_xlabel("Descriptor index  (row-major: i·l_max + k)", fontsize=9)
    _ax.set_ylabel("p_{nk}", fontsize=9)
    _ax.set_title(
        f"Descriptor vector  ({len(_vec)} elements, L2 norm = {float(np.linalg.norm(_vec)):.4g})",
        fontsize=10,
    )
    _ax.set_xlim(0, len(_vec) - 1)
    # plt.yscale('log')
    plt.tight_layout()
    _fig2
    return


@app.cell(hide_code=True)
def _(P, compute_error, mo, np):
    if compute_error or P is None:
        mo.stop(True)

    _vec = P.ravel()
    mo.md(f"""
    ### Descriptor statistics

    | Metric | Value |
    |---|---|
    | Shape | {P.shape[0]} × {P.shape[1]} |
    | Non-zero elements | {int(np.count_nonzero(_vec))} / {len(_vec)} |
    | Min | {float(_vec.min()):.6g} |
    | Max | {float(_vec.max()):.6g} |
    | Mean | {float(_vec.mean()):.6g} |
    | L2 norm | {float(np.linalg.norm(_vec)):.6g} |
    | Sparsity | {100.0 * float(np.mean(_vec == 0)):.1f}% |
    """)
    return


@app.cell(hide_code=True)
def _(P, compute_error, descriptor_hash, mo, n_words_slider, vec):
    if compute_error or P is None:
        mo.stop(True)

    _hash = descriptor_hash(vec, n_words=n_words_slider.value)
    _n_bits = n_words_slider.value * 16
    mo.md(f"""
    ### Descriptor hash

    | | |
    |---|---|
    | Hash | `{_hash}` |
    | Words | {n_words_slider.value} |
    | Bits | {_n_bits} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Radial basis functions

    The basis functions $R_n(q)$ are evaluated over $q \in [0, q_{max}]$
    using the current **n_max**, **sin(θ)/λ_max**, and **radial basis** settings.
    """)
    return


@app.cell(hide_code=True)
def _(basis_dd, n_max_slider, np, plt, stol_slider):
    from rinse_descriptor._radial_basis import evaluate_radial_basis as _eval_basis

    _q_max = 2.0 * stol_slider.value
    _n_max = n_max_slider.value
    _basis = basis_dd.value

    _q = np.linspace(0.0, _q_max, 500)
    _R = _eval_basis(_q, q_max=_q_max, n_max=_n_max, basis=_basis)  # (500, n_max)

    # Choose a colour palette that works for up to 32 curves
    _cmap_rb = plt.get_cmap("turbo")
    _colors = [_cmap_rb(i / max(_n_max - 1, 1)) for i in range(_n_max)]

    _fig_rb, _ax_rb = plt.subplots(figsize=(11, 3.5))
    for _n in range(_n_max):
        _ax_rb.plot(
            _q,
            _R[:, _n],
            color=_colors[_n],
            lw=1.2,
            alpha=0.85,
            label=f"n={_n}" if _n_max <= 8 else None,
        )
    _ax_rb.axhline(0, color="black", lw=0.5, ls="--")
    _ax_rb.set_xlim(0, _q_max)
    _ax_rb.set_xlabel(r"$q = |G|$  (Å$^{-1}$)", fontsize=10)
    _ax_rb.set_ylabel(r"$R_n(q)$", fontsize=10)
    _ax_rb.set_title(
        f"{_basis.capitalize()} radial basis  (n_max = {_n_max}, "
        f"$q_{{\\rm max}}$ = {_q_max:.2f} Å$^{{-1}}$)",
        fontsize=11,
    )
    if _n_max <= 8:
        _ax_rb.legend(fontsize=8, ncol=2, loc="upper right")
    else:
        # Colour-bar stand-in: annotate a few curve indices
        for _n in [0, _n_max // 4, _n_max // 2, 3 * _n_max // 4, _n_max - 1]:
            _ax_rb.annotate(
                f"n={_n}",
                xy=(_q_max * 0.97, float(_R[-1, _n])),
                fontsize=7,
                color=_colors[_n],
                va="center",
                ha="right",
            )
    plt.tight_layout()
    _fig_rb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    This example scales the unit cell and
    Cartesian positions isotropically from 90% to 110% (step 1%), computes
    one descriptor per scaled structure, and plots a descriptor heatmap.
    """)
    return


@app.cell(hide_code=True)
def _(
    Crystal,
    P,
    compute_error,
    crystal,
    dataclasses,
    descriptor,
    ff_dd,
    l2_normalisation_cb,
    log1p_compression_cb,
    mo,
    norm_dd,
    np,
    params,
    plt,
):
    if compute_error or P is None:
        mo.stop(True)
    _base = crystal

    _scales = np.round(np.arange(0.90, 1.1001, 0.01), 2)
    _params = dataclasses.replace(
        params, log1p=log1p_compression_cb.value, l2=l2_normalisation_cb.value, flatten=False
    )

    _scaled_descriptors = []
    for _scale in _scales:
        _scaled = Crystal(
            cell=_base.cell * _scale,
            positions=_base.positions * _scale,
            species=_base.species.copy(),
            occupancies=_base.occupancies.copy(),
            u_iso=_base.u_iso.copy(),
            u_aniso=_base.u_aniso.copy(),
            pbc=_base.pbc.copy(),
        )
        _scaled_descriptors.append(
            descriptor(
                _scaled,
                params=_params,
                structure_factor_type=norm_dd.value,
                form_factor_type=ff_dd.value,
            ).T.ravel()
        )

    _descriptor_matrix = np.stack(_scaled_descriptors, axis=0)

    from scipy.spatial import distance as _distance

    _distance_metric = _distance.correlation

    _distances = [
        float(_distance_metric(_descriptor, _descriptor_matrix[int(len(_scales) / 2)]))
        for _descriptor in _descriptor_matrix
    ]

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(12, 6))
    _image1 = _ax1.imshow(
        _descriptor_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        origin="lower",
    )
    _ax1.set_title(" Descriptor across isotropic cell scaling", fontsize=11)
    _ax1.set_xlabel("Descriptor feature index (flattened)", fontsize=9)
    _ax1.set_ylabel("Scale factor (distance)", fontsize=9)

    _yticks1 = list(range(0, len(_scales), 2))
    _ax1.set_yticks(_yticks1)
    _ylabels1 = [f"{_scales[i]:.2f}x ({_distances[i]:.5f})" for i in _yticks1]
    _ax1.set_yticklabels(_ylabels1, fontsize=8)

    _cbar1 = _fig.colorbar(_image1, ax=_ax1, shrink=0.9)
    _cbar1.set_label("Descriptor value", fontsize=9)

    _delta_matrix = np.stack(
        [row - _descriptor_matrix[int(len(_scales) / 2)] for row in _descriptor_matrix]
    )
    _minmax = max(abs(max(_delta_matrix.ravel())), abs(min(_delta_matrix.ravel())))
    _image2 = _ax2.imshow(
        _delta_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="seismic",
        origin="lower",
        vmin=-_minmax,
        vmax=_minmax,
    )
    _ax2.set_title("Delta", fontsize=11)
    _ax2.set_xlabel("Descriptor feature index (flattened)", fontsize=9)
    _ax2.set_ylabel("Scale factor", fontsize=9)

    _yticks2 = list(range(0, len(_scales), 2))
    _ax2.set_yticks(_yticks2)
    _ax2.set_yticklabels([f"{_scales[i]:.2f}x" for i in _yticks2], fontsize=8)

    _cbar2 = _fig.colorbar(_image2, ax=_ax2, shrink=0.9)
    _cbar2.set_label("Descriptor value", fontsize=9)

    plt.tight_layout()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    This example skews the unit cell according to
    [[1,x,0],[0,1,0],[0,0,1]] for x between -0.1 and 0.1, computes
    one descriptor per scaled structure, and plots a descriptor heatmap.
    """)
    return


@app.cell(hide_code=True)
def _(
    Crystal,
    P,
    compute_error,
    crystal,
    dataclasses,
    descriptor,
    ff_dd,
    l2_normalisation_cb,
    log1p_compression_cb,
    mo,
    norm_dd,
    np,
    params,
    plt,
):
    if compute_error or P is None:
        mo.stop(True)

    _base = crystal

    _scales = np.round(np.arange(-0.1, 0.1001, 0.01), 2)
    _params = dataclasses.replace(
        params, log1p=log1p_compression_cb.value, l2=l2_normalisation_cb.value, flatten=False
    )

    _scaled_descriptors = []
    for _scale in _scales:
        _scaled_cell = np.matmul(_base.cell, np.matrix([[1, _scale, 0], [0, 1, 0], [0, 0, 1]]))
        _scaled = Crystal(
            cell=_scaled_cell,
            positions=_base.positions.copy(),
            species=_base.species.copy(),
            occupancies=_base.occupancies.copy(),
            u_iso=_base.u_iso.copy(),
            u_aniso=_base.u_aniso.copy(),
            pbc=_base.pbc.copy(),
        )
        _scaled_descriptors.append(
            descriptor(
                _scaled,
                params=_params,
                structure_factor_type=norm_dd.value,
                form_factor_type=ff_dd.value,
            ).T.ravel()
        )

    _descriptor_matrix = np.stack(_scaled_descriptors, axis=0)

    from scipy.spatial import distance as _distance

    _distance_metric = _distance.correlation

    _distances = [
        float(_distance_metric(_descriptor, _descriptor_matrix[int(len(_scales) / 2)]))
        for _descriptor in _descriptor_matrix
    ]

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(12, 6))
    _image1 = _ax1.imshow(
        _descriptor_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        origin="lower",
    )
    _ax1.set_title(" Descriptor vs cell skewing (distance)", fontsize=11)
    _ax1.set_xlabel("Descriptor feature index (flattened)", fontsize=9)
    _ax1.set_ylabel("Scale factor", fontsize=9)

    _yticks1 = list(range(0, len(_scales), 2))
    _ax1.set_yticks(_yticks1)
    _ylabels1 = [f"{_scales[i]:.2f}x ({_distances[i]:.5f})" for i in _yticks1]
    _ax1.set_yticklabels(_ylabels1, fontsize=8)

    _cbar1 = _fig.colorbar(_image1, ax=_ax1, shrink=0.9)
    _cbar1.set_label("Descriptor value", fontsize=9)

    _delta_matrix = np.stack(
        [row - _descriptor_matrix[int(len(_scales) / 2)] for row in _descriptor_matrix]
    )
    _minmax = max(abs(max(_delta_matrix.ravel())), abs(min(_delta_matrix.ravel())))
    _image2 = _ax2.imshow(
        _delta_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="seismic",
        origin="lower",
        vmin=-_minmax,
        vmax=_minmax,
    )
    _ax2.set_title("Delta", fontsize=11)
    _ax2.set_xlabel("Descriptor feature index (flattened)", fontsize=9)
    _ax2.set_ylabel("Scale factor", fontsize=9)

    _yticks2 = list(range(0, len(_scales), 2))
    _ax2.set_yticks(_yticks2)
    _ax2.set_yticklabels([f"{_scales[i]:.2f}x" for i in _yticks2], fontsize=8)

    _cbar2 = _fig.colorbar(_image2, ax=_ax2, shrink=0.9)
    _cbar2.set_label("Descriptor value", fontsize=9)

    plt.tight_layout()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    This example changes the occupancy of the first atom in the cif from 0 to 1
    in 0.1 steps, computes one descriptor per structure, and plots a descriptor heatmap.
    """)
    return


@app.cell(hide_code=True)
def _(
    Crystal,
    P,
    compute_error,
    crystal,
    dataclasses,
    descriptor,
    ff_dd,
    l2_normalisation_cb,
    log1p_compression_cb,
    mo,
    norm_dd,
    np,
    params,
    plt,
):
    if compute_error or P is None:
        mo.stop(True)
    _base = crystal

    _scales = np.round(np.arange(0, 1.001, 0.1), 2)
    _params = dataclasses.replace(
        params, log1p=log1p_compression_cb.value, l2=l2_normalisation_cb.value, flatten=False
    )

    _scaled_descriptors = []
    for _scale in _scales:
        _occupancies = _base.occupancies.copy()
        _occupancies[0] = _scale
        _scaled = Crystal(
            cell=_base.cell.copy(),
            positions=_base.positions.copy(),
            species=_base.species.copy(),
            occupancies=_occupancies,
            u_iso=_base.u_iso.copy(),
            u_aniso=_base.u_aniso.copy(),
            pbc=_base.pbc.copy(),
        )
        _scaled_descriptors.append(
            descriptor(
                _scaled,
                params=_params,
                structure_factor_type=norm_dd.value,
                form_factor_type=ff_dd.value,
            ).T.ravel()
        )

    _descriptor_matrix = np.stack(_scaled_descriptors, axis=0)

    from scipy.spatial import distance as _distance

    _distance_metric = _distance.correlation

    _distances = [
        float(_distance_metric(_descriptor, _descriptor_matrix[0]))
        for _descriptor in _descriptor_matrix
    ]

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(12, 6))
    _image1 = _ax1.imshow(
        _descriptor_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        origin="lower",
    )
    _ax1.set_title("Descriptor vs first atom occupancy", fontsize=11)
    _ax1.set_xlabel("Descriptor feature index (flattened)", fontsize=9)
    _ax1.set_ylabel("First atom occupancy (distance)", fontsize=9)

    _yticks1 = list(range(0, len(_scales), 2))
    _ax1.set_yticks(_yticks1)
    _ylabels1 = [f"{_scales[i]:.2f}x ({_distances[i]:.5f})" for i in _yticks1]
    _ax1.set_yticklabels(_ylabels1, fontsize=8)

    _cbar1 = _fig.colorbar(_image1, ax=_ax1, shrink=0.9)
    _cbar1.set_label("Descriptor value", fontsize=9)

    _delta_matrix = np.stack([row - _descriptor_matrix[0] for row in _descriptor_matrix])
    _minmax = max(abs(max(_delta_matrix.ravel())), abs(min(_delta_matrix.ravel())))
    _image2 = _ax2.imshow(
        _delta_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="seismic",
        origin="lower",
        vmin=-_minmax,
        vmax=_minmax,
    )
    _ax2.set_title("Delta", fontsize=11)
    _ax2.set_xlabel("Descriptor feature index (flattened)", fontsize=9)
    _ax2.set_ylabel("Scale factor", fontsize=9)

    _yticks2 = list(range(0, len(_scales), 2))
    _ax2.set_yticks(_yticks2)
    _ax2.set_yticklabels([f"{_scales[i]:.2f}x" for i in _yticks2], fontsize=8)

    _cbar2 = _fig.colorbar(_image2, ax=_ax2, shrink=0.9)
    _cbar2.set_label("Descriptor value", fontsize=9)

    plt.tight_layout()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    Input a second structure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cif_upload2 = mo.ui.file(
        filetypes=[".cif"],
        label="Upload second CIF file",
    )
    cif_upload2
    return (cif_upload2,)


@app.cell(hide_code=True)
def _(
    Crystal,
    RinseParams,
    basis_dd,
    cif_upload2,
    compute_power_spectrum,
    compute_structure_factors,
    ff_dd,
    include_odd_l_cb,
    l2_normalisation_cb,
    l_max_slider,
    l_min_slider,
    log1p_compression_cb,
    n_max_slider,
    norm_dd,
    os,
    power_spectrum_to_vector,
    stol_slider,
    tempfile,
):
    _crystal = None
    _struct_label = ""
    _error = None

    try:
        if cif_upload2.value:
            _f = cif_upload2.value[0]
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as _tmp:
                _tmp.write(_f.contents)
                _tmp_path = _tmp.name
            try:
                _crystal = Crystal.from_cif(_tmp_path)
                _struct_label = _f.name
            finally:
                os.unlink(_tmp_path)
    except Exception as _e:
        _error = str(_e)

    _params = None
    _refls = None
    _P = None
    _vec = None

    if _crystal is not None and _error is None:
        try:
            _params = RinseParams(
                n_max=n_max_slider.value,
                l_max=l_max_slider.value,
                l_min=l_min_slider.value,
                include_odd_l=include_odd_l_cb.value,
                sin_theta_over_lambda_max=stol_slider.value,
                radial_basis=basis_dd.value,
                log1p=log1p_compression_cb.value,
                l2=l2_normalisation_cb.value,
                flatten=False,
            )
            _refls = compute_structure_factors(
                _crystal,
                sin_theta_over_lambda_max=stol_slider.value,
                form_factor_type=ff_dd.value,
                structure_factor_type=norm_dd.value,
                debug=True,
            )
            _P = compute_power_spectrum(
                _refls,
                params=_params,
                debug=True,
            )
            _vec = power_spectrum_to_vector(_P)
        except Exception as _e:
            _error = str(_e)

    P2 = _P
    vec2 = _vec
    compute_error2 = _error
    return P2, compute_error2, vec2


@app.cell(hide_code=True)
def _(P, P2, compute_error, compute_error2, mo, plt, vec, vec2):
    if (compute_error or compute_error2) or (P is None or P2 is None):
        mo.stop(True)

    from scipy.spatial import distance as _distance

    _vec = vec
    _vec2 = vec2
    _fig2, _ax = plt.subplots(figsize=(12, 2.5))
    _ax.fill_between(range(len(_vec)), _vec, alpha=0.5, color="#66CCEE")
    _ax.fill_between(range(len(_vec2)), _vec2, alpha=0.5, color="#EECC66")
    _ax.set_xlabel("Descriptor index  (row-major: i·l_max + k)", fontsize=9)
    _ax.set_ylabel("p_{nk}", fontsize=9)
    _corr = _distance.correlation(vec, vec2)
    _ax.set_title(
        f"Descriptor vector2 ({len(_vec)} elements, correlation distance = {_corr:.5f})",
        fontsize=10,
    )
    _ax.set_xlim(0, len(_vec) - 1)
    # plt.yscale('log')
    plt.tight_layout()
    _fig2
    return


if __name__ == "__main__":
    app.run()
