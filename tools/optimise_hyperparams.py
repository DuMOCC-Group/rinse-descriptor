"""Optimise RinseParams hyperparameters using Bayesian optimisation (Optuna).

Additional dependencies (not in project – install before running):
    uv pip install optuna mp-api pymatgen

The objective is evaluated with *k*-fold cross-validation (default 5 folds) for
robustness.  It is a single objective posed as a retrieval / verification task:

  * **discrimination (AUROC)** – maximise the area under the ROC curve of
    same-vs-different structure pairs, scored on one cosine-similarity scale.
    *Positive* pairs are redeterminations sharing a *cluster* group (same refcode
    family, similar cell); *negative* pairs come from different structures
    (``distinct`` diversity structures and cross-group cluster pairs).  Because
    only the *ranking* of same above different matters, the metric is invariant
    to descriptor scale/dimensionality and cannot be gamed by coarsening the
    descriptor (which merely shrinks the shared common mode).  ``d'`` and the
    mean within/between similarities are reported as diagnostics.

    Each CV fold pairs *all* cluster structures (positives) with one fold of the
    ``distinct`` background structures (extra negatives), so positives are
    preserved while the negative background is cross-validated.

  * **legacy separation** (``--no-clusters``) – minimise the mean pairwise
    Pearson correlation of the ``distinct`` structures.  Retained for sets with
    no redetermination groups; note this can be gamed by degenerately coarse
    descriptors and is not recommended.

Data sources (in priority order):
  * ``--full-cache`` (default ``training_set_full.pkl``): group-aware records
    produced by ``build_training_set.py`` (supplies the positive pairs).
  * ``--cache`` (default ``mp_opt_structures.pkl``): a flat ``(id, cif)`` list;
    if absent, N pseudo-random Materials Project structures are fetched.

When ``--test-cache`` (default ``training_set_test.pkl``) is present, the
reported params are additionally scored on this unseen held-out split.

Usage:
    uv run tools/build_training_set.py --api-key $MP_API_KEY
    uv run tools/optimise_hyperparams.py            # uses training_set_full.pkl if present
    uv run tools/optimise_hyperparams.py --n-trials 200 --n-folds 5
    uv run tools/optimise_hyperparams.py --jobs 8   # parallel descriptor computation
    uv run tools/optimise_hyperparams.py \
        --eval-params n_max=16,l_min=4,l_max=20,sin_theta_over_lambda_max=0.5
        # score one fixed parameter set and add it to the study
        # (key=value avoids shell quoting; a JSON object is also accepted)
    uv run tools/optimise_hyperparams.py --cache-only   # fetch and cache only
    uv run tools/optimise_hyperparams.py --no-fetch     # skip fetch, use cache

API key:  pass --api-key or set MP_API_KEY / PMG_MAPI_KEY environment variable.

Outputs:
    mp_opt_structures.pkl  – cached (material_id, cif_string) pairs
    optuna_rinse.db        – SQLite study database (resumable across runs)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import io
import json
import os
import pickle
import random
import sys
import tempfile
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import optuna
except ImportError:
    sys.exit(
        "optuna not found.\n"
        "Install the required tools with:\n"
        "    uv pip install optuna mp-api pymatgen"
    )

from libtbx.utils import Sorry  # type: ignore[import-untyped]
from rinse_descriptor import (
    RinseParams,
    compute_power_spectrum,
    compute_structure_factors,
    load_cif,
    load_res,
    power_spectrum_to_vector,
)
from tqdm import tqdm

if TYPE_CHECKING:
    import optuna.trial

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECOVERABLE = (Sorry, ValueError, RuntimeError, OSError, NameError, AttributeError, KeyError)

_DEFAULT_CACHE = Path("mp_opt_structures.pkl")
_DEFAULT_FULL_CACHE = Path("training_set_full.pkl")
_DEFAULT_TEST_CACHE = Path("training_set_test.pkl")
_DEFAULT_SEED = 42
_DEFAULT_N_STRUCTURES = 1000
_DEFAULT_N_TRIALS = 100
_DEFAULT_N_FOLDS = 5
_DEFAULT_STUDY_DB = "sqlite:///optuna_rinse.db"
_DEFAULT_STUDY_NAME = "rinse_hyperparams"


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------


def _build_params(trial: optuna.trial.Trial) -> RinseParams | None:
    """Sample a RinseParams from an Optuna trial, returning None if invalid."""
    n_max = trial.suggest_int("n_max", 4, 32)
    l_min = trial.suggest_int("l_min", 4, 4, step=2)
    # l_max is parameterised as l_min + 2 * n_l_levels to guarantee l_max > l_min
    # and to keep both even (required when include_odd_l=False).
    n_l_levels = trial.suggest_int("n_l_levels", 1, 16)
    l_max = l_min + 2 * n_l_levels
    trial.set_user_attr("l_max", l_max)

    # Resolution range for the descriptor's reciprocal-space cutoff.
    sin_theta = trial.suggest_float("sin_theta_over_lambda_max", 0.2, 0.6, step=0.05)
    radial_basis = trial.suggest_categorical(
        "radial_basis",
        [
            #"chebyshev",
            #"bessel",
            #"smooth_shells_cw",
            "smooth_shells_nl"
        ],
    )
    intensity_norm = trial.suggest_categorical(
        "intensity_normalisation",
        [
            "none",
            #"double_exponential",
            #"empirical"
        ],
    )
    intensity_falloff = trial.suggest_categorical(
        "intensity_falloff",
        [
            "none",
            #"debye_waller"
        ],
    )
    # u_iso is only meaningful when Debye-Waller falloff is active, but Optuna
    # needs a consistent parameter space across trials.  We sample it always and
    # pass it through; RinseParams ignores it when falloff="none".
    u_iso = trial.suggest_float("intensity_falloff_u_iso", 0.0, 0.0, step=0.02)
    log1p = trial.suggest_categorical("log1p", [
        False,
        #True
        ])

    try:
        return RinseParams(
            n_max=n_max,
            l_max=l_max,
            l_min=l_min,
            sin_theta_over_lambda_max=sin_theta,
            radial_basis=radial_basis,  # type: ignore[arg-type]
            intensity_normalisation=intensity_norm,  # type: ignore[arg-type]
            intensity_falloff=intensity_falloff,  # type: ignore[arg-type]
            intensity_falloff_u_iso=u_iso,
            log1p=log1p,
            l2=True,  # always L2-normalise so Pearson is comparable across trials
            flatten=True,
            include_odd_l=False,
            monopole_normalisation=True,
        )
    except ValueError as exc:
        print(f"  [skip] invalid params: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------


def _compute_descriptor(xrs: Any, params: RinseParams) -> np.ndarray | None:
    """Compute one descriptor vector, returning None if the structure fails."""
    try:
        reflections = compute_structure_factors(
            xrs,
            sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
            intensity_normalisation=params.intensity_normalisation,
            intensity_normalisation_n_bins=params.intensity_normalisation_n_bins,
            intensity_normalisation_min_bin_size=params.intensity_normalisation_min_bin_size,
            intensity_falloff=params.intensity_falloff,
            intensity_falloff_u_iso=params.intensity_falloff_u_iso,
            use_reported_adps=params.use_reported_adps,
        )
        P = compute_power_spectrum(reflections, params=params)
        v = power_spectrum_to_vector(P)
    except _RECOVERABLE:
        return None
    if np.isfinite(v).all() and np.linalg.norm(v) > 0.0:
        return v
    return None


def _compute_descriptors(
    xrs_list: list[Any],
    params: RinseParams,
    ids: list[str] | None = None,
    executor: cf.ProcessPoolExecutor | None = None,
    n_workers: int = 1,
) -> list[np.ndarray | None]:
    """Compute descriptors for all structures with a progress bar.

    When *executor* is provided (and ``n_workers > 1``), the work is dispatched
    to a persistent process pool whose workers hold their own copy of the
    structures (loaded once via the pool initialiser); only the params and index
    ranges cross the process boundary.  Otherwise the loop runs serially.
    """
    n = len(xrs_list)
    if executor is None or n_workers <= 1:
        vecs: list[np.ndarray | None] = []
        for xrs in tqdm(
            xrs_list,
            total=n,
            desc="  descriptors",
            unit="struct",
            leave=False,
            file=sys.stderr,
        ):
            vecs.append(_compute_descriptor(xrs, params))
        return vecs

    # Parallel path: split into chunks so the progress bar advances smoothly.
    chunk = max(1, min(64, -(-n // (n_workers * 4))))
    tasks = [(params, s, min(s + chunk, n)) for s in range(0, n, chunk)]
    out: list[np.ndarray | None] = [None] * n
    with tqdm(
        total=n,
        desc="  descriptors",
        unit="struct",
        leave=False,
        file=sys.stderr,
    ) as bar:
        for start, chunk_vecs in executor.map(_worker_compute_chunk, tasks):
            out[start : start + len(chunk_vecs)] = chunk_vecs
            bar.update(len(chunk_vecs))
    return out


# ---------------------------------------------------------------------------
# Process-pool workers (used when --jobs > 1)
# ---------------------------------------------------------------------------

_WORKER_XRS: list[Any] | None = None


def _init_worker(records_path: str) -> None:
    """Pool initialiser: load the structures once into a per-process global."""
    global _WORKER_XRS
    with open(records_path, "rb") as fh:
        records = pickle.load(fh)
    # Suppress the per-load status prints so only the parent reports progress.
    with contextlib.redirect_stderr(io.StringIO()):
        xrs_list, _, _, _ = _load_structures(records)
    _WORKER_XRS = xrs_list


def _worker_compute_chunk(
    task: tuple[RinseParams, int, int],
) -> tuple[int, list[np.ndarray | None]]:
    """Compute descriptors for ``xrs_list[start:stop]`` in a worker process."""
    params, start, stop = task
    assert _WORKER_XRS is not None, "worker not initialised"
    return start, [_compute_descriptor(_WORKER_XRS[i], params) for i in range(start, stop)]



def _mean_pearson(X: np.ndarray) -> float:
    """Mean pairwise Pearson correlation for the rows of *X*.

    Uses the upper triangle only (excluding self-correlations of 1.0).
    For L2-normalised, zero-mean descriptors this reduces to the mean
    off-diagonal cosine similarity.
    """
    n = X.shape[0]
    if n < 2:
        return 0.0
    C = np.corrcoef(X)  # (n, n)
    idx = np.triu_indices(n, k=1)
    return float(np.mean(C[idx]))


# ---------------------------------------------------------------------------
# Discrimination (same-vs-different) metrics
# ---------------------------------------------------------------------------


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix for *X* (rows are descriptors)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    U = X / norms
    return U @ U.T


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U rank statistic.

    ``auroc`` is the probability that a randomly chosen *positive* score
    (same-structure pair similarity) exceeds a randomly chosen *negative* score
    (different-structure pair similarity).  0.5 = chance, 1.0 = perfect ranking.
    """
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    rank_pos = float(ranks[:n_pos].sum())
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _discrimination_metrics(
    X: np.ndarray, group_ids: np.ndarray
) -> dict[str, float] | None:
    """Same-vs-different discrimination metrics on one cosine-similarity scale.

    Positive pairs share a ``group_id`` (redeterminations of one structure);
    negative pairs come from different groups.  Because both are scored on the
    *same* cosine scale, the metrics reward a descriptor that ranks same-structure
    pairs above different-structure pairs and cannot be gamed by shrinking the
    descriptor's shared common mode (which the old mean-correlation objective
    rewarded).  Returns None when there are no positive pairs.

    Keys: ``auroc`` (0.5 chance ... 1.0 perfect), ``dprime``
    ``(mu_within - mu_between) / pooled_std``, ``mu_within``, ``mu_between``,
    ``n_pos``, ``n_neg``.
    """
    n = X.shape[0]
    if n < 2:
        return None
    S = _cosine_similarity_matrix(X)
    iu, ju = np.triu_indices(n, k=1)
    sims = S[iu, ju]
    same = group_ids[iu] == group_ids[ju]
    pos = sims[same]
    neg = sims[~same]
    if pos.size == 0 or neg.size == 0:
        return None
    mu_w = float(pos.mean())
    mu_b = float(neg.mean())
    pooled = float(np.sqrt(0.5 * (pos.var() + neg.var())))
    dprime = (mu_w - mu_b) / pooled if pooled > 0.0 else 0.0
    return {
        "auroc": _auroc(pos, neg),
        "dprime": dprime,
        "mu_within": mu_w,
        "mu_between": mu_b,
        "n_pos": float(pos.size),
        "n_neg": float(neg.size),
    }


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------


def _kfold_indices(positions: list[int], k: int, seed: int) -> list[np.ndarray]:
    """Partition *positions* into *k* shuffled, balanced folds.

    Returns a list of index arrays (interleaved split for balance).  With
    ``k <= 1`` a single fold containing all positions is returned.
    """
    idx = list(positions)
    random.Random(seed).shuffle(idx)
    if k <= 1:
        return [np.array(idx, dtype=int)]
    folds = [np.array(idx[i::k], dtype=int) for i in range(k)]
    return [fold for fold in folds if fold.size > 0]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

# Minimum number of valid descriptors required to evaluate a trial.
_MIN_VALID = 20

# Minimum valid descriptors within a single fold for it to count.
_MIN_FOLD_VALID = 2


def _objective(
    trial: optuna.trial.Trial,
    xrs_list: list[Any],
    groups: list[str],
    sep_folds: list[np.ndarray],
    cluster_positions: list[int],
    use_auroc: bool,
    ids: list[str] | None = None,
    executor: cf.ProcessPoolExecutor | None = None,
    n_workers: int = 1,
) -> float:
    """Evaluate a trial with *k*-fold cross-validation for robustness.

    Two objective modes, both single-objective:

    * ``use_auroc=True`` (default when cluster/redetermination groups exist):
      maximise the same-vs-different **AUROC**.  Each fold combines *all* cluster
      structures (which supply the positive, same-structure pairs) with one fold
      of the ``distinct`` background structures (which supply extra negatives), so
      positives are preserved while the negative background is cross-validated.
      This measures fingerprint quality directly and is invariant to descriptor
      scale/dimensionality, so it cannot be gamed by coarsening the descriptor.

    * ``use_auroc=False`` (legacy, ``--no-clusters`` or no groups present):
      minimise the mean pairwise Pearson correlation of the ``distinct``
      structures.  Kept for backwards compatibility; note this rewards shrinking
      the shared common mode and can favour degenerately coarse descriptors.
    """
    params = _build_params(trial)
    if params is None:
        raise optuna.TrialPruned()

    vecs = _compute_descriptors(xrs_list, params, ids, executor, n_workers)

    if not use_auroc:
        return _objective_separation(trial, vecs, sep_folds)

    # ---- Discrimination (AUROC) objective, cross-validated over folds ----
    fold_auroc: list[float] = []
    fold_dprime: list[float] = []
    mu_within: list[float] = []
    mu_between: list[float] = []
    n_valid_total = 0
    for fold in sep_folds:
        idx = list(cluster_positions) + [int(i) for i in fold]
        sub_vecs = [vecs[i] for i in idx if vecs[i] is not None]
        sub_groups = [groups[i] for i in idx if vecs[i] is not None]
        n_valid_total += len(sub_vecs)
        if len(sub_vecs) < _MIN_FOLD_VALID:
            continue
        m = _discrimination_metrics(
            np.array(sub_vecs, dtype=np.float64),
            np.asarray(sub_groups, dtype=object),
        )
        if m is None:
            continue
        fold_auroc.append(m["auroc"])
        fold_dprime.append(m["dprime"])
        mu_within.append(m["mu_within"])
        mu_between.append(m["mu_between"])

    if not fold_auroc:
        print(
            "  [prune] no fold produced positive (same-structure) pairs",
            file=sys.stderr,
        )
        raise optuna.TrialPruned()

    auroc_cv = float(np.mean(fold_auroc))
    auroc_std = float(np.std(fold_auroc))
    dprime_cv = float(np.mean(fold_dprime))
    trial.set_user_attr("auroc_cv_mean", auroc_cv)
    trial.set_user_attr("auroc_cv_std", auroc_std)
    trial.set_user_attr("dprime_cv_mean", dprime_cv)
    trial.set_user_attr("mu_within", float(np.mean(mu_within)))
    trial.set_user_attr("mu_between", float(np.mean(mu_between)))
    trial.set_user_attr("n_folds_used", len(fold_auroc))

    print(
        f"  trial {trial.number}: auroc={auroc_cv:.6f}±{auroc_std:.6f}  "
        f"d'={dprime_cv:.4f}  "
        f"(mu_in={np.mean(mu_within):.4f}, mu_out={np.mean(mu_between):.4f}, "
        f"folds={len(fold_auroc)})",
        file=sys.stderr,
    )
    return auroc_cv


def _objective_separation(
    trial: optuna.trial.Trial,
    vecs: list[np.ndarray | None],
    sep_folds: list[np.ndarray],
) -> float:
    """Legacy separation objective: minimise mean pairwise Pearson correlation.

    Retained for the ``--no-clusters`` fallback.  See :func:`_objective` for the
    caveats about why this metric can be gamed by degenerately coarse
    descriptors.
    """
    fold_seps: list[float] = []
    n_valid_distinct = 0
    for fold in sep_folds:
        fold_vecs = [vecs[i] for i in fold if vecs[i] is not None]
        n_valid_distinct += len(fold_vecs)
        if len(fold_vecs) >= _MIN_FOLD_VALID:
            fold_seps.append(_mean_pearson(np.array(fold_vecs, dtype=np.float64)))

    if n_valid_distinct < _MIN_VALID or not fold_seps:
        print(
            f"  [prune] only {n_valid_distinct} valid separation descriptors "
            f"across {len(fold_seps)} usable folds (need >= {_MIN_VALID})",
            file=sys.stderr,
        )
        raise optuna.TrialPruned()

    sep_cv = float(np.mean(fold_seps))
    sep_std = float(np.std(fold_seps))
    trial.set_user_attr("sep_cv_mean", sep_cv)
    trial.set_user_attr("sep_cv_std", sep_std)
    trial.set_user_attr("n_folds_used", len(fold_seps))
    print(
        f"  trial {trial.number}: sep_cv={sep_cv:.6f}±{sep_std:.6f}  "
        f"(n_sep={n_valid_distinct}, folds={len(fold_seps)})",
        file=sys.stderr,
    )
    return sep_cv


# ---------------------------------------------------------------------------
# MP structure fetching helpers
# ---------------------------------------------------------------------------


def _doc_get(doc: Any, key: str) -> Any:
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)


def _dataset_columns(docs: Any, fields: list[str]) -> dict[str, list[Any]]:
    """Extract columns from MP API responses without row-wise iteration."""
    if hasattr(docs, "delta_table"):
        table = docs.delta_table
        if hasattr(table, "to_pyarrow_table"):
            return table.to_pyarrow_table(columns=fields).to_pydict()
    if hasattr(docs, "pyarrow_dataset"):
        table = docs.pyarrow_dataset.to_table(columns=fields)
        return table.to_pydict()
    out: dict[str, list[Any]] = {f: [] for f in fields}
    for doc in docs:
        for f in fields:
            out[f].append(_doc_get(doc, f))
    return out


def _chunks(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_and_cache(
    api_key: str,
    n: int,
    seed: int,
    cache_path: Path,
    fetch_batch_size: int = 200,
    id_chunk_size: int = 5000,
) -> list[tuple[str, str]]:
    """Fetch *n* pseudo-random MP structures and cache as (material_id, cif_string) pairs."""
    from mp_api.client import MPRester  # pyright: ignore[reportMissingImports]
    from pymatgen.io.cif import CifWriter  # pyright: ignore[reportMissingImports]

    print("Fetching material IDs from Materials Project...", file=sys.stderr)
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            all_fields=False,
            fields=["material_id"],
            chunk_size=id_chunk_size,
        )
        columns = _dataset_columns(docs, ["material_id"])
    all_ids: list[str] = [
        str(mid) for mid in columns.get("material_id", []) if mid is not None
    ]
    all_ids.sort()
    print(f"Found {len(all_ids)} total material IDs", file=sys.stderr)

    rng = random.Random(seed)
    selected = rng.sample(all_ids, min(n, len(all_ids)))
    selected.sort()
    print(
        f"Selected {len(selected)} pseudo-random IDs (seed={seed})",
        file=sys.stderr,
    )

    pairs: list[tuple[str, str]] = []
    skipped = 0

    for batch in _chunks(selected, fetch_batch_size):
        start = len(pairs) + skipped + 1
        end = min(start + len(batch) - 1, n)
        print(
            f"  Fetching structures {start}-{end} / {len(selected)}...",
            file=sys.stderr,
        )
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                material_ids=batch,
                fields=["material_id", "structure"],
                chunk_size=max(len(batch), 1),
            )
            columns = _dataset_columns(docs, ["material_id", "structure"])

        for mid, structure in zip(
            columns.get("material_id", []),
            columns.get("structure", []),
            strict=False,
        ):
            if mid is None or structure is None:
                skipped += 1
                continue
            try:
                cif_text = str(CifWriter(structure))
                pairs.append((str(mid), cif_text))
            except Exception as exc:
                print(f"  Skipping {mid}: {exc}", file=sys.stderr)
                skipped += 1

    print(
        f"Fetched {len(pairs)} CIF strings ({skipped} skipped)",
        file=sys.stderr,
    )
    with open(cache_path, "wb") as fh:
        pickle.dump(pairs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved cache to {cache_path}", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Load xrs objects (group-aware) from cached CIF strings
# ---------------------------------------------------------------------------


def _pairs_to_records(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Wrap flat (id, cif) pairs as distinct-role CIF records (own group)."""
    return [
        {"id": mid, "text": cif, "format": "cif", "group": mid, "role": "distinct"}
        for mid, cif in pairs
    ]


def _load_structures(
    records: list[dict[str, Any]],
) -> tuple[list[Any], list[str], list[str], list[str]]:
    """Convert structure records to cctbx xrs objects, keeping group/role aligned.

    Each record's ``format`` field selects the parser (``"cif"`` -> load_cif,
    ``"res"`` -> load_res).  Legacy records that carry a ``cif`` key instead of
    ``text`` are still supported.  Returns ``(xrs_list, groups, roles, ids)`` with
    failures skipped.
    """
    xrs_list: list[Any] = []
    groups: list[str] = []
    roles: list[str] = []
    ids: list[str] = []
    skipped = 0
    for record in records:
        entry_id = record.get("id", "<unknown>")
        text = record.get("text", record.get("cif"))
        fmt = record.get("format", "cif")
        loader = load_res if fmt == "res" else load_cif
        try:
            xrs = loader(StringIO(text))
            if xrs.scatterers().size() > 0:
                xrs_list.append(xrs)
                groups.append(str(record.get("group", entry_id)))
                roles.append(str(record.get("role", "distinct")))
                ids.append(str(entry_id))
            else:
                skipped += 1
        except _RECOVERABLE as exc:
            print(f"  Skipping {entry_id} during xrs load: {exc}", file=sys.stderr)
            skipped += 1
    print(
        f"Loaded {len(xrs_list)} xrs structures ({skipped} skipped)",
        file=sys.stderr,
    )
    return xrs_list, groups, roles, ids


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _resolve_api_key(cli_key: str | None) -> str:
    key = cli_key or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise ValueError(
            "Materials Project API key missing. "
            "Pass --api-key or set the MP_API_KEY environment variable."
        )
    return key


def _print_trial_params(trial: optuna.trial.Trial) -> None:
    """Print the RinseParams encoded by a trial."""
    l_max = trial.user_attrs.get("l_max")
    if l_max is None:
        l_max = trial.params["l_min"] + 2 * trial.params["n_l_levels"]
    print(f"  n_max                     = {trial.params['n_max']}")
    print(f"  l_min                     = {trial.params['l_min']}")
    print(f"  l_max (derived)           = {l_max}")
    print(f"  n_l_levels (derived)      = {trial.params['n_l_levels']}")
    print(f"  sin_theta_over_lambda_max = {trial.params['sin_theta_over_lambda_max']:.4f}")
    print(f"  radial_basis              = {trial.params['radial_basis']!r}")
    print(f"  intensity_normalisation   = {trial.params['intensity_normalisation']!r}")
    print(f"  intensity_falloff         = {trial.params['intensity_falloff']!r}")
    print(f"  intensity_falloff_u_iso   = {trial.params['intensity_falloff_u_iso']:.4f}")
    print(f"  log1p                     = {trial.params['log1p']}")
    print("  l2                        = True")


def _params_from_dict(p: dict[str, Any]) -> RinseParams | None:
    """Rebuild a RinseParams from a stored Optuna ``trial.params`` mapping."""
    try:
        l_min = p["l_min"]
        l_max = l_min + 2 * p["n_l_levels"]
        return RinseParams(
            n_max=p["n_max"],
            l_max=l_max,
            l_min=l_min,
            sin_theta_over_lambda_max=p["sin_theta_over_lambda_max"],
            radial_basis=p["radial_basis"],
            intensity_normalisation=p["intensity_normalisation"],
            intensity_falloff=p["intensity_falloff"],
            intensity_falloff_u_iso=p["intensity_falloff_u_iso"],
            log1p=p["log1p"],
            l2=True,
            flatten=True,
            include_odd_l=False,
        )
    except (KeyError, ValueError) as exc:
        print(f"  [test] cannot rebuild params: {exc}", file=sys.stderr)
        return None


def _coerce_scalar(value: str) -> Any:
    """Coerce a ``key=value`` string token to bool/int/float, else leave as str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_param_spec(text: str) -> dict[str, Any]:
    """Parse a parameter spec from JSON or shell-friendly ``key=value`` pairs.

    A leading ``{`` selects JSON (``{"n_max": 16, ...}``).  Otherwise a
    comma-separated ``key=value`` form is accepted (e.g.
    ``n_max=16,l_min=4,l_max=20``), which avoids having to escape double quotes
    on shells such as PowerShell that strip them before the program sees them.
    Scalar values are coerced to int/float/bool where possible.
    """
    text = text.strip()
    if text.startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("JSON spec must be an object of parameter values")
        return obj
    spec: dict[str, Any] = {}
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"expected 'key=value' but got {item!r} "
                "(separate parameters with commas)"
            )
        key, _, raw = item.partition("=")
        spec[key.strip()] = _coerce_scalar(raw.strip())
    return spec


def _spec_to_trial_params(spec: dict[str, Any]) -> dict[str, Any]:
    """Translate a user RinseParams-style spec into an Optuna trial-param dict.

    Accepts the same field names as :class:`RinseParams` (``l_max`` is converted
    to the search-space parameter ``n_l_levels``; ``n_l_levels`` may also be
    given directly).  Any omitted field falls back to the RinseParams default so
    a partial spec is enough.  The returned mapping is suitable for
    ``study.enqueue_trial`` and matches the names sampled in ``_build_params``.
    """
    n_max = int(spec.get("n_max", 16))
    l_min = int(spec.get("l_min", 4))
    if "n_l_levels" in spec:
        n_l_levels = int(spec["n_l_levels"])
    else:
        l_max = int(spec.get("l_max", l_min + 16))
        diff = l_max - l_min
        if diff <= 0 or diff % 2 != 0:
            raise ValueError(
                f"l_max ({l_max}) must exceed l_min ({l_min}) by a positive even "
                "amount (l_max = l_min + 2 * n_l_levels)"
            )
        n_l_levels = diff // 2
    return {
        "n_max": n_max,
        "l_min": l_min,
        "n_l_levels": n_l_levels,
        "sin_theta_over_lambda_max": float(spec.get("sin_theta_over_lambda_max", 0.6)),
        "radial_basis": spec.get("radial_basis", "smooth_shells_nl"),
        "intensity_normalisation": spec.get(
            "intensity_normalisation", "none"
        ),
        "intensity_falloff": spec.get("intensity_falloff", "none"),
        "intensity_falloff_u_iso": float(spec.get("intensity_falloff_u_iso", 0.05)),
        "log1p": bool(spec.get("log1p", False)),
    }


def _evaluate_on_set(
    params: RinseParams,
    xrs_list: list[Any],
    groups: list[str],
    roles: list[str],
    ids: list[str] | None = None,
) -> dict[str, float | None]:
    """Evaluate discrimination metrics on a held-out set.

    No cross-validation folds are used — the whole set is scored once.  Returns a
    mapping with ``auroc`` / ``dprime`` / ``mu_within`` / ``mu_between`` (the
    same-vs-different discrimination metrics; higher ``auroc``/``dprime`` is
    better) and the legacy ``sep`` (mean Pearson over the ``distinct``
    structures, lower better) as a diagnostic.  Values are None when the set has
    too few members of the relevant kind.
    """
    vecs = _compute_descriptors(xrs_list, params, ids)

    valid = [(v, g) for v, g in zip(vecs, groups) if v is not None]
    disc: dict[str, float] | None = None
    if len(valid) >= 2:
        X = np.array([v for v, _ in valid], dtype=np.float64)
        gid = np.asarray([g for _, g in valid], dtype=object)
        disc = _discrimination_metrics(X, gid)

    sep_vecs = [
        v
        for v, role in zip(vecs, roles)
        if role == "distinct" and v is not None
    ]
    sep = (
        _mean_pearson(np.array(sep_vecs, dtype=np.float64))
        if len(sep_vecs) >= 2
        else None
    )

    return {
        "auroc": disc["auroc"] if disc else None,
        "dprime": disc["dprime"] if disc else None,
        "mu_within": disc["mu_within"] if disc else None,
        "mu_between": disc["mu_between"] if disc else None,
        "sep": sep,
    }


def _report_test_set(
    trials: list[optuna.trial.FrozenTrial],
    xrs_list: list[Any],
    groups: list[str],
    roles: list[str],
    ids: list[str] | None = None,
) -> None:
    """Print held-out test-set metrics for each reported trial's parameters."""
    print(
        "\nHeld-out test-set evaluation "
        "(higher auroc/d' better; lower sep better):"
    )
    for trial in trials:
        params = _params_from_dict(trial.params)
        if params is None:
            continue
        m = _evaluate_on_set(params, xrs_list, groups, roles, ids)

        def _fmt(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.6f}"

        print(
            f"- trial {trial.number}: auroc={_fmt(m['auroc'])}  "
            f"d'={_fmt(m['dprime'])}  "
            f"mu_in={_fmt(m['mu_within'])}  mu_out={_fmt(m['mu_between'])}  "
            f"sep={_fmt(m['sep'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimise RinseParams hyperparameters with Bayesian optimisation"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key (prefer env var MP_API_KEY).",
    )
    parser.add_argument(
        "--n-structures",
        type=int,
        default=_DEFAULT_N_STRUCTURES,
        help=f"Number of pseudo-random structures to use (default: {_DEFAULT_N_STRUCTURES}).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=_DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials (default: {_DEFAULT_N_TRIALS}).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=_DEFAULT_N_FOLDS,
        help=(
            "Number of cross-validation folds for the objective "
            f"(default: {_DEFAULT_N_FOLDS}; use 1 to disable folding)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Random seed for structure sampling and TPE sampler (default: {_DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=_DEFAULT_CACHE,
        help=f"Path for the structure cache pickle (default: {_DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--full-cache",
        type=Path,
        default=_DEFAULT_FULL_CACHE,
        help=(
            "Group-aware record pickle from build_training_set.py "
            f"(default: {_DEFAULT_FULL_CACHE}). Used in preference to --cache "
            "when present; enables the clustering objective."
        ),
    )
    parser.add_argument(
        "--no-clusters",
        action="store_true",
        help=(
            "Ignore cluster groups and run the legacy single-objective "
            "separation study (minimise mean pairwise correlation) instead of "
            "the same-vs-different AUROC objective. Not recommended."
        ),
    )
    parser.add_argument(
        "--test-cache",
        type=Path,
        default=_DEFAULT_TEST_CACHE,
        help=(
            "Held-out test-split record pickle from build_training_set.py "
            f"(default: {_DEFAULT_TEST_CACHE}). When present, the reported "
            "params are evaluated on this unseen set."
        ),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Fetch and cache structures, then exit without optimising.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching; require cache to exist.",
    )
    parser.add_argument(
        "--study-name",
        default=_DEFAULT_STUDY_NAME,
        help=f"Optuna study name (default: {_DEFAULT_STUDY_NAME!r}).",
    )
    parser.add_argument(
        "--storage",
        default=_DEFAULT_STUDY_DB,
        help=f"Optuna storage URL (default: {_DEFAULT_STUDY_DB!r}). "
        "The study is resumable across runs.",
    )
    parser.add_argument(
        "--id-chunk-size",
        type=int,
        default=5000,
        help="Page size used when listing all material IDs (default: 5000).",
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=200,
        help="Number of material IDs per structure-fetch batch (default: 200).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of worker processes for descriptor computation "
            "(default: 1 = serial). Each worker loads its own copy of the "
            "structures, so higher values use proportionally more memory. "
            "Use 0 to auto-select os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--eval-params",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Evaluate a single user-specified RinseParams set as one trial and "
            "add it to the study (so it is scored on the same objective), "
            "instead of running the search. Accepts "
            "either comma-separated key=value pairs (recommended on PowerShell, "
            "no quoting needed) e.g. "
            "n_max=16,l_min=4,l_max=20,sin_theta_over_lambda_max=0.5 "
            "or a JSON object of RinseParams fields. Omitted fields use defaults; "
            "give either l_max or n_l_levels."
        ),
    )
    args = parser.parse_args()

    # ---- Load structures: prefer the group-aware full cache ----
    records: list[dict[str, Any]] | None = None
    if not args.no_clusters and args.full_cache and args.full_cache.exists():
        print(
            f"Loading group-aware records from {args.full_cache}...",
            file=sys.stderr,
        )
        with open(args.full_cache, "rb") as fh:
            records = pickle.load(fh)
        print(f"Loaded {len(records)} records", file=sys.stderr)

    if records is None:
        # Fall back to the flat (id, cif) cache / MP fetch flow.
        if args.cache.exists():
            print(f"Loading structure cache from {args.cache}...", file=sys.stderr)
            with open(args.cache, "rb") as fh:
                pairs: list[tuple[str, str]] = pickle.load(fh)
            print(f"Loaded {len(pairs)} cached structures", file=sys.stderr)
        elif args.no_fetch:
            sys.exit(f"Cache not found at {args.cache} and --no-fetch was set.")
        else:
            api_key = _resolve_api_key(args.api_key)
            pairs = _fetch_and_cache(
                api_key,
                args.n_structures,
                args.seed,
                args.cache,
                fetch_batch_size=args.fetch_batch_size,
                id_chunk_size=args.id_chunk_size,
            )

        if args.cache_only:
            print("--cache-only: exiting after fetch.", file=sys.stderr)
            return

        records = _pairs_to_records(pairs)
    elif args.cache_only:
        print("--cache-only: nothing to fetch (full cache present).", file=sys.stderr)
        return

    # ---- Convert CIF strings to xrs objects (once per run) ----
    print("Loading xrs structures from CIF cache...", file=sys.stderr)
    xrs_list, groups, roles, ids = _load_structures(records)
    if len(xrs_list) < _MIN_VALID:
        sys.exit(
            f"Only {len(xrs_list)} structures loaded; need at least {_MIN_VALID}."
        )

    # ---- Determine objective mode from cluster group availability ----
    cluster_group_sizes = Counter(
        group for group, role in zip(groups, roles) if role == "cluster"
    )
    usable_cluster_groups = {
        group for group, size in cluster_group_sizes.items() if size >= 2
    }
    n_cluster_groups = len(usable_cluster_groups)
    cluster_positions = [
        i
        for i, (group, role) in enumerate(zip(groups, roles))
        if role == "cluster" and group in usable_cluster_groups
    ]
    use_auroc = (not args.no_clusters) and n_cluster_groups >= 1

    if not use_auroc and not args.no_clusters:
        sys.exit(
            "No cluster/redetermination groups with >= 2 members were found, so "
            "the same-vs-different AUROC objective has no positive pairs.\n"
            "Provide a group-aware training set (build_training_set.py, "
            "--full-cache) or pass --no-clusters to fall back to the legacy "
            "separation objective."
        )

    # ---- Cross-validation folds over the background (distinct) structures ----
    # In AUROC mode each fold combines all cluster structures (positives) with
    # one fold of these distinct structures (extra negatives); in legacy mode
    # they are the separation structures whose mean correlation is minimised.
    sep_positions = [
        i for i, role in enumerate(roles) if role == "distinct"
    ]
    sep_folds = _kfold_indices(sep_positions, args.n_folds, args.seed)
    if use_auroc:
        print(
            f"Objective: single (same-vs-different AUROC, maximise)"
            f"; {n_cluster_groups} cluster group(s) supplying positive pairs, "
            f"{len(cluster_positions)} cluster + {len(sep_positions)} distinct "
            f"background structures in {len(sep_folds)} CV fold(s)",
            file=sys.stderr,
        )
    else:
        print(
            f"Objective: single (legacy separation, minimise)"
            f"; {len(sep_positions)} distinct structures "
            f"in {len(sep_folds)} CV fold(s)",
            file=sys.stderr,
        )

    # ---- Bayesian optimisation ----
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # AUROC is a maximisation study; keep it under a distinct name so a resumed
    # legacy (minimise) study in the same storage does not clash on direction.
    study_name = f"{args.study_name}_auroc" if use_auroc else args.study_name
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize" if use_auroc else "minimize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing:
        print(
            f"Resuming study '{study_name}' "
            f"({n_existing} completed trials found).",
            file=sys.stderr,
        )

    # ---- Optionally enqueue a single user-specified parameter set ----
    n_trials = args.n_trials
    if args.eval_params:
        try:
            spec = _parse_param_spec(args.eval_params)
            trial_params = _spec_to_trial_params(spec)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            sys.exit(f"--eval-params could not be parsed: {exc}")
        study.enqueue_trial(trial_params, skip_if_exists=False)
        n_trials = 1
        print(
            "Evaluating a single user-specified parameter set "
            "(added to the study):",
            file=sys.stderr,
        )
        for key, value in trial_params.items():
            print(f"  {key} = {value!r}", file=sys.stderr)

    print(
        f"Starting optimisation: {n_trials} trials, "
        f"{len(xrs_list)} structures",
        file=sys.stderr,
    )

    # ---- Optional process pool for parallel descriptor computation ----
    n_workers = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    executor: cf.ProcessPoolExecutor | None = None
    records_tmp: str | None = None
    if n_workers > 1:
        # Persist the in-use records so each worker loads them once (via the
        # pool initialiser) instead of re-pickling on every trial.
        fd, records_tmp = tempfile.mkstemp(suffix=".pkl", prefix="rinse_opt_recs_")
        os.close(fd)
        with open(records_tmp, "wb") as fh:
            pickle.dump(records, fh)
        executor = cf.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(records_tmp,),
        )
        print(
            f"Using {n_workers} worker processes for descriptor computation.",
            file=sys.stderr,
        )

    try:
        study.optimize(
            lambda trial: _objective(
                trial,
                xrs_list,
                groups,
                sep_folds,
                cluster_positions,
                use_auroc,
                ids,
                executor,
                n_workers,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # ---- Report results ----

        def _describe_trial(trial: optuna.trial.FrozenTrial, prefix: str) -> None:
            if use_auroc:
                std = trial.user_attrs.get("auroc_cv_std", 0.0)
                dp = trial.user_attrs.get("dprime_cv_mean", float("nan"))
                mu_in = trial.user_attrs.get("mu_within", float("nan"))
                mu_out = trial.user_attrs.get("mu_between", float("nan"))
                print(
                    f"{prefix} auroc={trial.value:.6f}±{std:.6f}  "
                    f"d'={dp:.4f}  mu_in={mu_in:.4f}  mu_out={mu_out:.4f}"
                )
            else:
                std = trial.user_attrs.get("sep_cv_std", 0.0)
                print(
                    f"{prefix} cross-validated mean Pearson (separation) "
                    f"{trial.value:.6f} (±{std:.6f})"
                )

        if args.eval_params:
            # A specific parameter set was requested: report and test-score only
            # that set, without re-evaluating the rest of the study.
            evaluated = study.trials[-1]
            reported = [evaluated]
            _describe_trial(
                evaluated,
                f"\nEvaluated parameter set (trial {evaluated.number}):",
            )
            _print_trial_params(evaluated)
        else:
            best = study.best_trial
            reported = [best]
            _describe_trial(best, "\nBest trial:")
            print("Best RinseParams:")
            _print_trial_params(best)

        # ---- Held-out test-set evaluation of the reported params ----
        if args.test_cache and args.test_cache.exists():
            print(
                f"\nLoading held-out test set from {args.test_cache}...",
                file=sys.stderr,
            )
            with open(args.test_cache, "rb") as fh:
                test_records = pickle.load(fh)
            test_xrs, test_groups, test_roles, test_ids = _load_structures(test_records)
            if test_xrs:
                _report_test_set(reported, test_xrs, test_groups, test_roles, test_ids)
            else:
                print("  No usable test structures loaded.", file=sys.stderr)
    finally:
        if executor is not None:
            executor.shutdown()
        if records_tmp is not None:
            Path(records_tmp).unlink(missing_ok=True)

    print(f"\nStudy saved to {args.storage!r} as '{study_name}'.")


if __name__ == "__main__":
    main()
