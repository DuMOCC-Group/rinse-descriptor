"""Optimise RinseParams hyperparameters using Bayesian optimisation (Optuna).

Additional dependencies (not in project – install before running):
    uv pip install optuna mp-api pymatgen

Fetches N pseudo-random Materials Project structures (default 1000), caches
their CIF representations locally, then uses Optuna's TPE sampler to minimise
the mean pairwise Pearson correlation between descriptor vectors.

A low mean Pearson correlation indicates that the descriptor vectors are
well-separated across chemically diverse structures – i.e. the descriptor
has good discriminative power.

Usage:
    uv run tools/optimise_hyperparams.py
    uv run tools/optimise_hyperparams.py --n-structures 500 --n-trials 200
    uv run tools/optimise_hyperparams.py --cache-only   # fetch and cache only
    uv run tools/optimise_hyperparams.py --no-fetch     # skip fetch, use cache

API key:  pass --api-key or set MP_API_KEY / PMG_MAPI_KEY environment variable.

Outputs:
    mp_opt_structures.pkl  – cached (material_id, cif_string) pairs
    optuna_rinse.db        – SQLite study database (resumable across runs)
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
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
    power_spectrum_to_vector,
)

if TYPE_CHECKING:
    import optuna.trial

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECOVERABLE = (Sorry, ValueError, RuntimeError, OSError, NameError, AttributeError, KeyError)

_DEFAULT_CACHE = Path("mp_opt_structures.pkl")
_DEFAULT_SEED = 42
_DEFAULT_N_STRUCTURES = 1000
_DEFAULT_N_TRIALS = 100
_DEFAULT_STUDY_DB = "sqlite:///optuna_rinse.db"
_DEFAULT_STUDY_NAME = "rinse_hyperparams"


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------


def _build_params(trial: optuna.trial.Trial) -> RinseParams | None:
    """Sample a RinseParams from an Optuna trial, returning None if invalid."""
    n_max = trial.suggest_int("n_max", 4, 32)
    l_min = trial.suggest_int("l_min", 0, 16, step=2)
    # l_max is parameterised as l_min + 2 * n_l_levels to guarantee l_max > l_min
    # and to keep both even (required when include_odd_l=False).
    n_l_levels = trial.suggest_int("n_l_levels", 1, 10)
    l_max = l_min + 2 * n_l_levels
    trial.set_user_attr("l_max", l_max)

    sin_theta = trial.suggest_float("sin_theta_over_lambda_max", 0.3, 1.2)
    radial_basis = trial.suggest_categorical(
        "radial_basis",
        ["chebyshev", "bessel", "smooth_shells_cw", "smooth_shells_nl"],
    )
    intensity_norm = trial.suggest_categorical(
        "intensity_normalisation",
        ["none", "double_exponential", "empirical"],
    )
    intensity_falloff = trial.suggest_categorical(
        "intensity_falloff",
        ["none", "debye_waller"],
    )
    # u_iso is only meaningful when Debye-Waller falloff is active, but Optuna
    # needs a consistent parameter space across trials.  We sample it always and
    # pass it through; RinseParams ignores it when falloff="none".
    u_iso = trial.suggest_float("intensity_falloff_u_iso", 0.01, 0.30)
    log1p = trial.suggest_categorical("log1p", [False, True])

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
        )
    except ValueError as exc:
        print(f"  [skip] invalid params: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Descriptor computation
# ---------------------------------------------------------------------------


def _compute_descriptors(
    xrs_list: list[Any],
    params: RinseParams,
) -> np.ndarray:
    """Compute descriptor vectors for all loaded cctbx structures.

    Returns an (M, D) array of valid descriptor vectors (M ≤ len(xrs_list)).
    Structures that fail are silently skipped.
    """
    vecs: list[np.ndarray] = []
    for xrs in xrs_list:
        try:
            reflections = compute_structure_factors(
                xrs,
                sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
            )
            P = compute_power_spectrum(reflections, params=params)
            v = power_spectrum_to_vector(P)
            if np.isfinite(v).all() and np.linalg.norm(v) > 0.0:
                vecs.append(v)
        except _RECOVERABLE:
            continue
    if not vecs:
        return np.empty((0, 0))
    return np.array(vecs, dtype=np.float64)


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
# Optuna objective
# ---------------------------------------------------------------------------

# Minimum number of valid descriptors required to evaluate a trial.
_MIN_VALID = 20


def _objective(trial: optuna.trial.Trial, xrs_list: list[Any]) -> float:
    params = _build_params(trial)
    if params is None:
        raise optuna.TrialPruned()

    X = _compute_descriptors(xrs_list, params)
    if X.shape[0] < _MIN_VALID:
        print(
            f"  [prune] only {X.shape[0]} valid descriptors "
            f"(need {_MIN_VALID})",
            file=sys.stderr,
        )
        raise optuna.TrialPruned()

    r = _mean_pearson(X)
    print(
        f"  trial {trial.number}: mean_pearson={r:.6f}  "
        f"(n_descriptors={X.shape[0]}, d={X.shape[1]})",
        file=sys.stderr,
    )
    return r


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
# Load xrs objects from cached CIF strings
# ---------------------------------------------------------------------------


def _load_xrs_list(pairs: list[tuple[str, str]]) -> list[Any]:
    """Convert cached CIF strings to cctbx xrs objects, skipping failures."""
    xrs_list = []
    skipped = 0
    for mid, cif_text in pairs:
        try:
            xrs = load_cif(StringIO(cif_text))
            if xrs.scatterers().size() > 0:
                xrs_list.append(xrs)
            else:
                skipped += 1
        except _RECOVERABLE as exc:
            print(f"  Skipping {mid} during xrs load: {exc}", file=sys.stderr)
            skipped += 1
    print(
        f"Loaded {len(xrs_list)} xrs structures ({skipped} skipped)",
        file=sys.stderr,
    )
    return xrs_list


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
    args = parser.parse_args()

    # ---- Fetch / load cached CIF strings ----
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

    # ---- Convert CIF strings to xrs objects (once per run) ----
    print("Loading xrs structures from CIF cache...", file=sys.stderr)
    xrs_list = _load_xrs_list(pairs)
    if len(xrs_list) < _MIN_VALID:
        sys.exit(
            f"Only {len(xrs_list)} structures loaded; need at least {_MIN_VALID}."
        )

    # ---- Bayesian optimisation ----
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing:
        print(
            f"Resuming study '{args.study_name}' "
            f"({n_existing} completed trials found).",
            file=sys.stderr,
        )

    print(
        f"Starting optimisation: {args.n_trials} trials, "
        f"{len(xrs_list)} structures",
        file=sys.stderr,
    )

    study.optimize(
        lambda trial: _objective(trial, xrs_list),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ---- Report results ----
    best = study.best_trial
    l_max = best.user_attrs.get("l_max")
    if l_max is None:
        l_max = best.params["l_min"] + 2 * best.params["n_l_levels"]

    print(f"\nBest mean Pearson correlation: {best.value:.6f}")
    print("Best RinseParams:")
    print(f"  n_max                     = {best.params['n_max']}")
    print(f"  l_min                     = {best.params['l_min']}")
    print(f"  l_max (derived)           = {l_max}")
    print(f"  n_l_levels (derived)      = {best.params['n_l_levels']}")
    print(f"  sin_theta_over_lambda_max = {best.params['sin_theta_over_lambda_max']:.4f}")
    print(f"  radial_basis              = {best.params['radial_basis']!r}")
    print(f"  intensity_normalisation   = {best.params['intensity_normalisation']!r}")
    print(f"  intensity_falloff         = {best.params['intensity_falloff']!r}")
    print(f"  intensity_falloff_u_iso   = {best.params['intensity_falloff_u_iso']:.4f}")
    print(f"  log1p                     = {best.params['log1p']}")
    print(f"  l2                        = True")

    print(f"\nStudy saved to {args.storage!r} as '{args.study_name}'.")


if __name__ == "__main__":
    main()
