"""Run chunked CSD descriptor/PCA jobs until hash assignments stabilise.

This meta-script incrementally builds a training set from chunked
``compute_csd_hashes.py`` outputs, re-runs ``compute_pca.py`` after each
additional chunk, and measures convergence on a fixed test chunk.

By default it splits the CSD into 1000 chunks and uses chunk 0 as the
convergence subset, matching the suggested workflow for monitoring whether
descriptor hashes stop changing as the PCA fit sees more data.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = REPO_ROOT / "convergence_runs"
DEFAULT_FINAL_OUTPUT = (
    REPO_ROOT / "python" / "rinse_descriptor" / "data" / "pca_components.json"
)

if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from rinse_descriptor import descriptor_hash


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally run compute_csd_hashes.py and compute_pca.py until "
            "hashes for a fixed chunk stop changing"
        )
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1000,
        help="Total number of CSD chunks to use (default: 1000)",
    )
    parser.add_argument(
        "--test-chunk",
        type=int,
        default=0,
        help="Chunk id used as the fixed convergence subset (default: 0)",
    )
    parser.add_argument(
        "--start-chunks",
        type=int,
        default=1,
        help="Evaluate convergence once at least this many chunks are merged (default: 1)",
    )
    parser.add_argument(
        "--chunk-step",
        type=int,
        default=1,
        help="Recompute PCA after every N newly added chunks (default: 1)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to include before stopping (default: num_chunks)",
    )
    parser.add_argument(
        "--stable-rounds",
        type=int,
        default=1,
        help=(
            "Require this many consecutive unchanged evaluations before declaring "
            "convergence (default: 1)"
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Forwarded to compute_pca.py --n-components (default: use its default)",
    )
    parser.add_argument(
        "--hash-words",
        type=int,
        default=1,
        help="Number of proquint words per hash when evaluating convergence (default: 1)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Directory for chunk outputs and intermediate PCA files (default: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT,
        help=(
            "Where to copy the converged PCA model after convergence "
            f"(default: {DEFAULT_FINAL_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--force-recompute-chunks",
        action="store_true",
        help="Re-run compute_csd_hashes.py even if a chunk pickle already exists",
    )
    return parser.parse_args()


def _run_python(script: Path, args: Sequence[str], cwd: Path) -> None:
    env = dict(**shutil.os.environ)
    python_path = str(REPO_ROOT / "python")
    if env.get("PYTHONPATH"):
        python_path = python_path + shutil.os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    command = [sys.executable, str(script), *args]
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _load_chunk(chunk_pickle: Path) -> tuple[list[str], list[object]]:
    with chunk_pickle.open("rb") as handle:
        refcodes, descriptors = pickle.load(handle)
    return list(refcodes), list(descriptors)


def _write_merged_pickle(path: Path, refcodes: Sequence[str], descriptors: Sequence[object]) -> None:
    with path.open("wb") as handle:
        pickle.dump((list(refcodes), list(descriptors)), handle)


def _hash_subset(
    refcodes: Sequence[str],
    descriptors: Sequence[object],
    pca_file: Path,
    hash_words: int,
) -> dict[str, str]:
    return {
        refcode: descriptor_hash(descriptor, n_words=hash_words, pca_file=str(pca_file))
        for refcode, descriptor in zip(refcodes, descriptors, strict=True)
    }


def _count_hash_changes(
    previous_hashes: dict[str, str] | None, current_hashes: dict[str, str]
) -> int | None:
    if previous_hashes is None:
        return None
    return sum(previous_hashes.get(refcode) != hash_str for refcode, hash_str in current_hashes.items())


def _write_history(path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = [
        "chunks_included",
        "descriptors",
        "changed_hashes",
        "stable_streak",
        "pca_file",
        "merged_pickle",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()

    if args.num_chunks < 1:
        raise SystemExit("--num-chunks must be at least 1")
    if args.test_chunk < 0 or args.test_chunk >= args.num_chunks:
        raise SystemExit("--test-chunk must be in the range [0, num_chunks)")
    if args.start_chunks < 1:
        raise SystemExit("--start-chunks must be at least 1")
    if args.chunk_step < 1:
        raise SystemExit("--chunk-step must be at least 1")
    if args.stable_rounds < 1:
        raise SystemExit("--stable-rounds must be at least 1")

    max_chunks = args.max_chunks if args.max_chunks is not None else args.num_chunks
    if max_chunks < 1 or max_chunks > args.num_chunks:
        raise SystemExit("--max-chunks must be in the range [1, num_chunks]")

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    compute_hashes_script = TOOLS_DIR / "compute_csd_hashes.py"
    compute_pca_script = TOOLS_DIR / "compute_pca.py"

    cumulative_refcodes: list[str] = []
    cumulative_descriptors: list[object] = []
    test_refcodes: list[str] | None = None
    test_descriptors: list[object] | None = None
    previous_hashes: dict[str, str] | None = None
    stable_streak = 0
    history_rows: list[dict[str, str]] = []
    last_pca_file: Path | None = None

    for chunk_id in range(max_chunks):
        chunk_pickle = work_dir / f"csd_descriptors_chunk_{chunk_id}.pkl"
        should_run_chunk = args.force_recompute_chunks or not chunk_pickle.exists()
        if should_run_chunk:
            print(
                f"[converge] computing chunk {chunk_id}/{args.num_chunks - 1} in {work_dir}",
                file=sys.stderr,
            )
            _run_python(
                compute_hashes_script,
                [str(args.num_chunks), str(chunk_id)],
                cwd=work_dir,
            )
        else:
            print(f"[converge] reusing existing chunk {chunk_id}", file=sys.stderr)

        if not chunk_pickle.exists():
            raise SystemExit(f"Expected chunk pickle was not created: {chunk_pickle}")

        chunk_refcodes, chunk_descriptors = _load_chunk(chunk_pickle)
        cumulative_refcodes.extend(chunk_refcodes)
        cumulative_descriptors.extend(chunk_descriptors)

        if chunk_id == args.test_chunk:
            test_refcodes = list(chunk_refcodes)
            test_descriptors = list(chunk_descriptors)
            print(
                f"[converge] loaded test subset from chunk {args.test_chunk} "
                f"({len(test_refcodes)} structures)",
                file=sys.stderr,
            )

        chunks_included = chunk_id + 1
        if chunks_included < args.start_chunks:
            continue
        if (chunks_included - args.start_chunks) % args.chunk_step != 0:
            continue
        if test_refcodes is None or test_descriptors is None:
            continue

        merged_pickle = work_dir / f"csd_descriptors_prefix_{chunks_included}.pkl"
        pca_output = work_dir / f"pca_components_prefix_{chunks_included}.json"
        _write_merged_pickle(merged_pickle, cumulative_refcodes, cumulative_descriptors)

        pca_args = ["--input", str(merged_pickle), "--output", str(pca_output)]
        if args.n_components is not None:
            pca_args.extend(["--n-components", str(args.n_components)])

        print(
            f"[converge] fitting PCA on {len(cumulative_refcodes)} structures "
            f"from {chunks_included} chunks",
            file=sys.stderr,
        )
        _run_python(compute_pca_script, pca_args, cwd=work_dir)

        current_hashes = _hash_subset(
            test_refcodes,
            test_descriptors,
            pca_output,
            args.hash_words,
        )
        changed_hashes = _count_hash_changes(previous_hashes, current_hashes)

        if changed_hashes == 0:
            stable_streak += 1
        else:
            stable_streak = 0

        history_rows.append(
            {
                "chunks_included": str(chunks_included),
                "descriptors": str(len(cumulative_refcodes)),
                "changed_hashes": "" if changed_hashes is None else str(changed_hashes),
                "stable_streak": str(stable_streak),
                "pca_file": str(pca_output),
                "merged_pickle": str(merged_pickle),
            }
        )
        _write_history(work_dir / "convergence_history.csv", history_rows)

        total_test = len(current_hashes)
        if changed_hashes is None:
            print(
                f"[converge] baseline established with {total_test} test hashes",
                file=sys.stderr,
            )
        else:
            print(
                f"[converge] chunk prefix {chunks_included}: {changed_hashes}/{total_test} "
                "test hashes changed",
                file=sys.stderr,
            )

        previous_hashes = current_hashes
        last_pca_file = pca_output

        if changed_hashes == 0 and stable_streak >= args.stable_rounds:
            args.final_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pca_output, args.final_output)
            print(
                f"[converge] convergence reached after {chunks_included} chunks; "
                f"copied PCA model to {args.final_output}",
                file=sys.stderr,
            )
            return 0

    if last_pca_file is not None:
        print(
            f"[converge] no convergence after {max_chunks} chunks; latest PCA is {last_pca_file}",
            file=sys.stderr,
        )
    else:
        print("[converge] no PCA evaluation was performed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())