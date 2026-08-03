"""Compute RINSE descriptors for every structure in a training-set pickle.

Reads a pickle produced by ``build_training_set.py`` (a list of record dicts,
each carrying a ``text`` structure string and a ``format`` of ``"cif"`` or
``"res"``), computes the RINSE descriptor for each structure, and writes the
descriptor back into the *same* pickle under a new ``descriptor`` key.

The script is **resumable**: any record that already carries a ``descriptor``
key (successful *or* failed) is skipped, and progress is checkpointed to disk
periodically and on Ctrl-C via an atomic temp-file replace.  An interrupted run
is continued by simply re-running the same command.  Records that fail to load
store ``descriptor = None`` and an ``descriptor_error`` message so they are not
retried; pass ``--retry-failed`` to attempt them again.

Progress (processed / total, percent, rate, elapsed and ETA) is printed to
stderr as a live-updating line.

Usage:
    uv run tools/compute_descriptors.py --input training_set_full.pkl
    uv run tools/compute_descriptors.py --input training_set_full.pkl --checkpoint-every 200
    uv run tools/compute_descriptors.py --input training_set_full.pkl --retry-failed
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from rinse_descriptor import RinseParams, descriptor, load_cif, load_res

Record = dict[str, Any]

# Exceptions from a single structure that should not abort the whole run.
_RECOVERABLE = (ValueError, RuntimeError, OSError, KeyError, AttributeError, IndexError)


def _load_records(path: Path) -> list[Record]:
    """Load the training-set records from *path*."""
    with open(path, "rb") as fh:
        records = pickle.load(fh)
    if not isinstance(records, list):
        raise TypeError(
            f"{path} does not contain a list of records "
            f"(got {type(records).__name__}); is this a build_training_set.py pickle?"
        )
    return records


def _save_records_atomic(records: list[Record], path: Path) -> None:
    """Write *records* to *path* atomically (temp file + os.replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _compute_descriptor(record: Record, params: RinseParams) -> np.ndarray:
    """Compute the RINSE descriptor for a single record.

    Loads the stored structure text with the loader selected by the record's
    ``format`` field and returns the flattened descriptor vector.
    """
    text = record["text"]
    fmt = record.get("format", "cif")
    loader = load_res if fmt == "res" else load_cif
    xrs = loader(StringIO(text))
    return descriptor(xrs, params=params)


def _is_done(record: Record, retry_failed: bool, recompute_all: bool) -> bool:
    """Return whether *record* already has a usable descriptor.

    A record is done when it carries a ``descriptor`` key.  When
    ``retry_failed`` is set, records whose descriptor is ``None`` (previous
    failures) are treated as not done so they are recomputed.
    """
    if recompute_all:
        return False
    if "descriptor" not in record:
        return False
    if retry_failed and record.get("descriptor") is None:
        return False
    return True


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    seconds = int(max(0.0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _print_progress(done: int, total: int, start: float, *, final: bool = False) -> None:
    """Print an in-place progress line to stderr."""
    elapsed = time.perf_counter() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    pct = 100.0 * done / total if total else 100.0
    line = (
        f"\r[compute_descriptors] {done}/{total} ({pct:5.1f}%)  "
        f"{rate:6.1f} struct/s  elapsed {_fmt_hms(elapsed)}  eta {_fmt_hms(remaining)}"
    )
    end = "\n" if final else ""
    print(line.ljust(90), end=end, file=sys.stderr, flush=True)


def main() -> None:
    """Compute descriptors for all records in a training-set pickle."""
    parser = argparse.ArgumentParser(
        description="Compute RINSE descriptors for a build_training_set.py pickle."
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Recompute descriptors for all records, ignoring any existing descriptor fields.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the training-set pickle (updated in place).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Checkpoint the pickle to disk after this many new descriptors (default: 100).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Recompute records whose previous attempt failed (descriptor is None).",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: input file {input_path} not found", file=sys.stderr)
        sys.exit(1)

    params = RinseParams()

    print(f"Loading records from {input_path}...", file=sys.stderr)
    records = _load_records(input_path)
    total = len(records)

    pending = [i for i, r in enumerate(records) if not _is_done(r, args.retry_failed, args.recompute_all)]
    already = total - len(pending)
    print(
        f"Loaded {total} records; {already} already have descriptors, "
        f"{len(pending)} to compute.",
        file=sys.stderr,
    )
    if not pending:
        print("Nothing to do.", file=sys.stderr)
        return

    start = time.perf_counter()
    done = already
    new_ok = 0
    new_failed = 0
    since_checkpoint = 0

    try:
        for idx in pending:
            record = records[idx]
            try:
                desc = _compute_descriptor(record, params)
                record["descriptor"] = np.asarray(desc, dtype=np.float64)
                record.pop("descriptor_error", None)
                new_ok += 1
            except _RECOVERABLE as exc:
                record["descriptor"] = None
                record["descriptor_error"] = f"{type(exc).__name__}: {exc}"
                new_failed += 1
                print(
                    f"\n[warn] {record.get('id', '?')}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            done += 1
            since_checkpoint += 1
            _print_progress(done, total, start)

            if since_checkpoint >= args.checkpoint_every:
                _save_records_atomic(records, input_path)
                since_checkpoint = 0
    except KeyboardInterrupt:
        _print_progress(done, total, start, final=True)
        print("\nInterrupted; saving checkpoint...", file=sys.stderr)
        _save_records_atomic(records, input_path)
        print(
            f"Saved {new_ok} new descriptor(s) ({new_failed} failed) to {input_path}. "
            "Re-run the same command to resume.",
            file=sys.stderr,
        )
        sys.exit(130)

    _print_progress(done, total, start, final=True)
    _save_records_atomic(records, input_path)
    print(
        f"Done. Computed {new_ok} new descriptor(s), {new_failed} failed; "
        f"{done}/{total} records now have a descriptor field. Saved to {input_path}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
