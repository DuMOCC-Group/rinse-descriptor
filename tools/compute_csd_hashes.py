"""Compute RINSE descriptor hashes for all structures in the CSD.

Descriptor computation is parallelised across worker processes (``--jobs``);
CSD access (entry iteration and SHELX RES export) runs serially in the main
process while the heavy crystallography runs in the pool.  The script also
supports index-based chunking for distributing work across separate machines.

Every 100 new structures a per-position letter histogram is printed to stderr
as a quick visual check that the hash characters are uniformly distributed.

Outputs:
    - csd_hashes_chunk_N.csv: CSV file with refcode and hash columns
    - csd_descriptors_chunk_N.pkl: Pickle file with refcodes and high-dimensional descriptors

Usage:
    python compute_csd_hashes.py                    # All structures, all CPUs
    python compute_csd_hashes.py --jobs 8           # Limit worker processes
    python compute_csd_hashes.py 10 0               # Process chunk 0 of 10
    python compute_csd_hashes.py 10 1               # Process chunk 1 of 10
    python compute_csd_hashes.py --refcode AABHTZ  # Single refcode
"""

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path

from ccdc.io import CrystalWriter, EntryReader
from rinse_descriptor import (
    RinseParams,
    compute_power_spectrum,
    compute_structure_factors,
    descriptor_hash,
    load_res,
    power_spectrum_to_vector,
)

# Proquint alphabet (mirrors rinse_descriptor._hash): a 5-character word is laid
# out as consonant, vowel, consonant, vowel, consonant.
_PROQUINT_CONSONANTS = "bdfghjklmnprstvz"  # 16 symbols (positions 1, 3, 5)
_PROQUINT_VOWELS = "aiou"  # 4 symbols (positions 2, 4)
_HIST_BLOCKS = " ▁▂▃▄▅▆▇█"


class _LetterHistogram:
    """Accumulate per-position letter counts across proquint hash strings.

    A hash of ``n_words`` proquint words has ``5 * n_words`` character
    positions.  For each position we tally how often each allowed symbol
    appears; :meth:`render` draws a sparkline per position so a uniform
    distribution shows as a roughly flat bar.
    """

    def __init__(self, n_words: int) -> None:
        self.n_words = n_words
        self.total = 0
        self._alphabets = [
            _PROQUINT_VOWELS if (p % 5) in (1, 3) else _PROQUINT_CONSONANTS
            for p in range(5 * n_words)
        ]
        self._counts = [dict.fromkeys(alpha, 0) for alpha in self._alphabets]

    def update(self, hash_str: str) -> None:
        """Tally the characters of one hash string."""
        pos = 0
        for word in hash_str.split("-"):
            for ch in word:
                if pos < len(self._counts) and ch in self._counts[pos]:
                    self._counts[pos][ch] += 1
                pos += 1
        self.total += 1

    def render(self) -> str:
        """Return a multi-line sparkline of the per-position distributions."""
        lines = [
            f"  hash letter distribution over {self.total} structures "
            f"({self.n_words} words):"
        ]
        for w in range(self.n_words):
            for ip in range(5):
                pos = w * 5 + ip
                alpha = self._alphabets[pos]
                counts = self._counts[pos]
                kind = "vowel" if ip in (1, 3) else "cons "
                peak = max(counts.values()) or 1
                bars = "".join(
                    _HIST_BLOCKS[min(8, round(8 * counts[ch] / peak))] for ch in alpha
                )
                prefix = f"    w{w + 1} p{ip + 1} {kind} "
                lines.append(f"{prefix}{alpha}")
                lines.append(f"{' ' * len(prefix)}{bars}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker process: turn a SHELX RES string into a descriptor vector.
# ---------------------------------------------------------------------------

_WORKER_PARAMS: RinseParams | None = None


def _worker_init(params: RinseParams) -> None:
    """Pool initialiser: stash the shared :class:`RinseParams` per worker."""
    global _WORKER_PARAMS
    _WORKER_PARAMS = params


def _worker_compute(task: tuple[str, str]) -> tuple[str, object, str | None]:
    """Compute a descriptor for one ``(refcode, res_string)`` task.

    Returns ``(refcode, descriptor, None)`` on success or
    ``(refcode, None, error_message)`` on failure.  Runs in a worker process,
    so exceptions are captured and returned rather than raised.
    """
    refcode, res_string = task
    params = _WORKER_PARAMS
    assert params is not None  # set by _worker_init
    try:
        xrs = load_res(StringIO(res_string))
        if xrs.scatterers().size() == 0:
            return refcode, None, "no scatterers"
        reflections = compute_structure_factors(
            xrs,
            sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
        )
        P = compute_power_spectrum(reflections, params=params)
        desc = power_spectrum_to_vector(P)
        return refcode, desc, None
    except Exception as exc:  # noqa: BLE001 - report and continue
        return refcode, None, f"{type(exc).__name__}: {exc}"


def _entry_to_res_string(entry: object) -> str:
    """Convert a CSD entry to SHELX RES text using the CSD writer API."""
    fd, tmp_name = tempfile.mkstemp(suffix=".res")
    os.close(fd)
    Path(tmp_name).unlink(missing_ok=True)

    try:
        with CrystalWriter(tmp_name, format="res") as writer:
            writer.write(entry.crystal)

        res_string = Path(tmp_name).read_text(encoding="utf-8", errors="replace")
        if not res_string.rstrip().endswith("END"):
            res_string = f"{res_string.rstrip()}\nEND\n"
        return res_string
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _process_single(refcode: str, n_words: int) -> None:
    """Compute and print the descriptor hash for one CSD refcode."""
    reader = EntryReader("CSD")
    entry = reader.entry(refcode)
    params = RinseParams()
    t0 = time.perf_counter()
    res_string = _entry_to_res_string(entry)
    t_res = time.perf_counter()
    xrs = load_res(StringIO(res_string))
    t_load = time.perf_counter()
    reflections = compute_structure_factors(
        xrs,
        sin_theta_over_lambda_max=params.sin_theta_over_lambda_max,
    )
    t_sf = time.perf_counter()
    P = compute_power_spectrum(reflections, params=params)
    desc = power_spectrum_to_vector(P)
    t_ps = time.perf_counter()
    hash_str = descriptor_hash(desc, n_words=n_words)
    t_hash = time.perf_counter()
    print(f"{refcode}\t{hash_str}")
    print(
        f"  res_string={(t_res - t0) * 1e3:.1f}ms  "
        f"load_res={(t_load - t_res) * 1e3:.1f}ms  "
        f"struct_factors={(t_sf - t_load) * 1e3:.1f}ms  "
        f"power_spectrum={(t_ps - t_sf) * 1e3:.1f}ms  "
        f"hash={(t_hash - t_ps) * 1e3:.1f}ms  "
        f"total={(t_hash - t0) * 1e3:.1f}ms",
        file=sys.stderr,
    )


def main():
    """Process all CSD structures and compute their RINSE descriptor hashes."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute RINSE descriptor hashes for CSD structures"
    )
    parser.add_argument(
        "num_chunks",
        type=int,
        nargs="?",
        default=1,
        help="Total number of chunks to split the CSD into (default: 1)",
    )
    parser.add_argument(
        "chunk_id",
        type=int,
        nargs="?",
        default=0,
        help="Which chunk to process (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--refcode",
        type=str,
        default=None,
        help="Process a single refcode and print its hash, then exit",
    )
    parser.add_argument(
        "--n-words",
        type=int,
        default=2,
        help="Number of proquint words per hash (default: 2).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes for descriptor computation (default: all CPUs).",
    )
    args = parser.parse_args()

    if args.refcode is not None:
        _process_single(args.refcode, args.n_words)
        return

    if args.chunk_id >= args.num_chunks:
        print(
            f"Error: chunk_id ({args.chunk_id}) must be less than num_chunks ({args.num_chunks})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Open CSD reader
    print("Opening CSD database...", file=sys.stderr)
    reader = EntryReader("CSD")

    # Determine output file names based on chunking
    if args.num_chunks > 1:
        chunk_suffix = f"_chunk_{args.chunk_id}"
        print(f"Processing chunk {args.chunk_id} of {args.num_chunks}...", file=sys.stderr)
    else:
        chunk_suffix = ""

    pickle_file = Path(f"csd_descriptors{chunk_suffix}.pkl")
    csv_file = Path(f"csd_hashes{chunk_suffix}.csv")

    # Storage for descriptors - load existing data if available
    resuming = False
    if pickle_file.exists():
        print("Loading existing descriptors from pickle file...", file=sys.stderr)
        with open(pickle_file, "rb") as f:
            existing_refcodes, existing_descriptors = pickle.load(f)
            refcodes = list(existing_refcodes)
            descriptors = list(existing_descriptors)
        processed_refcodes = set(refcodes)
        print(f"Loaded {len(refcodes)} existing descriptors", file=sys.stderr)
        resuming = True
    else:
        refcodes = []
        descriptors = []
        processed_refcodes = set()

    params = RinseParams()
    n_words = args.n_words
    jobs = max(1, args.jobs)
    histogram = _LetterHistogram(n_words)

    # Open CSV file for writing
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["refcode", "hash"])

        # If resuming, replay existing descriptors into the CSV and histogram.
        if resuming:
            print("Writing existing hashes to CSV...", file=sys.stderr)
            for refcode, desc in zip(refcodes, descriptors):
                hash_str = descriptor_hash(desc, n_words=n_words)
                writer.writerow([refcode, hash_str])
                histogram.update(hash_str)
            csvfile.flush()

        new_count = 0
        errors = 0
        errors_local: list[int] = []
        stats = {"checked": 0, "skipped_chunk": 0}
        start_time = time.perf_counter()

        def _producer():
            """Yield ``(refcode, res_string)`` for structures in this chunk.

            CSD access (entry iteration and SHELX RES export) runs serially in
            the main process; the returned RES strings are farmed out to worker
            processes for the heavy descriptor computation.
            """
            for entry_idx, entry in enumerate(reader):
                if args.num_chunks > 1 and entry_idx % args.num_chunks != args.chunk_id:
                    stats["skipped_chunk"] += 1
                    continue
                stats["checked"] += 1
                refcode = entry.identifier
                if refcode in processed_refcodes:
                    continue
                try:
                    res_string = _entry_to_res_string(entry)
                except Exception as e:
                    errors_local.append(1)
                    print(f"Error exporting {refcode}: {e}", file=sys.stderr)
                    continue
                yield refcode, res_string

        print(
            f"Processing structures with {jobs} worker process(es)...",
            file=sys.stderr,
        )

        pool = None
        if jobs > 1:
            pool = mp.Pool(jobs, initializer=_worker_init, initargs=(params,))
            results = pool.imap_unordered(_worker_compute, _producer(), chunksize=1)
        else:
            _worker_init(params)
            results = (_worker_compute(task) for task in _producer())

        try:
            for refcode, desc, _err in results:
                if desc is None:
                    errors += 1
                    continue

                refcodes.append(refcode)
                descriptors.append(desc)
                processed_refcodes.add(refcode)
                new_count += 1

                hash_str = descriptor_hash(desc, n_words=n_words)
                writer.writerow([refcode, hash_str])
                histogram.update(hash_str)

                # Every 100 new structures: checkpoint and show the distribution.
                if new_count % 100 == 0:
                    csvfile.flush()
                    with open(pickle_file, "wb") as f:
                        pickle.dump((refcodes, descriptors), f)

                    elapsed = time.perf_counter() - start_time
                    rate = new_count / elapsed if elapsed > 0 else 0.0
                    checked = stats["checked"]
                    total_errors = errors + len(errors_local)
                    chunk_note = f" (chunk: {checked})" if args.num_chunks > 1 else ""
                    print(
                        f"\nChecked {checked}{chunk_note} structures, added "
                        f"{new_count} new ({total_errors} errors)  [{rate:.1f} struct/s]",
                        file=sys.stderr,
                    )
                    print(histogram.render(), file=sys.stderr)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        checked = stats["checked"]
        total_errors = errors + len(errors_local)
        if args.num_chunks > 1:
            print(
                f"\nComplete! Checked {checked} structures in chunk, "
                f"added {new_count} new ({total_errors} errors).",
                file=sys.stderr,
            )
        else:
            print(
                f"\nComplete! Checked {checked} structures, "
                f"added {new_count} new ({total_errors} errors).",
                file=sys.stderr,
            )
        if histogram.total:
            print(histogram.render(), file=sys.stderr)
        print(f"Results saved to {csv_file}", file=sys.stderr)

    # Final save of descriptors
    with open(pickle_file, "wb") as f:
        pickle.dump((refcodes, descriptors), f)
    print(
        f"Descriptors saved to {pickle_file} ({len(refcodes)} total structures)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
