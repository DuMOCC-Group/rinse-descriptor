"""Enumerate all 16-bit proquints and report which ones are English words.

The script generates every possible 5-character proquint (all 2^16 values)
and checks them against one or more plain-text word lists.

By default the script uses a packaged English word-frequency list from
``wordfreq``. You can also pass one or more plain-text word lists with
``--word-list`` for broader coverage.

Word lists are expected to contain one word per line. Matching is case-
insensitive and ignores lines that do not reduce to a simple ASCII word.

Examples
--------
    uv run tools/find_proquint_words.py --word-list C:/path/to/words.txt
    uv run tools/find_proquint_words.py --word-list C:/path/to/words.txt --output matches.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterable
from pathlib import Path

_CONSONANTS = "bdfghjklmnprstvz"
_VOWELS = "aiou"
_N_VALUES = 1 << 16
_DEFAULT_WORDFREQ_COUNT = 2_000_000


def int16_to_proquint(value: int) -> str:
    """Encode a 16-bit unsigned integer as a five-character proquint word."""
    if not 0 <= value < _N_VALUES:
        raise ValueError(f"value must be in [0, {(_N_VALUES - 1)}], got {value}")

    c1 = (value >> 12) & 0xF
    v1 = (value >> 10) & 0x3
    c2 = (value >> 6) & 0xF
    v2 = (value >> 4) & 0x3
    c3 = value & 0xF
    return _CONSONANTS[c1] + _VOWELS[v1] + _CONSONANTS[c2] + _VOWELS[v2] + _CONSONANTS[c3]


def iter_proquints() -> Iterable[tuple[int, str]]:
    """Yield every possible 16-bit proquint in numeric order."""
    for value in range(_N_VALUES):
        yield value, int16_to_proquint(value)


def _normalize_word(raw: str) -> str | None:
    """Return a lowercase ASCII word if the line looks like a plain word."""
    token = raw.strip().lower()
    if not token:
        return None

    token = token.split()[0]
    token = token.replace("'", "")
    if token.isascii() and token.isalpha():
        return token
    return None


def load_word_list(paths: Iterable[Path]) -> set[str]:
    """Load and normalize words from one or more text files."""
    words: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Word list not found: {path}")

        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                word = _normalize_word(line)
                if word is not None:
                    words.add(word)

    return words


def load_wordfreq_words(limit: int = _DEFAULT_WORDFREQ_COUNT) -> set[str]:
    """Load a built-in English frequency list from the wordfreq package."""
    try:
        from wordfreq import top_n_list
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing
        raise RuntimeError(
            "wordfreq is not installed. Install project dependencies or pass --word-list."
        ) from exc

    words: set[str] = set()
    for raw in top_n_list("en", limit):
        word = _normalize_word(raw)
        if word is not None:
            words.add(word)
    return words


def _commonness_score(word: str) -> float:
    """Return an English commonness score for sorting results."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return 0.0

    return zipf_frequency(word, "en")


def _default_word_list_paths() -> list[Path]:
    """Return common word-list locations, if any are present."""
    env_path = os.environ.get("PROQUINT_WORD_LIST")
    candidates = [Path(env_path)] if env_path else []
    candidates.extend(
        [
            Path("/usr/share/dict/words"),
            Path("/usr/dict/words"),
            Path("/usr/share/dict/web2"),
            Path("/usr/share/dict/web2a"),
        ]
    )
    return [path for path in candidates if path.exists()]


def _load_default_words() -> set[str]:
    """Load words from the best available built-in source."""
    try:
        return load_wordfreq_words()
    except RuntimeError:
        paths = _default_word_list_paths()
        if paths:
            return load_word_list(paths)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate all 2^16 proquints and report which ones are English words."
    )
    parser.add_argument(
        "--word-list",
        dest="word_lists",
        action="append",
        type=Path,
        help=(
            "Path to a plain-text English word list. Repeat the flag to combine multiple lists."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--wordfreq-count",
        type=int,
        default=_DEFAULT_WORDFREQ_COUNT,
        help="Number of English words to load from wordfreq when no explicit word list is given.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.word_lists:
        english_words = load_word_list(args.word_lists)
        source_label = f"{len(args.word_lists)} explicit word list(s)"
    else:
        try:
            english_words = load_wordfreq_words(args.wordfreq_count)
            source_label = f"wordfreq top {args.wordfreq_count:,} English words"
        except RuntimeError:
            word_list_paths = _default_word_list_paths()
            if not word_list_paths:
                print(
                    "No word source available. Install wordfreq or pass --word-list PATH.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            english_words = load_word_list(word_list_paths)
            source_label = f"{len(word_list_paths)} system word list(s)"

    matches: list[tuple[float, int, str]] = [
        (_commonness_score(proquint), value, proquint)
        for value, proquint in iter_proquints()
        if proquint in english_words
    ]
    matches.sort(key=lambda item: (-item[0], item[1], item[2]))

    output_handle = (
        open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    )
    try:
        writer = csv.writer(output_handle)
        writer.writerow(["value", "proquint"])
        for _, value, proquint in matches:
            writer.writerow([value, proquint])
    finally:
        if args.output:
            output_handle.close()

    print(
        f"Scanned {_N_VALUES} proquints against {len(english_words)} normalized words from "
        f"{source_label}; found {len(matches)} matches.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
