"""Locality-sensitive hash (SimHash) → proquint string for descriptor vectors.

Usage
-----
>>> from rinse_descriptor import descriptor, descriptor_hash
>>> x = descriptor(crystal, flatten=True)
>>> descriptor_hash(x)
'lusab-babad-gutih-tugad-mudof'

Algorithm
---------
Given a flat descriptor vector **x** of length *D*:

1. Project **x** onto the first ``n_bits`` principal components from PCA,
   using the precomputed PCA model stored in ``pca_components.json``.
2. Compute ``bits = (projected_x > 0)`` — the hash bit vector.
3. Pack the bits into ``n_words`` 16-bit integers and encode each as a
   five-character proquint word (``CVCVC`` pattern, 4+2+4+2+4 = 16 bits).

The hash is **deterministic** and uses learned principal components
rather than random projections. Two structurally similar descriptors
will produce the same hash because they project to nearby points in
the PCA subspace.

Proquint alphabet
-----------------
* Consonants (4-bit index): ``b d f g h j k l m n p r s t v z``
* Vowels    (2-bit index):  ``a i o u``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Proquint alphabet
# ---------------------------------------------------------------------------

_CONSONANTS = "bdfghjklmnprstvz"  # 16 characters → 4-bit index
_VOWELS = "aiou"  # 4 characters → 2-bit index

# Bits per proquint word: C(4) V(2) C(4) V(2) C(4) = 16
_BITS_PER_WORD = 16

#: Default number of proquint words produced by :func:`descriptor_hash`.
DEFAULT_HASH_WORDS = 1

# ---------------------------------------------------------------------------
# PCA model loading
# ---------------------------------------------------------------------------

_PCA_CACHE: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}


def _get_default_pca_path() -> Path:
    """Get the default path to the bundled PCA components file."""
    return Path(__file__).parent / "data" / "pca_components.json"


def _load_pca_components(
    pca_file: str | Path | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load PCA components and mean from JSON file (cached).

    Parameters
    ----------
    pca_file : str, Path, or None
        Path to PCA components JSON file. If None, uses the bundled
        data file in the package.

    Returns
    -------
    components : ndarray
        PCA components matrix of shape (n_components, n_features)
    mean : ndarray
        Mean vector used for centering, shape (n_features,)
    """
    if pca_file is None:
        path = _get_default_pca_path()
        cache_key = "__default__"
    else:
        path = Path(pca_file)
        cache_key = str(pca_file)

    if cache_key in _PCA_CACHE:
        return _PCA_CACHE[cache_key]

    if not path.exists():
        raise FileNotFoundError(
            f"PCA components file not found: {path}\n"
            f"The bundled PCA model should be in the package data directory."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    components = np.array(data["components"], dtype=np.float64)
    mean = np.array(data["mean"], dtype=np.float64)

    _PCA_CACHE[cache_key] = (components, mean)
    return components, mean


def _int16_to_proquint(n: int) -> str:
    """Encode a 16-bit unsigned integer as a five-character proquint word."""
    c1 = (n >> 12) & 0xF
    v1 = (n >> 10) & 0x3
    c2 = (n >> 6) & 0xF
    v2 = (n >> 4) & 0x3
    c3 = n & 0xF
    return _CONSONANTS[c1] + _VOWELS[v1] + _CONSONANTS[c2] + _VOWELS[v2] + _CONSONANTS[c3]


def _proquint_to_int16(word: str) -> int:
    """Decode a five-character proquint word back to a 16-bit unsigned integer."""
    if len(word) != 5:
        raise ValueError(f"Proquint word must be exactly 5 characters, got {word!r}")
    c1 = _CONSONANTS.index(word[0])
    v1 = _VOWELS.index(word[1])
    c2 = _CONSONANTS.index(word[2])
    v2 = _VOWELS.index(word[3])
    c3 = _CONSONANTS.index(word[4])
    return (c1 << 12) | (v1 << 10) | (c2 << 6) | (v2 << 4) | c3


# ---------------------------------------------------------------------------
# Offensive-word blocklist
# ---------------------------------------------------------------------------
# Any proquint word containing one of these substrings is replaced by flipping
# its LSB, yielding a different valid proquint that is not a recognisable word.
_BLOCKED_SUBSTRINGS: tuple[str, ...] = ("fag", "nud", "jihad", "fuk", "hamas", "isis")


def _sanitise_word(word: str) -> str:
    """Replace a blocked proquint word by incrementing its value until clean."""

    def _is_blocked(w: str) -> bool:
        return any(sub in w for sub in _BLOCKED_SUBSTRINGS)

    if not _is_blocked(word):
        return word
    value = _proquint_to_int16(word)
    replacement = word
    shift = 1
    while _is_blocked(replacement):
        replacement = _int16_to_proquint((value ^ shift) & 0xFFFF)
        shift <<= 1
    return replacement


def descriptor_hash(
    x: NDArray[np.float64],
    n_words: int = DEFAULT_HASH_WORDS,
    pca_file: str | Path | None = None,
) -> str:
    """Convert a descriptor vector to a pronounceable proquint hash string.

    Parameters
    ----------
    x:
        Flat descriptor vector (any length).  Pass the output of
        :func:`~rinse_descriptor.power_spectrum_to_vector` or use
        ``descriptor(..., flatten=True)``.
    n_words:
        Number of proquint words in the output (default 1).
        Each word encodes 16 bits, so ``n_words=1`` → 16 hash bits total.
        The output string has ``5 * n_words + (n_words - 1)`` characters
        (words joined by ``"-"``).
    pca_file:
        Path to the PCA components JSON file. If None (default), uses
        the bundled PCA model from the package data directory.

    Returns
    -------
    str
        Hyphen-separated proquint words, e.g. ``"lusab-babad"``.

    Notes
    -----
    The descriptor is first centered using the PCA mean, then projected
    onto the first ``n_bits`` principal components. The sign of each
    projection coefficient determines the corresponding hash bit.
    """
    vec = np.asarray(x, dtype=np.float64).ravel()
    dim = vec.shape[0]
    if dim == 0:
        raise ValueError("Descriptor vector must not be empty.")

    n_bits = n_words * _BITS_PER_WORD

    # Load PCA components and mean
    components, mean = _load_pca_components(pca_file)

    # Check that we have enough components
    if components.shape[0] < n_bits:
        raise ValueError(
            f"Need at least {n_bits} PCA components for {n_words} words, "
            f"but only {components.shape[0]} available"
        )

    # Check dimension matches
    if components.shape[1] != dim:
        raise ValueError(
            f"Descriptor dimension {dim} does not match PCA feature dimension "
            f"{components.shape[1]}. Ensure the PCA was fit on the same descriptor type."
        )

    # Project onto first n_bits principal components
    # PCA projection: centered_x @ components.T = (x - mean) @ components.T
    centered = vec - mean
    projection = components[:n_bits] @ centered  # (n_bits,)
    bits: NDArray[np.bool_] = projection > 0  # (n_bits,)

    words: list[str] = []
    for i in range(n_words):
        chunk = bits[i * _BITS_PER_WORD : (i + 1) * _BITS_PER_WORD]
        # Pack 16 bools into a uint16 (big-endian: first bit is MSB)
        raw = np.packbits(chunk)  # 2 bytes
        value = (int(raw[0]) << 8) | int(raw[1])
        words.append(_sanitise_word(_int16_to_proquint(value)))

    return "-".join(words)


def hash_to_bits(hash_str: str) -> NDArray[np.bool_]:
    """Decode a proquint hash string back to a boolean bit vector.

    Parameters
    ----------
    hash_str:
        Hyphen-separated proquint words as returned by :func:`descriptor_hash`.

    Returns
    -------
    (n_words * 16,) bool array
    """
    words = hash_str.split("-")
    bits_list: list[NDArray[np.bool_]] = []
    for word in words:
        value = _proquint_to_int16(word)
        raw = np.array([value >> 8, value & 0xFF], dtype=np.uint8)
        bits_list.append(np.unpackbits(raw).astype(np.bool_))
    return np.concatenate(bits_list)
