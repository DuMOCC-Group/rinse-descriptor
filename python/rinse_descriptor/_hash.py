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

1. Draw a deterministic random projection matrix **W** of shape
   ``(n_bits, D)`` using :func:`numpy.random.default_rng` with seed 0.
2. Compute ``bits = (W @ x > 0)`` — the SimHash bit vector.
3. Pack the bits into ``n_words`` 16-bit integers and encode each as a
   five-character proquint word (``CVCVC`` pattern, 4+2+4+2+4 = 16 bits).

The hash is **deterministic** for the same input vector length and
``n_words`` value.  Two structurally similar descriptors will tend to
produce the same hash because nearby vectors in ℝᴰ agree on most
random-halfspace signs (Johnson–Lindenstrauss locality preservation).

Proquint alphabet
-----------------
* Consonants (4-bit index): ``b d f g h j k l m n p r s t v z``
* Vowels    (2-bit index):  ``a i o u``
"""

from __future__ import annotations

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
# Any proquint word containing a substring from this set is replaced by
# flipping bits until a clean word is found.
# Note: Only valid proquint characters (consonants: bdfghjklmnprstvz, vowels: aiou)
_BLOCKED_WORDS: frozenset[str] = frozenset(
    [
        "fag",
        "jihad",
        "nud",
        "bum",
        "dik",  # Changed from "dic" - 'c' is not a valid proquint consonant
        "fuk",
    ]
)


def _sanitise_word(word: str) -> str:
    """Replace a blocked proquint word by incrementing its value until clean."""
    # Check if any blocked word appears as a substring
    if not any(blocked in word for blocked in _BLOCKED_WORDS):
        return word
    value = _proquint_to_int16(word)
    replacement = word
    shift = 1
    # Keep trying different values until we find one without blocked substrings
    while any(blocked in replacement for blocked in _BLOCKED_WORDS):
        replacement = _int16_to_proquint((value ^ shift) & 0xFFFF)
        shift <<= 1
        if shift > 0xFFFF:  # Prevent infinite loop
            break
    return replacement


def descriptor_hash(
    x: NDArray[np.float64],
    n_words: int = DEFAULT_HASH_WORDS,
) -> str:
    """Convert a descriptor vector to a pronounceable proquint hash string.

    Parameters
    ----------
    x:
        Flat descriptor vector (any length).  Pass the output of
        :func:`~rinse_descriptor.power_spectrum_to_vector` or use
        ``descriptor(..., flatten=True)``.
    n_words:
        Number of proquint words in the output (default 5).
        Each word encodes 16 bits, so ``n_words=5`` → 80 hash bits total.
        The output string has ``5 * n_words + (n_words - 1)`` characters
        (words joined by ``"-"``).

    Returns
    -------
    str
        Hyphen-separated proquint words, e.g. ``"lusab-babad-gutih-tugad-mudof"``.

    Notes
    -----
    The projection matrix **W** is generated deterministically from
    :func:`numpy.random.default_rng` with seed 0 for the given
    ``(n_bits, input_dim)`` shape, so the hash is fully reproducible.
    """
    vec = np.asarray(x, dtype=np.float64).ravel()
    dim = vec.shape[0]
    if dim == 0:
        raise ValueError("Descriptor vector must not be empty.")

    n_bits = n_words * _BITS_PER_WORD
    W = np.random.default_rng(0).standard_normal((n_bits, dim))  # (n_bits, D)
    bits: NDArray[np.bool_] = (W @ vec) > 0  # (n_bits,)

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
