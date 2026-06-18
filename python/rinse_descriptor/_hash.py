"""Locality-sensitive hash (SimHash) → proquint string for descriptor vectors.

Usage
-----
>>> from rinse_descriptor import descriptor, descriptor_hash
>>> x = descriptor(crystal)           # flat 1-D vector by default
>>> descriptor_hash(x)                # whitened SimHash, 1 proquint word
'lusab'
>>> descriptor_hash(x, n_words=5)     # 80-bit hash
'lusab-babad-gutih-tugad-mudof'

Algorithm
---------
Given a flat descriptor vector **x** of length *D*:

1. **Whiten**: apply PCA whitening via :func:`default_whitening` (see below),
   projecting *x* into a *k*-dimensional space where every direction has unit
   variance.  This makes all proquint characters equally reachable.
2. **Project**: draw a deterministic random matrix **W** of shape
   ``(n_bits, k)`` using :func:`numpy.random.default_rng` with seed 0.
3. **Binarise**: ``bits = (W @ x_white > 0)`` — the SimHash bit vector.
4. **Encode**: pack bits into ``n_words`` 16-bit integers, each encoded as a
   five-character proquint word (``CVCVC``, 4+2+4+2+4 = 16 bits).

Locality sensitivity is preserved: two structurally similar descriptors agree
on most random-halfspace signs (Johnson–Lindenstrauss), so they tend to
produce the same hash.

Proquint alphabet
-----------------
* Consonants (4-bit index): ``b d f g h j k l m n p r s t v z``
* Vowels    (2-bit index):  ``a i o u``

Whitening
---------
By default, :func:`descriptor_hash` applies :func:`default_whitening`: a PCA
transform synthesised from 2000 pseudo-random positive unit vectors drawn from
an exponential distribution (mimicking the heavy-tailed, all-positive shape of
L2-normalised power spectra).  The synthesis uses a fixed seed so the result
is fully deterministic without shipping a data file.

To use a transform fitted from your own structures::

    wt = fit_hash_whitening(descriptor_many(structures))
    wt.save("my_whitening.npz")          # persist for later
    descriptor_hash(x, whitening=wt)

To reload::

    wt = HashWhitening.load("my_whitening.npz")
    descriptor_hash(x, whitening=wt)

To disable whitening entirely::

    descriptor_hash(x, whitening=False)  # bare log1p only
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from typing import IO

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
# Any proquint word appearing in this set is replaced by flipping its LSB,
# yielding a different valid proquint that is not a recognisable word.
# All entries must be exactly 5 characters using the proquint alphabet.
_BLOCKED_WORDS: frozenset[str] = frozenset(
    [
        "fagot",
        "fagit",
        "fagut",
        "fagor",
        "jihad",
        "nudif",
    ]
)


def _sanitise_word(word: str) -> str:
    """Replace a blocked proquint word by incrementing its value until clean."""
    if word not in _BLOCKED_WORDS:
        return word
    value = _proquint_to_int16(word)
    replacement = word
    shift = 1
    while replacement in _BLOCKED_WORDS:
        replacement = _int16_to_proquint((value ^ shift) & 0xFFFF)
        shift <<= 1
    return replacement


# ---------------------------------------------------------------------------
# PCA whitening
# ---------------------------------------------------------------------------


class HashWhitening:
    """PCA whitening transform for :func:`descriptor_hash`.

    :func:`descriptor_hash` uses :func:`default_whitening` automatically.
    Use this class directly only when you want to fit from real structures
    or serialise/deserialise a custom transform.

    The transform maps a descriptor vector through:

    1. ``log1p`` — compresses the positive-orthant cluster.
    2. Mean subtraction — centres the data at the origin.
    3. PCA projection — rotates into the directions of maximum variance.
    4. Scale division — whitens each axis to unit variance.

    The result is a *k*-dimensional vector whose components are approximately
    independent and unit-variance, so every SimHash projection row has a
    near-zero bias across the training distribution and all proquint characters
    become equally reachable.

    Fitting from real structures
    ----------------------------
    >>> wt = fit_hash_whitening(descriptor_many(structures))
    >>> wt.save("whitening.npz")
    >>> descriptor_hash(x, whitening=wt)

    Reloading
    ---------
    >>> wt = HashWhitening.load("whitening.npz")
    >>> descriptor_hash(x, whitening=wt)
    """

    def __init__(
        self,
        mean: NDArray[np.float64],
        components: NDArray[np.float64],
        scale: NDArray[np.float64],
    ) -> None:
        self.mean = mean  # (D,)  log1p-space mean
        self.components = components  # (k, D)  top-k PCA eigenvectors
        self.scale = scale  # (k,)  sqrt of sample eigenvalues

    @property
    def n_components(self) -> int:
        """Number of retained principal components."""
        return int(self.components.shape[0])

    @property
    def input_dim(self) -> int:
        """Descriptor dimension expected by :meth:`transform`."""
        return int(self.components.shape[1])

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a descriptor vector to a whitened *k*-dimensional vector.

        Applies ``log1p → subtract mean → project onto PCs → divide by scale``.

        Raises
        ------
        ValueError
            If *x* has a different length than :attr:`input_dim`.
        """
        vec = np.log1p(np.asarray(x, dtype=np.float64).ravel())
        if vec.shape[0] != self.input_dim:
            raise ValueError(
                f"Descriptor has {vec.shape[0]} dimensions but this whitening "
                f"transform expects {self.input_dim}.  Re-fit with matching "
                f"RinseParams, or use default_whitening() only with default "
                f"RinseParams."
            )
        vec = vec - self.mean
        proj = self.components @ vec  # (k,)
        return proj / self.scale

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save to a NumPy ``.npz`` archive."""
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
            scale=self.scale,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str] | IO[bytes]) -> HashWhitening:
        """Load from a NumPy ``.npz`` archive saved by :meth:`save`.

        *path* may be a file-system path, a :class:`pathlib.Path`, or any
        binary file-like object (e.g. from :mod:`importlib.resources`).
        """
        data = np.load(path)
        return cls(
            mean=data["mean"].astype(np.float64),
            components=data["components"].astype(np.float64),
            scale=data["scale"].astype(np.float64),
        )


def fit_hash_whitening(
    descriptors: Iterable[NDArray[np.float64]],
    n_components: int = 64,
) -> HashWhitening:
    """Fit a PCA whitening transform from a collection of descriptor vectors.

    Parameters
    ----------
    descriptors:
        Iterable of flat, non-negative descriptor vectors (all the same
        length).  Pass the output of :func:`~rinse_descriptor.descriptor` or
        :func:`~rinse_descriptor.descriptor_many`.
    n_components:
        Number of principal components to retain.  Automatically capped at
        ``min(n_samples - 1, input_dim)``.

    Returns
    -------
    HashWhitening
        Fitted transform.  Persist with :meth:`HashWhitening.save` and reload
        later with :meth:`HashWhitening.load`.

    Notes
    -----
    PCA is computed via an economy SVD of the log1p-transformed, centred data
    matrix, which is numerically stable for both ``N < D`` and ``N > D``.
    """
    mat = np.stack(
        [np.log1p(np.asarray(d, dtype=np.float64).ravel()) for d in descriptors]
    )  # (N, D)
    if mat.ndim != 2 or mat.shape[0] < 2:
        raise ValueError("Need at least 2 descriptor vectors to fit a whitening transform.")

    mean = mat.mean(axis=0)
    centered = mat - mean  # (N, D)

    # Economy SVD — stable for both fat (N < D) and tall (N > D) matrices
    _U, s, Vt = np.linalg.svd(centered, full_matrices=False)  # s: (min(N,D),)
    # Centring removes one degree of freedom, so the true rank is at most N-1
    k = min(n_components, mat.shape[0] - 1, mat.shape[1])
    components = Vt[:k]  # (k, D)

    # Sample eigenvalues of the covariance matrix
    eigenvalues = (s[:k] ** 2) / (mat.shape[0] - 1)
    scale = np.sqrt(eigenvalues)
    scale = np.where(scale > 0.0, scale, 1.0)  # guard near-zero trailing eigenvalues

    return HashWhitening(mean=mean, components=components, scale=scale)


def _synthetic_whitening(
    dim: int,
    n_samples: int = 2000,
    n_components: int = 64,
    seed: int = 42,
) -> HashWhitening:
    """Fit a whitening transform from synthetic positive unit vectors.

    Draws *n_samples* vectors from an exponential distribution (mimicking the
    heavy-tailed, all-positive shape of L2-normalised power spectra), then
    normalises each to unit L2 norm and fits PCA whitening.
    Using a fixed *seed* makes the result fully deterministic.
    """
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=(n_samples, dim)).astype(np.float64)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vecs: list[NDArray[np.float64]] = list(raw / norms)
    return fit_hash_whitening(vecs, n_components=n_components)


@functools.lru_cache(maxsize=1)
def default_whitening() -> HashWhitening:
    """Return a PCA whitening transform sized for default :class:`~rinse_descriptor.RinseParams`.

    The transform is synthesised on the first call from pseudo-random positive
    unit vectors (exponential distribution, L2-normalised) that approximate
    the shape of real power-spectrum descriptors.  A fixed seed ensures it is
    fully deterministic and reproducible without shipping a data file.

    The result is cached; subsequent calls return the same object.

    Raises
    ------
    ValueError
        If the descriptor passed to :func:`descriptor_hash` was not computed
        with default :class:`~rinse_descriptor.RinseParams`.
    """
    from ._descriptor import RinseParams

    dim = RinseParams().descriptor_length
    return _synthetic_whitening(dim)


def descriptor_hash(
    x: NDArray[np.float64],
    n_words: int = DEFAULT_HASH_WORDS,
    *,
    whitening: HashWhitening | None | bool = None,
) -> str:
    """Convert a descriptor vector to a pronounceable proquint hash string.

    Parameters
    ----------
    x:
        Flat descriptor vector (any length).  Pass the output of
        :func:`~rinse_descriptor.power_spectrum_to_vector` or use
        ``descriptor(..., flatten=True)``.
    n_words:
        Number of proquint words in the output.
        Each word encodes 16 bits, so ``n_words=5`` → 80 hash bits total.
        The output string has ``5 * n_words + (n_words - 1)`` characters
        (words joined by ``"-"``).
    whitening:
        Controls PCA whitening before hashing:

        * ``None`` (default) — use :func:`default_whitening`, which is a
          transform synthesised from pseudo-random positive unit vectors
          matching the shape of power-spectrum descriptors.
        * A :class:`HashWhitening` instance — use that transform instead
          (e.g. one fitted from real structures via :func:`fit_hash_whitening`).
        * ``False`` — disable whitening entirely and use bare ``log1p``
          preprocessing.

    Returns
    -------
    str
        Hyphen-separated proquint words, e.g. ``"lusab-babad-gutih-tugad-mudof"``.

    Notes
    -----
    With whitening (default): ``log1p → centre → PCA project → scale``
    produces a spherically distributed input so that every proquint character
    is equally reachable at every position.

    Without whitening (``whitening=False``): the vector is transformed by
    ``log1p`` only, which reduces but does not eliminate projection bias.

    The projection matrix **W** is drawn deterministically (seed 0) with
    shape ``(n_bits, input_dim)`` where *input_dim* is the post-whitening
    dimension, so hashes computed with different whitening transforms are
    **not** comparable.
    """
    vec = np.asarray(x, dtype=np.float64).ravel()
    if vec.shape[0] == 0:
        raise ValueError("Descriptor vector must not be empty.")

    if whitening is False:
        # Bare log1p fallback — no whitening
        vec = np.log1p(np.abs(vec)) * np.sign(vec)
    else:
        w: HashWhitening = default_whitening() if whitening is None else whitening  # type: ignore[assignment]
        vec = w.transform(vec)

    n_bits = n_words * _BITS_PER_WORD
    W = np.random.default_rng(0).standard_normal((n_bits, vec.shape[0]))  # (n_bits, dim)
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
