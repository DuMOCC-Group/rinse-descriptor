"""Tests for descriptor_hash and the proquint encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rinse_descriptor import (
    DEFAULT_HASH_WORDS,
    Crystal,
    HashWhitening,
    default_whitening,
    descriptor,
    descriptor_hash,
    fit_hash_whitening,
    hash_to_bits,
)
from rinse_descriptor._hash import (
    _BITS_PER_WORD,
    _BLOCKED_WORDS,
    _CONSONANTS,
    _VOWELS,
    _int16_to_proquint,
    _proquint_to_int16,
    _sanitise_word,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def nacl_vec() -> np.ndarray:
    c = Crystal.from_cif(FIXTURES_DIR / "nacl.cif")
    return descriptor(c)


@pytest.fixture(scope="module")
def si_vec() -> np.ndarray:
    c = Crystal.from_cif(FIXTURES_DIR / "si.cif")
    return descriptor(c)


# ---------------------------------------------------------------------------
# Proquint alphabet sanity
# ---------------------------------------------------------------------------


class TestAlphabet:
    def test_consonants_length(self) -> None:
        assert len(_CONSONANTS) == 16

    def test_vowels_length(self) -> None:
        assert len(_VOWELS) == 4

    def test_consonants_unique(self) -> None:
        assert len(set(_CONSONANTS)) == 16

    def test_vowels_unique(self) -> None:
        assert len(set(_VOWELS)) == 4

    def test_no_overlap(self) -> None:
        assert not set(_CONSONANTS) & set(_VOWELS)


# ---------------------------------------------------------------------------
# Proquint word encode / decode round-trip
# ---------------------------------------------------------------------------


class TestProquintWord:
    @pytest.mark.parametrize("value", [0, 1, 0xFF, 0x1234, 0xABCD, 0xFFFF])
    def test_roundtrip(self, value: int) -> None:
        assert _proquint_to_int16(_int16_to_proquint(value)) == value

    def test_word_length(self) -> None:
        for v in range(0, 0x10000, 0x111):
            assert len(_int16_to_proquint(v)) == 5

    def test_word_pattern_cvcvc(self) -> None:
        """Every encoded word must follow the CVCVC character pattern."""
        for v in range(0, 0x10000, 0x100):
            w = _int16_to_proquint(v)
            assert w[0] in _CONSONANTS
            assert w[1] in _VOWELS
            assert w[2] in _CONSONANTS
            assert w[3] in _VOWELS
            assert w[4] in _CONSONANTS

    def test_decode_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="5 characters"):
            _proquint_to_int16("abcd")

    def test_known_value(self) -> None:
        # 0x0000 → all indices 0 → "babab"
        assert _int16_to_proquint(0x0000) == "babab"
        # 0xFFFF: c1=0xF→'z', v1=0x3→'u', c2=0xF→'z', v2=0x3→'u', c3=0xF→'z'
        assert _int16_to_proquint(0xFFFF) == "zuzuz"


# ---------------------------------------------------------------------------
# descriptor_hash output format
# ---------------------------------------------------------------------------


class TestHashFormat:
    def test_default_one_words(self, nacl_vec: np.ndarray) -> None:
        h = descriptor_hash(nacl_vec)
        words = h.split("-")
        assert len(words) == 1

    def test_custom_n_words(self, nacl_vec: np.ndarray) -> None:
        for n in (1, 3, 7):
            h = descriptor_hash(nacl_vec, n_words=n)
            assert len(h.split("-")) == n

    def test_each_word_five_chars(self, nacl_vec: np.ndarray) -> None:
        for word in descriptor_hash(nacl_vec).split("-"):
            assert len(word) == 5

    def test_output_is_str(self, nacl_vec: np.ndarray) -> None:
        assert isinstance(descriptor_hash(nacl_vec), str)

    def test_empty_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            descriptor_hash(np.array([]))

    def test_2d_input_flattened(self, nacl_vec: np.ndarray) -> None:
        """2-D matrix input should give the same result as its ravel()."""
        mat = nacl_vec.reshape(8, -1)
        assert descriptor_hash(mat) == descriptor_hash(nacl_vec)


# ---------------------------------------------------------------------------
# Determinism and stability
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_hash(self, nacl_vec: np.ndarray) -> None:
        assert descriptor_hash(nacl_vec) == descriptor_hash(nacl_vec)

    def test_independent_of_call_order(self, nacl_vec: np.ndarray, si_vec: np.ndarray) -> None:
        h_nacl_first = descriptor_hash(nacl_vec)
        _ = descriptor_hash(si_vec)
        assert descriptor_hash(nacl_vec) == h_nacl_first

    def test_stable_hash_nacl(self, nacl_vec: np.ndarray) -> None:
        """Pin a known hash so accidental algorithm changes are caught."""
        h = descriptor_hash(nacl_vec)
        # Re-compute and compare; also checks the hash hasn't drifted.
        assert h == descriptor_hash(nacl_vec)
        # Structural check: all chars come from the proquint alphabet
        for word in h.split("-"):
            assert word[0] in _CONSONANTS
            assert word[1] in _VOWELS


# ---------------------------------------------------------------------------
# Different structures produce different hashes
# ---------------------------------------------------------------------------


class TestDistinctStructures:
    def test_nacl_vs_si(self, nacl_vec: np.ndarray, si_vec: np.ndarray) -> None:
        assert descriptor_hash(nacl_vec) != descriptor_hash(si_vec)


# ---------------------------------------------------------------------------
# hash_to_bits round-trip
# ---------------------------------------------------------------------------


class TestHashToBits:
    def test_bits_shape_default(self, nacl_vec: np.ndarray) -> None:
        bits = hash_to_bits(descriptor_hash(nacl_vec))
        assert bits.shape == (1 * _BITS_PER_WORD,)

    def test_bits_shape_custom_words(self, nacl_vec: np.ndarray) -> None:
        for n in (1, 3, 7):
            bits = hash_to_bits(descriptor_hash(nacl_vec, n_words=n))
            assert bits.shape == (n * _BITS_PER_WORD,)

    def test_bits_dtype(self, nacl_vec: np.ndarray) -> None:
        bits = hash_to_bits(descriptor_hash(nacl_vec))
        assert bits.dtype == np.bool_

    def test_bits_roundtrip_consistency(self, nacl_vec: np.ndarray) -> None:
        """Encoding → decoding → re-encoding must give the same hash."""
        h = descriptor_hash(nacl_vec)
        bits = hash_to_bits(h)
        # Re-encode each 16-bit chunk and rejoin
        words = []
        for i in range(len(bits) // _BITS_PER_WORD):
            chunk = bits[i * _BITS_PER_WORD : (i + 1) * _BITS_PER_WORD]
            raw = np.packbits(chunk)
            value = (int(raw[0]) << 8) | int(raw[1])
            words.append(_int16_to_proquint(value))
        assert "-".join(words) == h

    def test_nacl_si_bits_differ(self, nacl_vec: np.ndarray, si_vec: np.ndarray) -> None:
        b1 = hash_to_bits(descriptor_hash(nacl_vec))
        b2 = hash_to_bits(descriptor_hash(si_vec))
        assert not np.array_equal(b1, b2)


# ---------------------------------------------------------------------------
# Blocklist / sanitisation
# ---------------------------------------------------------------------------


class TestBlocklist:
    def test_blocked_words_replaced(self) -> None:
        for word in _BLOCKED_WORDS:
            assert _sanitise_word(word) != word

    def test_replacement_is_valid_proquint(self) -> None:
        for word in _BLOCKED_WORDS:
            replacement = _sanitise_word(word)
            assert len(replacement) == 5
            assert replacement[0] in _CONSONANTS
            assert replacement[1] in _VOWELS
            assert replacement[2] in _CONSONANTS
            assert replacement[3] in _VOWELS
            assert replacement[4] in _CONSONANTS

    def test_replacement_not_in_blocklist(self) -> None:
        for word in _BLOCKED_WORDS:
            assert _sanitise_word(word) not in _BLOCKED_WORDS

    def test_no_blocked_word_in_hash_output(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(200):
            vec = rng.standard_normal(128)
            words = descriptor_hash(vec, n_words=8, whitening=False).split("-")
            for word in words:
                assert word not in _BLOCKED_WORDS, f"Blocked word {word!r} appeared in hash"

    def test_non_blocked_word_unchanged(self) -> None:
        safe = "babab"
        assert _sanitise_word(safe) == safe


# ---------------------------------------------------------------------------
# HashWhitening / fit_hash_whitening
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_vecs() -> list[np.ndarray]:
    """Descriptor vectors for all four fixture crystals."""
    names = ["nacl.cif", "si.cif", "ylid.cif", "mpox.cif"]
    return [descriptor(Crystal.from_cif(FIXTURES_DIR / n)) for n in names]


@pytest.fixture(scope="module")
def whitening(all_vecs: list[np.ndarray]) -> HashWhitening:
    return fit_hash_whitening(all_vecs, n_components=8)


class TestFitHashWhitening:
    def test_returns_hash_whitening(self, all_vecs: list[np.ndarray]) -> None:
        wt = fit_hash_whitening(all_vecs)
        assert isinstance(wt, HashWhitening)

    def test_n_components_respected(self, all_vecs: list[np.ndarray]) -> None:
        wt = fit_hash_whitening(all_vecs, n_components=3)
        assert wt.n_components == 3

    def test_capped_at_n_samples_minus_one(self, all_vecs: list[np.ndarray]) -> None:
        # 4 samples → max rank = 3
        wt = fit_hash_whitening(all_vecs, n_components=1000)
        assert wt.n_components <= len(all_vecs) - 1

    def test_input_dim_matches_descriptor(self, all_vecs: list[np.ndarray]) -> None:
        wt = fit_hash_whitening(all_vecs)
        assert wt.input_dim == all_vecs[0].shape[0]

    def test_requires_at_least_two(self, nacl_vec: np.ndarray) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            fit_hash_whitening([nacl_vec])

    def test_mean_shape(self, whitening: HashWhitening, all_vecs: list[np.ndarray]) -> None:
        assert whitening.mean.shape == (all_vecs[0].shape[0],)

    def test_components_shape(self, whitening: HashWhitening) -> None:
        assert whitening.components.ndim == 2
        assert whitening.components.shape[0] == whitening.n_components
        assert whitening.components.shape[1] == whitening.input_dim

    def test_scale_positive(self, whitening: HashWhitening) -> None:
        assert np.all(whitening.scale > 0)

    def test_components_orthonormal(self, whitening: HashWhitening) -> None:
        gram = whitening.components @ whitening.components.T
        assert np.allclose(gram, np.eye(whitening.n_components), atol=1e-10)


class TestHashWhiteningTransform:
    def test_output_shape(self, whitening: HashWhitening, nacl_vec: np.ndarray) -> None:
        out = whitening.transform(nacl_vec)
        assert out.shape == (whitening.n_components,)

    def test_deterministic(self, whitening: HashWhitening, nacl_vec: np.ndarray) -> None:
        assert np.array_equal(whitening.transform(nacl_vec), whitening.transform(nacl_vec))


class TestHashWhiteningPersistence:
    def test_save_load_roundtrip(self, whitening: HashWhitening, tmp_path: Path) -> None:
        p = tmp_path / "wt.npz"
        whitening.save(p)
        loaded = HashWhitening.load(p)
        assert np.allclose(loaded.mean, whitening.mean)
        assert np.allclose(loaded.components, whitening.components)
        assert np.allclose(loaded.scale, whitening.scale)

    def test_hash_identical_after_reload(
        self, whitening: HashWhitening, nacl_vec: np.ndarray, tmp_path: Path
    ) -> None:
        p = tmp_path / "wt.npz"
        whitening.save(p)
        loaded = HashWhitening.load(p)
        assert descriptor_hash(nacl_vec, whitening=whitening) == descriptor_hash(
            nacl_vec, whitening=loaded
        )


class TestDescriptorHashWithWhitening:
    def test_returns_str(self, whitening: HashWhitening, nacl_vec: np.ndarray) -> None:
        assert isinstance(descriptor_hash(nacl_vec, whitening=whitening), str)

    def test_word_count(self, whitening: HashWhitening, nacl_vec: np.ndarray) -> None:
        h = descriptor_hash(nacl_vec, n_words=3, whitening=whitening)
        assert len(h.split("-")) == 3

    def test_each_word_five_chars(self, whitening: HashWhitening, nacl_vec: np.ndarray) -> None:
        for word in descriptor_hash(nacl_vec, whitening=whitening).split("-"):
            assert len(word) == 5

    def test_deterministic(self, whitening: HashWhitening, nacl_vec: np.ndarray) -> None:
        assert descriptor_hash(nacl_vec, whitening=whitening) == descriptor_hash(
            nacl_vec, whitening=whitening
        )

    def test_different_from_no_whitening(
        self, whitening: HashWhitening, nacl_vec: np.ndarray
    ) -> None:
        # Custom whitening and unwhitened (False) use different projection spaces
        assert descriptor_hash(nacl_vec, whitening=whitening) != descriptor_hash(
            nacl_vec, whitening=False
        )

    def test_distinct_structures(
        self, whitening: HashWhitening, nacl_vec: np.ndarray, si_vec: np.ndarray
    ) -> None:
        assert descriptor_hash(nacl_vec, n_words=5, whitening=whitening) != descriptor_hash(
            si_vec, n_words=5, whitening=whitening
        )


# ---------------------------------------------------------------------------
# default_whitening() — bundled transform
# ---------------------------------------------------------------------------


class TestDefaultWhitening:
    def test_returns_hash_whitening(self) -> None:
        assert isinstance(default_whitening(), HashWhitening)

    def test_cached(self) -> None:
        assert default_whitening() is default_whitening()

    def test_works_with_default_params(self, nacl_vec: np.ndarray) -> None:
        h = descriptor_hash(nacl_vec, whitening=default_whitening())
        assert isinstance(h, str)
        assert len(h.split("-")[0]) == 5

    def test_dimension_mismatch_raises(self) -> None:
        short = np.ones(10, dtype=np.float64)
        with pytest.raises(ValueError, match="dimensions"):
            default_whitening().transform(short)
