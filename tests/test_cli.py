"""Tests for CLI comparison distance metrics."""

from __future__ import annotations

import numpy as np
import pytest
from rinse_descriptor._cli import _compare_distance


def test_euclidean_compare_distance() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    assert _compare_distance(a, b, "euclidean") == pytest.approx(np.sqrt(2.0))


def test_cosine_compare_distance() -> None:
    a = np.array([1.0, 0.0])
    orthogonal = np.array([0.0, 1.0])
    same_direction = np.array([3.0, 0.0])
    opposite_direction = np.array([-1.0, 0.0])

    assert _compare_distance(a, orthogonal, "cosine") == pytest.approx(1.0)
    assert _compare_distance(a, same_direction, "cosine") == pytest.approx(0.0)
    assert _compare_distance(a, opposite_direction, "cosine") == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.zeros(2), np.ones(2)),
        (np.ones(2), np.zeros(2)),
    ],
)
def test_cosine_compare_distance_rejects_zero_vectors(a: np.ndarray, b: np.ndarray) -> None:
    with pytest.raises(ValueError, match="undefined for a zero-length descriptor"):
        _compare_distance(a, b, "cosine")
