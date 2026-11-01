"""S0 acceptance: seeding utility is deterministic across RNG backends."""

import random

import numpy as np
import pytest

from src.seeds import set_all_seeds


def test_numpy_seeding_is_deterministic():
    set_all_seeds(42)
    a = np.random.rand(8)
    set_all_seeds(42)
    b = np.random.rand(8)
    assert np.array_equal(a, b)


def test_stdlib_random_seeding_is_deterministic():
    set_all_seeds(7)
    a = [random.random() for _ in range(8)]
    set_all_seeds(7)
    b = [random.random() for _ in range(8)]
    assert a == b


def test_different_seed_gives_different_stream():
    set_all_seeds(1)
    a = np.random.rand(8)
    set_all_seeds(2)
    b = np.random.rand(8)
    assert not np.array_equal(a, b)


def test_torch_seeding_when_available():
    torch = pytest.importorskip("torch")
    set_all_seeds(42)
    t1 = torch.rand(6)
    set_all_seeds(42)
    t2 = torch.rand(6)
    assert torch.equal(t1, t2)
