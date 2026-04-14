"""Shared test fixtures."""
import sys
from pathlib import Path

# Make project root importable so tests can `import models`, `import losses`, etc.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import torch


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """All tests run on CPU — model tests should be small enough to not need GPU."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def rng_seed() -> int:
    return 12345


@pytest.fixture(autouse=True)
def _deterministic(rng_seed: int):
    """Every test starts with a fresh deterministic torch RNG."""
    torch.manual_seed(rng_seed)
