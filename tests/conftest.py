"""Shared pytest fixtures for QuantHive test suite."""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def seeded_rng():
    return np.random.default_rng(42)


@pytest.fixture
def base_obs_25():
    """A sample 25-dim observation (base + regime indicator)."""
    return np.random.default_rng(42).standard_normal(25).astype(np.float32)


@pytest.fixture
def rm_obs_25(base_obs_25):
    return base_obs_25


@pytest.fixture
def pm_obs_28(base_obs_25):
    rm_msg = np.array([0.5, 1.0, 0.0], dtype=np.float32)
    return np.concatenate([base_obs_25, rm_msg])


@pytest.fixture
def trader_obs_30(pm_obs_28):
    pm_msg = np.array([0.6, 0.0], dtype=np.float32)
    return np.concatenate([pm_obs_28, pm_msg])
