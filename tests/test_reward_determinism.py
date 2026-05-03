"""Tests for reward determinism and agent-specific reward functions."""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.reward import normalize_reward, compute_raw_reward


class TestRewardNormalization:
    def test_positive_stays_positive(self):
        assert normalize_reward(1.0) > 0.0

    def test_negative_stays_negative(self):
        assert normalize_reward(-1.0) < 0.0

    def test_zero_maps_to_zero(self):
        assert abs(normalize_reward(0.0)) < 1e-10

    def test_bounded(self):
        for val in [100, -100, 50, -50]:
            normed = normalize_reward(float(val))
            assert -1.0 <= normed <= 1.0

    def test_monotonic(self):
        vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        normed = [normalize_reward(v) for v in vals]
        for i in range(len(normed) - 1):
            assert normed[i] <= normed[i+1], f"Monotonicity violated at {vals[i]} -> {vals[i+1]}"


class TestDirectionalReward:
    def test_correct_long_beats_wrong(self):
        r_correct = compute_raw_reward(profit=0.001, drawdown=0.0, volatility=0.01,
                                        sharpe=0.5, trade_count=1, direction=1, price_trend=0.01)
        r_wrong = compute_raw_reward(profit=0.001, drawdown=0.0, volatility=0.01,
                                      sharpe=0.5, trade_count=1, direction=1, price_trend=-0.01)
        assert r_correct > r_wrong

    def test_correct_short_beats_wrong(self):
        r_correct = compute_raw_reward(profit=0.001, drawdown=0.0, volatility=0.01,
                                        sharpe=0.5, trade_count=1, direction=2, price_trend=-0.01)
        r_wrong = compute_raw_reward(profit=0.001, drawdown=0.0, volatility=0.01,
                                      sharpe=0.5, trade_count=1, direction=1, price_trend=-0.01)
        assert r_correct > r_wrong

    def test_hold_no_directional_bonus(self):
        r = compute_raw_reward(profit=0.0, drawdown=0.0, volatility=0.01,
                                sharpe=0.0, trade_count=0, direction=0, price_trend=0.01)
        # Hold should not get a large positive/negative directional bonus
        assert abs(r) < 1.0


class TestRewardDeterminism:
    def test_same_inputs_same_reward(self):
        kwargs = dict(profit=0.005, drawdown=0.03, volatility=0.02,
                      sharpe=1.2, trade_count=2, direction=1, price_trend=0.005)
        r1 = compute_raw_reward(**kwargs)
        r2 = compute_raw_reward(**kwargs)
        assert r1 == r2, "Reward must be deterministic"
