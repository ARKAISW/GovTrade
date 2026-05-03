"""Tests for the Market Regime Engine."""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.regime_engine import (
    MarketRegimeEngine, REGIME_CATALOG, ADVERSARIAL_CATALOG, ALL_REGIMES, REGIME_TO_ID
)


class TestRegimeCatalog:
    def test_all_standard_regimes_defined(self):
        expected = {"bull_steady", "bull_volatile", "bear_steady", "crash",
                    "sideways_choppy", "mean_revert", "bubble_pop", "flash_crash"}
        assert expected == set(REGIME_CATALOG.keys())

    def test_all_adversarial_regimes_defined(self):
        expected = {"spoofing", "delayed_signal", "correlated_selloff", "cascading_liquidation"}
        assert expected == set(ADVERSARIAL_CATALOG.keys())

    def test_regime_to_id_unique(self):
        ids = list(REGIME_TO_ID.values())
        assert len(ids) == len(set(ids)), "Regime IDs must be unique"


class TestMarketRegimeEngine:
    def test_deterministic_seeding(self):
        e1 = MarketRegimeEngine(seed=42)
        df1, labels1 = e1.generate(n_steps=100, difficulty="hard")
        e2 = MarketRegimeEngine(seed=42)
        df2, labels2 = e2.generate(n_steps=100, difficulty="hard")
        np.testing.assert_array_almost_equal(df1["close"].values, df2["close"].values)
        assert labels1 == labels2

    def test_output_shape(self):
        engine = MarketRegimeEngine(seed=0)
        df, labels = engine.generate(n_steps=200, difficulty="medium")
        assert len(df) == 200
        assert len(labels) == 200
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}

    def test_labels_are_valid_regime_names(self):
        engine = MarketRegimeEngine(seed=7)
        _, labels = engine.generate(n_steps=300, difficulty="adversarial")
        for lbl in labels:
            assert lbl in ALL_REGIMES, f"Unknown regime label: {lbl}"

    def test_forced_regime(self):
        engine = MarketRegimeEngine(seed=1)
        _, labels = engine.generate(n_steps=100, forced_regime="crash")
        assert all(lbl == "crash" for lbl in labels)

    def test_easy_only_uses_easy_regimes(self):
        engine = MarketRegimeEngine(seed=5)
        _, labels = engine.generate(n_steps=200, difficulty="easy")
        allowed = {"bull_steady", "mean_revert"}
        for lbl in labels:
            assert lbl in allowed, f"Easy difficulty used non-easy regime: {lbl}"

    def test_flash_crash_produces_drawdown(self):
        engine = MarketRegimeEngine(seed=10)
        df, _ = engine.generate(n_steps=100, forced_regime="flash_crash")
        prices = df["close"].values
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / (peak + 1e-10)
        assert drawdown.max() > 0.10, "Flash crash should produce significant drawdown"

    def test_prices_positive(self):
        engine = MarketRegimeEngine(seed=99)
        df, _ = engine.generate(n_steps=500, difficulty="hard")
        assert (df["close"] > 0).all(), "Prices must remain positive"

    def test_high_above_low(self):
        engine = MarketRegimeEngine(seed=3)
        df, _ = engine.generate(n_steps=200, difficulty="hard")
        assert (df["high"] >= df["low"]).all(), "High must be >= Low"
