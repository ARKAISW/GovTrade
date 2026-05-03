"""
Market Regime Engine for QuantHive.

Generates synthetic market data with explicit, named regime transitions.
Each regime is calibrated to real historical market parameters and exposes
its label in the observation vector for regime-conditional governance.

Supports:
  - 8 standard market regimes (bull, bear, crash, sideways, etc.)
  - 4 adversarial stress scenarios (spoofing, delayed signals, etc.)
  - Markov-chain regime transitions with configurable transition matrix
  - Deterministic seeding for reproducible benchmarks
  - Out-of-distribution evaluation via held-out regime sets

Reference calibrations:
  bull_steady   → S&P 500 2017 (low-vol rally)
  crash         → March 2020 COVID crash
  flash_crash   → May 6, 2010 Flash Crash
  bubble_pop    → BTC Nov 2021 → Jun 2022
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Regime Definitions ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeParams:
    """Statistical parameters for a single market regime.

    Attributes:
        name:       Human-readable regime identifier.
        mu:         Annualized drift (expected return).
        sigma:      Annualized volatility.
        jump_prob:  Per-step probability of a jump event.
        jump_mean:  Mean of jump magnitude (log-normal).
        jump_std:   Std of jump magnitude.
        df:         Degrees of freedom for Student-t noise (lower = fatter tails).
        description: Brief description for documentation.
    """
    name: str
    mu: float
    sigma: float
    jump_prob: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    df: float = 30.0
    description: str = ""


# Standard regimes (calibrated to real market analogs)
REGIME_CATALOG: Dict[str, RegimeParams] = {
    "bull_steady": RegimeParams(
        name="bull_steady", mu=0.30, sigma=0.08, df=30,
        description="Low-volatility rally (S&P 500 2017)",
    ),
    "bull_volatile": RegimeParams(
        name="bull_volatile", mu=0.40, sigma=0.35, jump_prob=0.02, jump_std=0.04, df=5,
        description="High-volatility rally with jump risk (Crypto 2021)",
    ),
    "bear_steady": RegimeParams(
        name="bear_steady", mu=-0.20, sigma=0.15, jump_prob=0.01, jump_std=0.03, df=8,
        description="Orderly decline (Tech sector 2022)",
    ),
    "crash": RegimeParams(
        name="crash", mu=-0.80, sigma=0.60, jump_prob=0.05,
        jump_mean=-0.02, jump_std=0.10, df=3,
        description="Violent market crash (COVID March 2020)",
    ),
    "sideways_choppy": RegimeParams(
        name="sideways_choppy", mu=0.0, sigma=0.25, jump_prob=0.01, jump_std=0.03, df=6,
        description="Range-bound with high noise (2018-2019 markets)",
    ),
    "mean_revert": RegimeParams(
        name="mean_revert", mu=0.0, sigma=0.12, df=15,
        description="Mean-reverting, low-volatility sideways",
    ),
    "bubble_pop": RegimeParams(
        name="bubble_pop", mu=1.00, sigma=0.50, df=4,
        description="Parabolic rise then crash (BTC Nov 2021 → Jun 2022)",
    ),
    "flash_crash": RegimeParams(
        name="flash_crash", mu=-2.0, sigma=0.80, jump_prob=0.10,
        jump_mean=-0.05, jump_std=0.15, df=2,
        description="Extreme short-duration crash (May 6, 2010 Flash Crash)",
    ),
}

# Adversarial stress scenarios
ADVERSARIAL_CATALOG: Dict[str, RegimeParams] = {
    "spoofing": RegimeParams(
        name="spoofing", mu=0.05, sigma=0.30, jump_prob=0.08,
        jump_mean=-0.03, jump_std=0.08, df=4,
        description="False liquidity signals followed by sharp reversal",
    ),
    "delayed_signal": RegimeParams(
        name="delayed_signal", mu=-0.15, sigma=0.20, df=8,
        description="Indicator lag increases — signals arrive 3-5 steps late",
    ),
    "correlated_selloff": RegimeParams(
        name="correlated_selloff", mu=-0.60, sigma=0.45, jump_prob=0.04,
        jump_mean=-0.04, jump_std=0.06, df=3,
        description="Multi-asset synchronized crash",
    ),
    "cascading_liquidation": RegimeParams(
        name="cascading_liquidation", mu=-1.20, sigma=0.70, jump_prob=0.08,
        jump_mean=-0.06, jump_std=0.12, df=2,
        description="Forced selling cascade creating positive feedback loop",
    ),
}

# Unified catalog
ALL_REGIMES: Dict[str, RegimeParams] = {**REGIME_CATALOG, **ADVERSARIAL_CATALOG}

# Integer encoding for observation vector
REGIME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(ALL_REGIMES.keys())}
NUM_REGIMES: int = len(ALL_REGIMES)


# ─── Default Markov Transition Matrix ──────────────────────────────────────────

def _build_default_transition_matrix() -> np.ndarray:
    """Build a regime transition probability matrix.

    Rows = current regime, Cols = next regime.
    Diagonal = probability of staying in current regime.
    """
    n = len(REGIME_CATALOG)  # Only standard regimes transition naturally
    names = list(REGIME_CATALOG.keys())

    # Start with high self-transition (regime persistence)
    T = np.eye(n) * 0.70

    # Add realistic transition probabilities
    idx = {name: i for i, name in enumerate(names)}

    # Bull → can crash or go sideways
    T[idx["bull_steady"], idx["bear_steady"]] = 0.05
    T[idx["bull_steady"], idx["bull_volatile"]] = 0.10
    T[idx["bull_steady"], idx["sideways_choppy"]] = 0.10
    T[idx["bull_steady"], idx["bubble_pop"]] = 0.05

    # Bull volatile → can crash hard
    T[idx["bull_volatile"], idx["crash"]] = 0.10
    T[idx["bull_volatile"], idx["bear_steady"]] = 0.08
    T[idx["bull_volatile"], idx["sideways_choppy"]] = 0.07
    T[idx["bull_volatile"], idx["bull_steady"]] = 0.05

    # Bear → recovery or crash
    T[idx["bear_steady"], idx["crash"]] = 0.08
    if "recovery" in idx:
        T[idx["bear_steady"], idx["recovery"]] = 0.10
    T[idx["bear_steady"], idx["sideways_choppy"]] = 0.07
    T[idx["bear_steady"], idx["mean_revert"]] = 0.05

    # Crash → recovery or continued bear
    T[idx["crash"], idx["bear_steady"]] = 0.10
    T[idx["crash"], idx["sideways_choppy"]] = 0.10
    T[idx["crash"], idx["mean_revert"]] = 0.05
    T[idx["crash"], idx["bull_steady"]] = 0.05

    # Sideways → anything
    T[idx["sideways_choppy"], idx["bull_steady"]] = 0.08
    T[idx["sideways_choppy"], idx["bear_steady"]] = 0.08
    T[idx["sideways_choppy"], idx["mean_revert"]] = 0.07
    T[idx["sideways_choppy"], idx["bull_volatile"]] = 0.07

    # Mean revert → trend emergence
    T[idx["mean_revert"], idx["bull_steady"]] = 0.10
    T[idx["mean_revert"], idx["bear_steady"]] = 0.08
    T[idx["mean_revert"], idx["sideways_choppy"]] = 0.07
    T[idx["mean_revert"], idx["bull_volatile"]] = 0.05

    # Bubble pop → crash or bear
    T[idx["bubble_pop"], idx["crash"]] = 0.12
    T[idx["bubble_pop"], idx["bear_steady"]] = 0.10
    T[idx["bubble_pop"], idx["sideways_choppy"]] = 0.08

    # Flash crash → recovery or sideways
    T[idx["flash_crash"], idx["sideways_choppy"]] = 0.12
    T[idx["flash_crash"], idx["mean_revert"]] = 0.08
    T[idx["flash_crash"], idx["bull_steady"]] = 0.05
    T[idx["flash_crash"], idx["bear_steady"]] = 0.05

    # Normalize rows to sum to 1
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / (row_sums + 1e-10)

    return T


DEFAULT_TRANSITION_MATRIX = _build_default_transition_matrix()


# ─── Regime Engine ─────────────────────────────────────────────────────────────

class MarketRegimeEngine:
    """Generates synthetic market data with explicit regime labels.

    Usage:
        engine = MarketRegimeEngine(seed=42)
        df, regime_labels = engine.generate(
            n_steps=500,
            difficulty="hard",
        )
        # df: OHLCV DataFrame
        # regime_labels: list of regime name strings, one per step
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        transition_matrix: Optional[np.ndarray] = None,
        min_regime_duration: int = 30,
        max_regime_duration: int = 150,
    ):
        self.rng = np.random.default_rng(seed)
        self.transition_matrix = transition_matrix if transition_matrix is not None else DEFAULT_TRANSITION_MATRIX
        self.min_regime_duration = min_regime_duration
        self.max_regime_duration = max_regime_duration

    def generate(
        self,
        n_steps: int = 500,
        difficulty: str = "hard",
        initial_price: float = 50000.0,
        forced_regime: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Generate OHLCV data with regime labels.

        Args:
            n_steps: Number of time steps.
            difficulty: 'easy', 'medium', 'hard', or 'adversarial'.
            initial_price: Starting price.
            forced_regime: If set, use this single regime for all steps.

        Returns:
            (df, regime_labels) where df is OHLCV DataFrame and
            regime_labels is a list of regime names per step.
        """
        if forced_regime is not None:
            regime_schedule = self._fixed_schedule(n_steps, forced_regime)
        else:
            regime_schedule = self._build_regime_schedule(n_steps, difficulty)

        returns, regime_labels = self._generate_returns(n_steps, regime_schedule)
        df = self._returns_to_ohlcv(returns, initial_price, n_steps)

        return df, regime_labels

    def _get_regime_pool(self, difficulty: str) -> List[str]:
        """Get available regimes for a difficulty level."""
        if difficulty == "easy":
            return ["bull_steady", "mean_revert"]
        elif difficulty == "medium":
            return ["bull_steady", "sideways_choppy", "mean_revert",
                    "bull_volatile", "bear_steady"]
        elif difficulty == "hard":
            return list(REGIME_CATALOG.keys())
        elif difficulty == "adversarial":
            return list(ALL_REGIMES.keys())
        else:
            return list(REGIME_CATALOG.keys())

    def _build_regime_schedule(
        self, n_steps: int, difficulty: str,
    ) -> List[Tuple[str, int, int]]:
        """Build a schedule of (regime_name, start_step, end_step) tuples."""
        pool = self._get_regime_pool(difficulty)
        schedule: List[Tuple[str, int, int]] = []
        step = 0

        # Pick initial regime
        current_regime = self.rng.choice(pool)

        while step < n_steps:
            remaining = n_steps - step
            lo = min(self.min_regime_duration, remaining)
            hi = min(self.max_regime_duration, remaining)
            if lo >= hi:
                duration = remaining
            else:
                duration = int(self.rng.integers(lo, hi + 1))
            duration = max(duration, 1)
            end = min(step + duration, n_steps)
            schedule.append((current_regime, step, end))
            step = end

            if step < n_steps:
                # Transition to next regime
                current_regime = self._transition(current_regime, pool)

        return schedule

    def _fixed_schedule(
        self, n_steps: int, regime_name: str,
    ) -> List[Tuple[str, int, int]]:
        """Single-regime schedule for controlled experiments."""
        return [(regime_name, 0, n_steps)]

    def _transition(self, current: str, pool: List[str]) -> str:
        """Sample next regime from transition matrix, restricted to pool."""
        std_names = list(REGIME_CATALOG.keys())

        if current in std_names:
            idx = std_names.index(current)
            probs = self.transition_matrix[idx].copy()

            # Mask out regimes not in pool
            for i, name in enumerate(std_names):
                if name not in pool:
                    probs[i] = 0.0

            total = probs.sum()
            if total > 0:
                probs /= total
                next_idx = self.rng.choice(len(std_names), p=probs)
                return std_names[next_idx]

        # Fallback: uniform random from pool
        return self.rng.choice(pool)

    def _generate_returns(
        self,
        n_steps: int,
        schedule: List[Tuple[str, int, int]],
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate log returns and per-step regime labels."""
        all_returns = np.zeros(n_steps)
        regime_labels: List[str] = [""] * n_steps
        dt = 1.0 / (24 * 365)  # Hourly steps

        for regime_name, start, end in schedule:
            seg_len = end - start
            params = ALL_REGIMES[regime_name]

            # Fill regime labels
            for i in range(start, end):
                regime_labels[i] = regime_name

            # Generate returns for this segment
            segment_returns = self._generate_segment(params, seg_len, dt)
            all_returns[start:end] = segment_returns

        return all_returns, regime_labels

    def _generate_segment(
        self, params: RegimeParams, seg_len: int, dt: float,
    ) -> np.ndarray:
        """Generate returns for one regime segment."""
        # Fat-tailed noise via Student-t distribution
        noise = self.rng.standard_t(df=params.df, size=seg_len) * params.sigma * np.sqrt(dt)

        # Drift
        drift = (params.mu - 0.5 * params.sigma ** 2) * dt

        # Jump diffusion (Merton model)
        jump_mask = self.rng.random(seg_len) < params.jump_prob
        jumps = jump_mask * self.rng.normal(params.jump_mean, params.jump_std + 1e-10, seg_len)

        # Special handling for specific regime types
        if params.name == "bubble_pop":
            returns = self._generate_bubble_pop(seg_len, dt, params, noise, drift, jumps)
        elif params.name == "mean_revert":
            returns = self._generate_mean_revert(seg_len, dt, drift, noise, jumps)
        elif params.name == "flash_crash":
            returns = self._generate_flash_crash(seg_len, dt, params)
        elif params.name == "spoofing":
            returns = self._generate_spoofing(seg_len, dt, params)
        elif params.name == "cascading_liquidation":
            returns = self._generate_cascading_liquidation(seg_len, dt, params)
        else:
            returns = drift + noise + jumps

        return returns

    def _generate_bubble_pop(
        self, seg_len: int, dt: float, params: RegimeParams,
        noise: np.ndarray, drift: float, jumps: np.ndarray,
    ) -> np.ndarray:
        """Parabolic rise then crash."""
        midpoint = seg_len // 2
        returns = np.zeros(seg_len)

        # First half: accelerating rise
        accel = np.linspace(1.0, 3.0, midpoint)
        returns[:midpoint] = drift * accel + noise[:midpoint] * 0.5 + jumps[:midpoint]

        # Second half: crash
        returns[midpoint:] = -abs(drift) * 2.5 + noise[midpoint:] * 2.0 + jumps[midpoint:]

        # Add crash jumps
        crash_jumps = (self.rng.random(seg_len - midpoint) > 0.9) * \
                      self.rng.normal(-0.05, 0.08, seg_len - midpoint)
        returns[midpoint:] += crash_jumps

        return returns

    def _generate_mean_revert(
        self, seg_len: int, dt: float, drift: float,
        noise: np.ndarray, jumps: np.ndarray,
    ) -> np.ndarray:
        """Mean-reverting price process (Ornstein-Uhlenbeck overlay)."""
        raw = drift + noise + jumps
        cumulative = np.cumsum(raw)
        reversion = -0.05 * cumulative * dt
        return raw + reversion

    def _generate_flash_crash(
        self, seg_len: int, dt: float, params: RegimeParams,
    ) -> np.ndarray:
        """Extreme short-duration crash followed by partial recovery."""
        returns = np.zeros(seg_len)
        crash_duration = min(10, seg_len // 3)
        recovery_duration = seg_len - crash_duration

        # Crash phase: extreme negative returns
        crash_noise = self.rng.standard_t(df=2, size=crash_duration)
        returns[:crash_duration] = -0.03 + crash_noise * 0.02

        # Recovery phase: partial bounce
        recovery_noise = self.rng.standard_t(df=6, size=recovery_duration)
        returns[crash_duration:] = 0.005 + recovery_noise * params.sigma * np.sqrt(dt) * 0.5

        return returns

    def _generate_spoofing(
        self, seg_len: int, dt: float, params: RegimeParams,
    ) -> np.ndarray:
        """False signals followed by reversals (adversarial)."""
        returns = np.zeros(seg_len)
        noise = self.rng.standard_t(df=params.df, size=seg_len) * params.sigma * np.sqrt(dt)

        # Create fake trend then reverse
        phase_len = max(5, seg_len // 6)
        for i in range(0, seg_len, phase_len * 2):
            # Fake uptrend
            end_up = min(i + phase_len, seg_len)
            returns[i:end_up] = 0.01 + noise[i:end_up] * 0.5

            # Sharp reversal
            end_down = min(end_up + phase_len, seg_len)
            returns[end_up:end_down] = -0.025 + noise[end_up:end_down]

        return returns

    def _generate_cascading_liquidation(
        self, seg_len: int, dt: float, params: RegimeParams,
    ) -> np.ndarray:
        """Positive feedback selling cascade."""
        returns = np.zeros(seg_len)
        noise = self.rng.standard_t(df=params.df, size=seg_len) * params.sigma * np.sqrt(dt)

        # Cascading effect: each step's loss amplifies the next
        cumulative_stress = 0.0
        for i in range(seg_len):
            base_return = params.mu * dt + noise[i]
            cascade_effect = -0.002 * max(cumulative_stress, 0)  # Positive feedback
            returns[i] = base_return + cascade_effect

            # Stress accumulates from negative returns
            if returns[i] < 0:
                cumulative_stress += abs(returns[i]) * 5.0
            else:
                cumulative_stress *= 0.95  # Slow decay

        return returns

    def _returns_to_ohlcv(
        self, returns: np.ndarray, initial_price: float, n_steps: int,
    ) -> pd.DataFrame:
        """Convert log returns to OHLCV DataFrame."""
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate realistic intrabar noise
        intrabar_noise = self.rng.normal(0, 0.003, n_steps)
        high_noise = np.abs(self.rng.normal(0, 0.008, n_steps))
        low_noise = np.abs(self.rng.normal(0, 0.008, n_steps))

        # Volume: correlated with absolute returns
        base_volume = self.rng.integers(100_000_000, 500_000_000, n_steps).astype(float)
        abs_rets = np.abs(returns)
        max_ret = abs_rets.max() + 1e-10
        vol_multiplier = 1.0 + 10.0 * (abs_rets / max_ret)
        volume = (base_volume * vol_multiplier).astype(int)

        df = pd.DataFrame({
            "open": prices * (1 + intrabar_noise),
            "high": prices * (1 + high_noise),
            "low": prices * (1 - low_noise),
            "close": prices,
            "volume": volume,
        }, index=pd.date_range("2024-01-01", periods=n_steps, freq="h"))

        df.index.name = "date"
        return df
