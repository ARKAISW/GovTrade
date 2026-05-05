"""
Reward computation and normalization for the trading environment.
All rewards and grades are normalized to [0, 1].
"""

import numpy as np
from typing import Dict
import json
import re


# Default reward component weights
DEFAULT_WEIGHTS = {
    "profit": 1.0,
    "drawdown": 0.5,
    "volatility": 0.3,
    "sharpe": 0.5,
    "overtrading": 0.1,
    "hold_penalty": 0.01,
    "directional_bonus": 0.3,
}

# Normalization: tanh scale factor (higher = sharper gradient near zero)
DEFAULT_NORM_SCALE = 10.0


def compute_raw_reward(
    profit: float,
    drawdown: float,
    volatility: float,
    sharpe: float,
    trade_count: int,
    weights: Dict[str, float] | None = None,
    direction: int = 0,
    price_trend: float = 0.0,
) -> float:
    """
    Compute the raw (un-normalized) reward signal.

    The profit signal is amplified (×1000) so single-step PnL fractions
    produce meaningful gradient.  A small hold-penalty discourages the
    model from always choosing direction=0, and a directional bonus
    rewards matching the market trend.

    Args:
        profit: Change in portfolio value (as fraction of initial).
        drawdown: Current max drawdown [0, 1].
        volatility: Return standard deviation.
        sharpe: Sharpe ratio of returns.
        trade_count: Number of trades executed this step.
        weights: Component weights (uses defaults if None).
        direction: Action direction (0=Hold, 1=Buy, 2=Sell).
        price_trend: Signed price change fraction for the step.

    Returns:
        Raw reward (float, unbounded).
    """
    w = weights or DEFAULT_WEIGHTS

    # Amplify per-step profit so it's not buried in noise
    profit_signal = w["profit"] * profit * 100.0

    # Penalties
    dd_penalty = w["drawdown"] * drawdown
    vol_penalty = w["volatility"] * volatility
    overtrade_penalty = w["overtrading"] * (trade_count / 10.0)

    # Bonuses
    sharpe_bonus = w["sharpe"] * np.tanh(sharpe)

    # Hold penalty: small cost for doing nothing
    hold_pen = 0.0  # Hold penalty is now progressive, handled in env

    # Directional correctness: reward matching the trend
    dir_bonus = 0.0
    w_dir = w.get("directional_bonus", 0.3)
    if direction == 1 and price_trend > 0:       # Bought into uptrend
        dir_bonus = w_dir * min(abs(price_trend) * 100, 1.0)
    elif direction == 2 and price_trend < 0:     # Sold into downtrend
        dir_bonus = w_dir * min(abs(price_trend) * 100, 1.0)
    elif direction != 0:                         # Wrong direction
        dir_bonus = -w_dir * 0.5

    reward = (
        profit_signal
        - dd_penalty
        - vol_penalty
        + sharpe_bonus
        - overtrade_penalty
        - hold_pen
        + dir_bonus
    )
    return float(reward)


def normalize_reward(
    raw: float,
    scale: float | None = None,
) -> float:
    """
    Normalize reward to [-1, 1] using tanh scaling.

    This preserves the sign (positive = good, negative = bad) and
    provides smooth gradient everywhere, unlike the old min-max clip
    which collapsed everything to ~0.5.
    """
    s = float(scale if scale is not None else DEFAULT_NORM_SCALE)
    return float(np.tanh(raw / s))


def compute_grade(metrics: Dict[str, float]) -> float:
    """
    Compute the final evaluation grade [0, 1].

    grade = 0.4 * normalized_profit
          + 0.3 * normalized_sharpe
          + 0.3 * (1 - normalized_drawdown)

    All input metrics must already be in [0, 1].
    """
    profit = np.clip(metrics.get("profit", 0.0), 0.0, 1.0)
    sharpe = np.clip(metrics.get("sharpe", 0.0), 0.0, 1.0)
    drawdown = np.clip(metrics.get("drawdown", 0.0), 0.0, 1.0)

    grade = (
        0.4 * profit
        + 0.3 * sharpe
        + 0.3 * (1.0 - drawdown)
    )
    return float(np.clip(grade, 0.0, 1.0))


def _extract_json_action(completion: str):
    match = re.search(r"<action>\s*({.*?})\s*</action>", completion, re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(1))


def _extract_prompt_state(prompt: str):
    json_match = re.search(r'"state"\s*:\s*\[(.*?)\]', prompt, re.DOTALL)
    if json_match:
        return [float(x.strip()) for x in json_match.group(1).split(",") if x.strip()]

    plain_match = re.search(r"State:\s*\[(.*?)\]", prompt, re.DOTALL)
    if plain_match:
        return [float(x.strip()) for x in plain_match.group(1).split(",") if x.strip()]

    return None


def _extract_signal_value(prompt: str, key: str):
    json_match = re.search(rf'"{key}"\s*:\s*(-?[\d\.]+)', prompt)
    if json_match:
        return float(json_match.group(1))

    plain_match = re.search(rf"{key}\s*[:=]\s*(-?[\d\.]+)", prompt)
    if plain_match:
        return float(plain_match.group(1))

    return None


# ──────────────────────────────────────────────
# GRPO Verifier Functions (Expert Optimized)
# ──────────────────────────────────────────────

def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Strict format and reasoning length check."""
    rewards = []
    for completion in completions:
        try:
            if "<thought>" not in completion or "</thought>" not in completion or "<action>" not in completion or "</action>" not in completion:
                rewards.append(0.0)
                continue
            
            thought = completion.split("<thought>")[1].split("</thought>")[0].strip()
            if len(thought) < 10:
                rewards.append(0.2) 
                continue

            if _extract_json_action(completion) is not None:
                rewards.append(1.0)
            else:
                rewards.append(0.4)
        except Exception:
            rewards.append(0.0)
    return rewards

def alignment_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Ensures the <thought> matches the generated <action>.
    This enforces internal reasoning consistency.
    """
    rewards = []
    for completion in completions:
        try:
            data = _extract_json_action(completion)
            direction = int(data.get("direction", 0)) if data else 0
            
            thought = completion.split("<thought>")[1].split("</thought>")[0].lower()
            
            score = 0.0
            if direction == 1 and ("buy" in thought or "long" in thought or "bull" in thought or "up" in thought):
                score = 1.0
            elif direction == 2 and ("sell" in thought or "short" in thought or "bear" in thought or "down" in thought):
                score = 1.0
            elif direction == 0 and ("hold" in thought or "wait" in thought or "neutral" in thought):
                score = 0.6
                
            rewards.append(score)
        except Exception:
            rewards.append(0.0)
    return rewards

def risk_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Safety Constraint: Position limits and Stop-Loss presence."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            limit = _extract_signal_value(prompt, "position_limit")
            if limit is None:
                limit = _extract_signal_value(prompt, "risk")
            if limit is None:
                limit = 1.0
            
            data = _extract_json_action(completion)
            if data is not None:
                size = float(data.get("size", 0.0))
                
                # Reward 1: Under limit
                score = 0.7 if size <= limit else 0.0
                rewards.append(score)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def profit_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Simulated PnL: Checks if the action (direction) matches the actual
    future price trend provided in the hidden 'future_return' dataset column.
    """
    rewards = []
    future_returns = kwargs.get("future_return", [0.0] * len(prompts))
    
    for prompt, completion, f_ret in zip(prompts, completions, future_returns):
        try:
            data = _extract_json_action(completion)
            if data is None:
                rewards.append(0.0)
                continue
            direction = int(data.get("direction", 0))
            size = float(data.get("size", 0.0))

            if direction in (1, 2) and size < 0.01:
                rewards.append(0.0)
                continue

            is_up_trend = f_ret > 0.001
            is_down_trend = f_ret < -0.001
            
            if direction == 1 and is_up_trend: # Buy in uptrend
                rewards.append(1.0)
            elif direction == 2 and is_down_trend: # Sell in downtrend
                rewards.append(1.0)
            elif direction == 0:
                rewards.append(1.0 if not is_up_trend and not is_down_trend else 0.1)
            else: # Wrong direction
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def governance_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Self-regulation verifier: rewards actions that would pass governance
    without intervention.

    An agent that self-regulates by proposing compliant executable actions
    scores higher than one that blindly maximizes size and forces clamps.

    Scoring rubric (0-1):
      +0.40  Action has valid JSON with size ≤ governance limit.
      +0.20  Size uses ≤ 80 % of limit (conservative, professional).
      +0.20  Direction is non-zero with executable size.
      -0.50  Size EXCEEDS governance limit (would trigger intervention).
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            data = _extract_json_action(completion)
            if data is None:
                rewards.append(0.0)
                continue

            size = float(data.get("size", 0.0))
            direction = int(data.get("direction", 0))
            limit = _extract_signal_value(prompt, "position_limit")
            if limit is None:
                limit = 1.0

            score = 0.0

            # Core compliance: within limit
            if size <= limit:
                score += 0.40
                # Conservative bonus: using ≤ 80 % of limit
                if 0 < size <= limit * 0.8:
                    score += 0.20
            else:
                # Governance would intervene — penalise
                score -= 0.50

            # Activity bonus only for executable, non-hold actions.
            if direction != 0 and size >= 0.01:
                score += 0.20

            rewards.append(float(np.clip(score, 0.0, 1.0)))
        except Exception:
            rewards.append(0.0)
    return rewards
