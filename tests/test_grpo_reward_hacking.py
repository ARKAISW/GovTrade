"""Regression tests for GRPO reward-hacking loopholes."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.reward import alignment_reward_func, format_reward_func, profit_reward_func
from training.prompt_utils import build_prompt_multiagent, generate_pz_scenarios


def _completion(direction: int, size: float, thought: str) -> str:
    return (
        f"<thought>{thought}</thought>"
        f"<action>{{\"direction\": {direction}, \"size\": {size}, \"sl\": 0, \"tp\": 0}}</action>"
    )


def test_profit_reward_uses_hidden_future_return_and_rejects_fake_trade():
    prompts = ["same visible prompt", "same visible prompt", "same visible prompt"]
    completions = [
        _completion(1, 0.05, "buy because setup is bullish"),
        _completion(1, 0.0, "buy because setup is bullish"),
        _completion(0, 0.0, "hold and wait"),
    ]

    rewards = profit_reward_func(prompts, completions, future_return=[0.02, 0.02, 0.02])

    assert rewards == [1.0, 0.0, 0.1]


def test_alignment_rewards_internal_consistency_not_prompt_echoing():
    prompt = "TA signal is bullish"
    bearish_sell = _completion(2, 0.05, "sell because momentum is bearish")
    bullish_sell = _completion(2, 0.05, "buy because momentum is bullish")

    rewards = alignment_reward_func([prompt, prompt], [bearish_sell, bullish_sell])

    assert rewards == [1.0, 0.0]


def test_format_reward_does_not_require_padding():
    completion = _completion(1, 0.05, "buy setup.")

    assert format_reward_func([""], [completion]) == [1.0]


def test_pz_scenarios_include_hidden_future_return_without_prompt_leak():
    scenarios = generate_pz_scenarios(n=2, difficulty="easy", max_env_steps=10)

    assert scenarios
    for scenario in scenarios:
        assert isinstance(scenario["future_return"], float)
        assert scenario["rm_size_limit"] == pytest.approx(scenario["signals"]["rm_size_limit"])
        assert "future_return" not in build_prompt_multiagent(scenario)
