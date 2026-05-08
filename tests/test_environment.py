"""Integration tests for the multi-agent environment with regime engine."""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.multi_agent_env import (
    MultiAgentTradingEnv, RISK_MANAGER, PORTFOLIO_MGR, TRADER, ALL_AGENTS,
    BASE_OBS_SIZE, RM_MSG_SIZE, PM_MSG_SIZE, MIN_TRADE_SIZE,
)


class TestEnvCreation:
    def test_default_construction(self):
        env = MultiAgentTradingEnv(max_steps=50)
        assert env.max_steps == 50
        assert env.possible_agents == ALL_AGENTS

    def test_seeded_construction(self):
        env = MultiAgentTradingEnv(max_steps=50, seed=42)
        assert hasattr(env, "_seed")
        assert env._seed == 42

    def test_forced_regime(self):
        env = MultiAgentTradingEnv(max_steps=50, forced_regime="crash")
        assert env._forced_regime == "crash"

    def test_difficulty_levels(self):
        for diff in ["easy", "medium", "hard", "adversarial"]:
            env = MultiAgentTradingEnv(max_steps=50, difficulty=diff)
            env.reset()
            assert env.agents == ALL_AGENTS


class TestObservationSpaces:
    def test_rm_obs_shape(self):
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()
        obs = env.observe(RISK_MANAGER)
        assert obs.shape == (BASE_OBS_SIZE,), f"RM obs shape: {obs.shape}, expected ({BASE_OBS_SIZE},)"

    def test_pm_obs_shape(self):
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()
        obs = env.observe(PORTFOLIO_MGR)
        assert obs.shape == (BASE_OBS_SIZE + RM_MSG_SIZE,)

    def test_trader_obs_shape(self):
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()
        # Trader can only observe when it's their turn; step RM and PM first
        env.step(np.array([0.5, 1.0, 0.0], dtype=np.float32))  # RM
        env.step(np.array([0.6, 0.0], dtype=np.float32))       # PM
        obs = env.observe(TRADER)
        assert obs.shape == (BASE_OBS_SIZE + RM_MSG_SIZE + PM_MSG_SIZE,)

    def test_regime_indicator_present(self):
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()
        obs = env.observe(RISK_MANAGER)
        # The last element of base obs should be regime indicator
        regime_val = obs[BASE_OBS_SIZE - 1]
        assert 0.0 <= regime_val <= 1.0, f"Regime indicator {regime_val} out of range"


class TestStepMechanics:
    def test_turn_order(self):
        """Agents should act in order: RM → PM → Trader."""
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()
        assert env.agent_selection == RISK_MANAGER

        env.step(np.array([0.5, 1.0, 0.0], dtype=np.float32))
        assert env.agent_selection == PORTFOLIO_MGR

        env.step(np.array([0.6, 0.0], dtype=np.float32))
        assert env.agent_selection == TRADER

    def test_full_cycle(self):
        """A full RM→PM→Trader cycle should advance the market step."""
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()
        initial_step = env._current_step

        # Full cycle
        env.step(np.array([0.5, 1.0, 0.0], dtype=np.float32))
        env.step(np.array([0.6, 0.0], dtype=np.float32))
        env.step({"direction": 0, "size": np.array([0.0]), "sl": np.array([0.0]), "tp": np.array([0.0])})

        assert env._current_step == initial_step + 1

    def test_rm_message_propagation(self):
        """RM's action should be stored as the internal message for PM and Trader."""
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()

        rm_action = np.array([0.3, 0.0, 1.0], dtype=np.float32)
        env.step(rm_action)

        # After RM acts, internal message should be set
        # (obs are regenerated after full cycle, but _rm_message is immediate)
        np.testing.assert_array_almost_equal(
            env._rm_message,
            np.array([0.3, 0.0, 1.0], dtype=np.float32),
            decimal=4,
        )

    def test_governance_intervention(self):
        """When trader exceeds RM's size limit, it should be clamped."""
        env = MultiAgentTradingEnv(max_steps=50)
        env.reset()

        # RM sets tight limit
        env.step(np.array([0.1, 1.0, 0.0], dtype=np.float32))
        # PM passes through
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        # Trader tries to buy with large size
        env.step({"direction": 1, "size": np.array([0.8]), "sl": np.array([0.0]), "tp": np.array([0.0])})

        # Check that intervention was logged
        info = env.infos.get(TRADER, {})
        gov = info.get("governance", {})
        assert len(gov.get("interventions", [])) > 0, "RM should have intervened"


class TestAntiRewardHacking:
    def test_ticket_fee_bleeds_wash_trade(self):
        env = MultiAgentTradingEnv(max_steps=10, seed=42, ticket_fee=5.0)
        env.reset()
        price = env._market.current_price()
        initial_value = env._portfolio.total_value(price, env.ticker)

        assert env._execute_trade(1, MIN_TRADE_SIZE, price * 0.98, price * 1.04, price)
        assert env._execute_trade(2, 1.0, price * 0.98, price * 1.04, price)

        final_value = env._portfolio.total_value(price, env.ticker)
        assert final_value < initial_value - 10.0

    def test_micro_trade_is_rejected_as_hold(self):
        env = MultiAgentTradingEnv(max_steps=10, seed=42)
        env.reset()

        env.step(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        env.step({
            "direction": 1,
            "size": np.array([MIN_TRADE_SIZE / 2], dtype=np.float32),
            "sl": np.array([0.0], dtype=np.float32),
            "tp": np.array([0.0], dtype=np.float32),
        })

        gov = env.infos[TRADER]["governance"]
        intervention_types = {item["type"] for item in gov["interventions"]}
        assert "min_trade_size" in intervention_types
        assert gov["executed"]["direction"] == 0
        assert env._trades_executed == 0
        assert env._consecutive_holds == 1

    def test_spoofed_stop_loss_is_clamped(self):
        env = MultiAgentTradingEnv(max_steps=10, seed=42)
        env.reset()
        price = env._market.current_price()

        env.step(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        env.step({
            "direction": 1,
            "size": np.array([0.05], dtype=np.float32),
            "sl": np.array([0.0001], dtype=np.float32),
            "tp": np.array([0.0], dtype=np.float32),
        })

        gov = env.infos[TRADER]["governance"]
        auto_sl = [item for item in gov.get("assists", []) if item["type"] == "auto_sl"]
        assert auto_sl
        assert gov["executed"]["sl"] == pytest.approx(price * 0.98)


class TestTermination:
    def test_episode_terminates(self):
        """Episode should terminate within max_steps cycles."""
        env = MultiAgentTradingEnv(max_steps=10)
        env.reset()

        steps = 0
        while env.agents and steps < 100:
            agent = env.agent_selection
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
            elif agent == RISK_MANAGER:
                env.step(np.array([0.5, 1.0, 0.0], dtype=np.float32))
            elif agent == PORTFOLIO_MGR:
                env.step(np.array([0.6, 0.0], dtype=np.float32))
            elif agent == TRADER:
                env.step({"direction": 0, "size": np.array([0.0]), "sl": np.array([0.0]), "tp": np.array([0.0])})
            steps += 1

        assert steps < 100, "Episode did not terminate within expected steps"


class TestReproducibility:
    def test_same_seed_same_result(self):
        """Two environments with the same seed should produce identical observations."""
        env1 = MultiAgentTradingEnv(max_steps=20, seed=42)
        env2 = MultiAgentTradingEnv(max_steps=20, seed=42)

        env1.reset(seed=42)
        env2.reset(seed=42)

        obs1 = env1.observe(RISK_MANAGER)
        obs2 = env2.observe(RISK_MANAGER)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seed_different_result(self):
        """Two environments with different seeds should generally differ."""
        env1 = MultiAgentTradingEnv(max_steps=20, seed=42)
        env2 = MultiAgentTradingEnv(max_steps=20, seed=99)

        env1.reset(seed=42)
        env2.reset(seed=99)

        obs1 = env1.observe(RISK_MANAGER)
        obs2 = env2.observe(RISK_MANAGER)
        # Price data should differ (extremely unlikely to be equal)
        assert not np.array_equal(obs1, obs2)


class TestRegimeIntegration:
    def test_regime_labels_exist(self):
        env = MultiAgentTradingEnv(max_steps=50, seed=42)
        assert hasattr(env, "_regime_labels")
        assert len(env._regime_labels) > 0

    def test_forced_regime_labels(self):
        env = MultiAgentTradingEnv(max_steps=50, forced_regime="crash")
        assert all(lbl == "crash" for lbl in env._regime_labels)

    def test_current_regime_updates(self):
        env = MultiAgentTradingEnv(max_steps=50, seed=42)
        env.reset()
        initial_regime = env._current_regime_label
        assert isinstance(initial_regime, str)
        assert len(initial_regime) > 0
