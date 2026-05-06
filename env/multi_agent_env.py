"""
Multi-Agent Trading Environment using PettingZoo AEC API.

Three independent RL agents operate in a decentralized governance framework:
  - risk_manager_0:    Rewarded for restricting dangerous trades. Penalized when Trader loses.
  - portfolio_manager_0: Oversees capital allocation. Rewarded for portfolio growth + drawdown control.
  - trader_0:          Rewarded purely for PnL. Sees Risk/PM constraints as observations.

The AEC (Agent-Environment Cycle) loop alternates agent turns each step.
Agent Negotiation: Each agent's *output message* (constraints, allocations) becomes
part of the next agent's observation, creating an emergent negotiation dynamic.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from gymnasium import spaces

from pettingzoo import AECEnv

try:
    # PettingZoo 1.25.0+ exposes the selector class as AgentSelector.
    from pettingzoo.utils import AgentSelector
except ImportError:
    # Older releases expose agent_selector directly, while some transitional
    # layouts expose a module with AgentSelector inside it.
    from pettingzoo.utils import agent_selector as _agent_selector

    AgentSelector = getattr(_agent_selector, "AgentSelector", _agent_selector)

from env.state import MarketState, PortfolioState, RiskState, get_observation
from env.reward import compute_raw_reward, normalize_reward, compute_grade
from env.failure_taxonomy import GovernanceFailureTaxonomy
from utils.indicators import compute_indicators


# ─── Agent IDs ─────────────────────────────────────────────────────────────────
RISK_MANAGER    = "risk_manager_0"
PORTFOLIO_MGR   = "portfolio_manager_0"
TRADER          = "trader_0"
ALL_AGENTS      = [RISK_MANAGER, PORTFOLIO_MGR, TRADER]
MIN_TRADE_SIZE = 0.01
DEFAULT_TICKET_FEE = 5.0

# ─── Observation Sizes ──────────────────────────────────────────────────────────
# Base market+portfolio+risk obs size: 14 + 5 + 5 = 24
BASE_OBS_SIZE = 25  # 24 original + 1 regime indicator
# Risk Manager message appended to PM and Trader observations: [size_limit, allow_new, force_reduce]
RM_MSG_SIZE = 3
# PM message appended to Trader observations: [cap_allocation, is_override_signaled]
PM_MSG_SIZE = 2


class MultiAgentTradingEnv(AECEnv):
    """
    A PettingZoo AEC environment for decentralized multi-agent trading governance.

    Turn order per step: risk_manager_0 → portfolio_manager_0 → trader_0
    On each full cycle, the market advances by one candle.

    Observations:
      risk_manager_0:   base_obs (24,)
      portfolio_mgr_0:  base_obs + rm_message (24 + 3 = 27,)
      trader_0:         base_obs + rm_message + pm_message (24 + 3 + 2 = 29,)

    Actions:
      risk_manager_0:   Box(3,) — [size_limit, allow_new_positions, force_reduce] — continuous
      portfolio_mgr_0:  Box(2,) — [capital_allocation_fraction, override_strength] — continuous
      trader_0:         Dict — direction (Discrete 3), size (Box 1), sl (Box 1), tp (Box 1)
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "multi_agent_trading_v1",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        initial_cash: float = 100_000.0,
        ticker: str = "default",
        commission: float = 0.001,
        ticket_fee: float = DEFAULT_TICKET_FEE,
        max_steps: Optional[int] = None,
        difficulty: str = "hard",
        seed: Optional[int] = None,
        forced_regime: Optional[str] = None,
    ):
        super().__init__()

        self.difficulty = difficulty
        self._seed = seed
        self._forced_regime = forced_regime
        self._initial_cash = initial_cash

        if df is None:
            gen_n = (max_steps + 1) if max_steps is not None else 500
            df, self._regime_labels = self._make_dummy_data(
                n=gen_n,
                difficulty=difficulty, seed=seed, forced_regime=forced_regime,
            )
        else:
            self._regime_labels = [""] * len(df)
        self.raw_df = df.copy()
        self.df = compute_indicators(df)
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.commission = commission
        self.ticket_fee = float(ticket_fee)
        self.max_steps = max_steps or (len(self.df) - 1)
        self._current_regime_label = self._regime_labels[0] if self._regime_labels else ""

        # ── PettingZoo required attributes ──────────────────────────────────
        self.agents = ALL_AGENTS[:]
        self.possible_agents = ALL_AGENTS[:]

        # ── Observation spaces ──────────────────────────────────────────────
        self.observation_spaces = {
            RISK_MANAGER:   spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(BASE_OBS_SIZE,), dtype=np.float32),
            PORTFOLIO_MGR:  spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(BASE_OBS_SIZE + RM_MSG_SIZE,), dtype=np.float32),
            TRADER:         spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(BASE_OBS_SIZE + RM_MSG_SIZE + PM_MSG_SIZE,), dtype=np.float32),
        }

        # ── Action spaces ───────────────────────────────────────────────────
        self.action_spaces = {
            RISK_MANAGER:  spaces.Box(low=np.array([0.01, 0.0, 0.0], dtype=np.float32),
                                      high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                      shape=(3,), dtype=np.float32),
            PORTFOLIO_MGR: spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                      high=np.array([1.0, 1.0], dtype=np.float32),
                                      shape=(2,), dtype=np.float32),
            TRADER:        spaces.Dict({
                "direction": spaces.Discrete(3),          # 0=Hold, 1=Buy, 2=Sell/Short
                "size":      spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "sl":        spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                "tp":        spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            }),
        }

        # ── Internal state (reset before first use) ─────────────────────────
        self._agent_selector = AgentSelector(ALL_AGENTS)
        self._reset_internal_state()

    # ───────────────────────────────────────────────────────────────────────────
    # PettingZoo required API
    # ───────────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = ALL_AGENTS[:]
        self._agent_selector.reinit(ALL_AGENTS)

        self._reset_internal_state()
        self._generate_observations()

        self.agent_selection = self._agent_selector.reset()

        # Zero-fill all rewards/terminations/truncations/infos for PZ compliance
        self.rewards         = {ag: 0.0 for ag in self.agents}
        self._cumulative_rewards = {ag: 0.0 for ag in self.agents}
        self.terminations    = {ag: False for ag in self.agents}
        self.truncations     = {ag: False for ag in self.agents}
        self.infos           = {ag: {} for ag in self.agents}

    def step(self, action):
        """Process one agent's action in the AEC turn order."""
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            # Dead-step: PZ compliance requires we handle this
            self._was_dead_step(action)
            return
        # The current agent's cumulative reward was already returned by last().
        # Reset its accumulation window before processing a fresh action.
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()

        # ── Route action to the correct handler ────────────────────────────
        if agent == RISK_MANAGER:
            self._step_risk_manager(action)
        elif agent == PORTFOLIO_MGR:
            self._step_portfolio_manager(action)
        elif agent == TRADER:
            self._step_trader(action)
            # After the trader acts, the market cycle is complete → advance step
            self._advance_market()

        # Advance to next agent
        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> np.ndarray:
        return self._observations[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def render(self):
        price = self._market.current_price()
        val   = self._portfolio.total_value(price, self.ticker)
        print(
            f"Step {self._current_step:4d} | "
            f"Price: {price:10,.2f} | "
            f"Value: {val:12,.2f} | "
            f"Agent: {self.agent_selection}"
        )

    def close(self):
        pass

    # ───────────────────────────────────────────────────────────────────────────
    # Per-Agent Step Handlers
    # ───────────────────────────────────────────────────────────────────────────

    def _step_risk_manager(self, action: np.ndarray):
        """
        Risk Manager decides governance constraints.
        action = [size_limit (0-1), allow_new_positions (0-1), force_reduce (0-1)]

        Reward logic:
          -0.40 for failing to restrict size during material drawdown
          shared pain/gain with portfolio (biased towards protection)
        """
        size_limit, allow_new_raw, force_reduce_raw = float(action[0]), float(action[1]), float(action[2])
        allow_new  = allow_new_raw  > 0.5
        force_reduce = force_reduce_raw > 0.5

        # Store message to pass to PM and Trader
        self._rm_message = np.array(
            [size_limit, float(allow_new), float(force_reduce)], dtype=np.float32
        )

        # Compute RM's step reward
        drawdown = self._risk.current_drawdown
        rm_reward = 0.0

        # Penalty-only drawdown response: never reward the RM for a crisis existing.
        if drawdown > 0.10 and size_limit > 0.30:
            rm_reward -= 0.40

        # Note: shared upside/downside is now handled in _advance_market to ensure
        # synchronized profit calculation across all agents.
        self._rm_cycle_reward = float(rm_reward)

    def _step_portfolio_manager(self, action: np.ndarray):
        """
        Portfolio Manager decides capital allocation and optionally signals override.
        action = [capital_allocation (0-1), override_strength (0-1)]

        Reward logic:
          Aligned with overall portfolio performance (grade-based).
          Penalized for excessive overrides that don't improve outcomes.
        """
        cap_alloc  = float(np.clip(action[0], 0.0, 1.0))
        override_s = float(action[1])

        self._pm_message = np.array([cap_alloc, override_s], dtype=np.float32)
        self._pm_capital_allocation = cap_alloc
        self._pm_override_strength  = override_s

        # PM reward is deferred until after the trader executes and the outcome is known.

    def _step_trader(self, action: Dict):
        """
        Trader proposes a trade using the constrained action space.
        Receives both RM and PM guidance in its observation.

        Reward logic (adversarial):
          Rewarded purely on PnL.
          Penalized when governance overrides (RM size cap, PM force-close) are triggered.
          Bonus for proposing compliant actions that need no governance intervention.
        """
        direction = int(action["direction"])
        size_raw  = float(action["size"][0]) if hasattr(action["size"], "__len__") else float(action["size"])
        sl_input  = float(action["sl"][0])   if hasattr(action["sl"],   "__len__") else float(action.get("sl", 0.0))
        tp_input  = float(action["tp"][0])   if hasattr(action["tp"],   "__len__") else float(action.get("tp", 0.0))

        size = float(np.clip(size_raw, 0.0, 1.0))

        # ── Apply Risk Manager constraints ──────────────────────────────────
        rm_size_limit  = float(self._rm_message[0])
        rm_allow_new   = bool(self._rm_message[1] > 0.5)
        rm_force_reduce = bool(self._rm_message[2] > 0.5)
        
        pos = self._portfolio.positions.get(self.ticker, 0.0)

        interventions: List[Dict] = []

        if direction != 0 and size > rm_size_limit:
            interventions.append({
                "agent": "RiskManager",
                "type":  "size_clamp",
                "original_size":  size,
                "enforced_size":  rm_size_limit,
            })
            size = rm_size_limit

        is_opening_or_adding = (direction == 1 and pos >= 0) or (direction == 2 and pos <= 0)
        if is_opening_or_adding and not rm_allow_new:
            interventions.append({
                "agent": "RiskManager",
                "type":  "no_new_positions",
                "reason": "RM blocked new positions during drawdown",
            })
            direction = 0  # Force hold

        if rm_force_reduce and direction == 1 and pos >= 0:
            interventions.append({
                "agent": "RiskManager",
                "type":  "force_reduce",
                "reason": "RM signaling to reduce longs",
            })
            direction = 2  # Flip to reduce

        # ── Apply Portfolio Manager override ────────────────────────────────
        cap_alloc  = self._pm_capital_allocation
        if direction != 0 and size > cap_alloc:
            interventions.append({
                "agent": "PortfolioManager",
                "type":  "capital_cap",
                "original_size": size,
                "enforced_size": cap_alloc,
            })
            size = min(size, cap_alloc)

        # PM strong override_strength >0.7 means PM wants to veto new positions
        if self._pm_override_strength > 0.7 and direction != 0 and is_opening_or_adding:
            interventions.append({
                "agent": "PortfolioManager",
                "type":  "pm_veto",
                "reason": "PM vetoed new trade (insufficient conviction signal)",
            })
            direction = 0

        # ── Auto SL/TP (governance baseline) ───────────────────────────────
        if direction != 0 and size < MIN_TRADE_SIZE:
            interventions.append({
                "agent": "Environment",
                "type": "min_trade_size",
                "original_size": size,
                "minimum_size": MIN_TRADE_SIZE,
                "reason": "Trade rejected as below the execution threshold",
            })
            direction = 0
            size = 0.0

        current_price = self._market.current_price()
        DEFAULT_SL = 0.02
        invalid_sl = False
        if direction == 1:
            invalid_sl = sl_input <= 0 or sl_input >= current_price or sl_input < current_price * 0.5
        elif direction == 2:
            invalid_sl = sl_input <= 0 or sl_input <= current_price or sl_input > current_price * 1.5
            
        if direction != 0 and invalid_sl:
            original_sl = sl_input
            if direction == 1:
                sl_input = current_price * (1 - DEFAULT_SL)
            else:
                sl_input = current_price * (1 + DEFAULT_SL)
            interventions.append({
                "agent": "RiskManager",
                "type": "auto_sl",
                "original_sl": original_sl,
                "enforced_sl": sl_input,
            })
            
        if direction != 0 and tp_input <= 0 and sl_input > 0:
            sl_dist = abs(current_price - sl_input)
            tp_input = (current_price + sl_dist * 2.0) if direction == 1 else (current_price - sl_dist * 2.0)
            interventions.append({"agent": "RiskManager", "type": "auto_tp"})

        # Store pending trade for market advance
        self._pending_trade = {
            "direction": direction,
            "size": size,
            "sl": sl_input,
            "tp": tp_input,
            "interventions": interventions,
            "original_direction": int(action["direction"]),
            "original_size": size_raw,
        }

        # Compliance reward/penalty — will be finalized after market moves
        n_interventions = len(interventions)
        compliance_penalty = -0.05 * n_interventions
        self._trader_compliance_bonus = compliance_penalty

    # ───────────────────────────────────────────────────────────────────────────
    # Market Advance (called after Trader acts)
    # ───────────────────────────────────────────────────────────────────────────

    def _advance_market(self):
        """Execute the pending trade, advance market, compute final rewards."""
        if not hasattr(self, "_pending_trade") or self._pending_trade is None:
            # No trade was staged (edge case)
            self._pending_trade = {"direction": 0, "size": 0.0, "sl": 0.0, "tp": 0.0,
                                   "interventions": [], "original_direction": 0, "original_size": 0.0}

        trade = self._pending_trade
        direction = trade["direction"]
        size      = trade["size"]
        sl_input  = trade["sl"]
        tp_input  = trade["tp"]

        current_price = self._market.current_price()
        prev_value    = self._portfolio.total_value(current_price, self.ticker)

        # Check SL/TP before executing new action
        sl_tp_hit = self._check_sl_tp(current_price)

        # Execute trade in portfolio state
        traded = self._execute_trade(direction, size, sl_input, tp_input, current_price)

        # Track activity based on actual execution (closing the fake-trade loophole)
        if not hasattr(self, "_steps_since_last_trade"):
            self._steps_since_last_trade = 0

        pos_qty = self._portfolio.positions.get(self.ticker, 0.0)
        is_invested = (abs(pos_qty) * current_price) / (prev_value + 1e-10) > 0.10
        is_meaningful_trade = traded and size >= 0.05

        if is_meaningful_trade or sl_tp_hit:
            self._consecutive_holds = 0
            self._steps_since_last_trade = 0
            if traded: self._trades_executed += 1
        else:
            self._steps_since_last_trade += 1
            if traded: self._trades_executed += 1
            
            # Trader is excused from hold penalty ONLY if meaningfully invested AND hasn't been holding indefinitely
            if is_invested and self._steps_since_last_trade < 20:
                pass  # Safely riding a trend, no penalty
            else:
                self._consecutive_holds += 1

        # Advance market step
        self._current_step += 1
        self._market.current_step = self._current_step

        # Update risk state
        new_price = self._market.current_price() if self._current_step < len(self.df) else current_price
        new_value = self._portfolio.total_value(new_price, self.ticker)
        self._risk.update(new_value)
        self._episode_values.append(new_value)

        # Compute portfolio delta
        profit = (new_value - prev_value) / (self.initial_cash + 1e-10)
        price_trend = (new_price - current_price) / (current_price + 1e-10)

        raw_r = compute_raw_reward(
            profit=profit,
            drawdown=self._risk.current_drawdown,
            volatility=self._risk.return_volatility(),
            sharpe=self._risk.sharpe_ratio(),
            trade_count=int(traded or sl_tp_hit),
            direction=direction,
            price_trend=price_trend,
            trade_size=size,
        )

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 1: Progressive Consecutive Hold Penalty (Applied to Raw Reward)
        # ═══════════════════════════════════════════════════════════════════
        # Escalates the longer the agent refuses to trade. Scaled to raw reward
        # to properly balance against ticket fees and directional bonuses.
        progressive_hold_raw_cost = 0.001 * min(self._consecutive_holds, 20)
        raw_r -= progressive_hold_raw_cost

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 2: Continuous Governance Penalties (no binary thresholds)
        # ═══════════════════════════════════════════════════════════════════
        cap_alloc = float(self._pm_message[0])
        size_limit = float(self._rm_message[0])
        rm_allow_new = bool(self._rm_message[1] > 0.5)
        rm_force_reduce = bool(self._rm_message[2] > 0.5)
        current_dd = float(self._risk.current_drawdown)

        # PM: penalize low capital allocation continuously
        utilization_penalty = max(0.0, 0.20 - cap_alloc) * 5.0
        
        # Always penalize absolute vetoes so PM cannot freeze the portfolio completely
        pm_veto_pen = 1.0 if current_dd < 0.10 else 0.20
        if self._pm_override_strength > 0.7:
            utilization_penalty += pm_veto_pen

        # RM: penalize overly tight size limits. Less strict during drawdowns, but never 0.
        rm_restrict_penalty = max(0.0, 0.20 - size_limit) * (0.5 if current_dd < 0.10 else 0.10)
        if not rm_allow_new:
            rm_restrict_penalty += 0.20 if current_dd < 0.10 else 0.05
        if rm_force_reduce:
            rm_restrict_penalty += 0.20 if current_dd < 0.10 else 0.05

        # Keep failure taxonomy for research logging (NOT for reward)
        self._failure_tax.check_step(
            step=self._current_step,
            rm_action=self._rm_message,
            pm_action=self._pm_message,
            portfolio_value=float(new_value),
            drawdown=float(self._risk.max_drawdown),
            regime_label=getattr(self, "_current_regime_label", ""),
        )

        # ── Trader reward ───────────────────────────────────────────────────
        trader_reward = normalize_reward(raw_r + self._trader_compliance_bonus)
        self.rewards[TRADER] = float(trader_reward)
        self._episode_rewards.append(trader_reward)

        # ── PM reward: risk-averse portfolio performance ────────────────────
        # PM uses the same raw reward basis as the trader but with a highly risk-averse profile
        pm_weights = {
            "profit": 1.0,
            "drawdown": 2.0,       # Double penalty for drawdowns
            "volatility": 1.0,     # Higher penalty for volatility
            "sharpe": 0.5,
            "overtrading": 0.0,
            "hold_penalty": 0.0,
            "directional_bonus": 0.0
        }
        pm_raw_r = compute_raw_reward(
            profit=profit,
            drawdown=self._risk.current_drawdown,
            volatility=self._risk.return_volatility(),
            sharpe=self._risk.sharpe_ratio(),
            trade_count=0,
            weights=pm_weights,
            direction=0,
            price_trend=0.0
        )
        pm_reward = normalize_reward(pm_raw_r)
        
        # We still compute grade for info/logging
        normalized_profit  = float(np.clip((profit + 1.0) / 2.0, 0.0, 1.0))
        normalized_sharpe  = float(np.clip((self._risk.sharpe_ratio() + 2.0) / 4.0, 0.0, 1.0))
        grade = float(compute_grade({
            "profit": normalized_profit,
            "sharpe": normalized_sharpe,
            "drawdown": float(self._risk.max_drawdown),
        }))

        if self._risk.max_drawdown > 0.20:
            pm_reward -= 0.50              # PM heavily penalized for deep drawdown
        pm_reward -= utilization_penalty   # PM owns the capital allocation penalty
        self.rewards[PORTFOLIO_MGR] = float(pm_reward)

        # ── RM: shared downside with final portfolio value ──────────────────
        # RM shares 50% of downside and 10% of upside to encourage allowing good trades
        profit_signal = profit * 100.0
        if profit_signal < 0:
            rm_pain = profit_signal * 0.5
        else:
            rm_pain = profit_signal * 0.1
            
        rm_reward_final = float(self._rm_cycle_reward + rm_pain)
        rm_reward_final -= rm_restrict_penalty  # RM owns the restrictiveness penalty
        self.rewards[RISK_MANAGER] = float(rm_reward_final)

        # ── Termination Check ───────────────────────────────────────────────
        blown_up = new_value < self.initial_cash * 0.10
        terminated = (
            self._current_step >= self.max_steps or
            blown_up
        )
        if terminated:
            for ag in self.agents:
                self.terminations[ag] = True
                
            if blown_up:
                # LAYER 4: Suicide Pact Penalty
                for ag in self.agents:
                    self.rewards[ag] -= 100.0

            # ═══════════════════════════════════════════════════════════════
            # LAYER 3: End-of-Episode Activity Gate
            # ═══════════════════════════════════════════════════════════════
            # If agent traded less than 15% of steps, crush all rewards
            trade_ratio = self._trades_executed / max(self._current_step, 1)
            if trade_ratio < 0.15:
                inactivity_slam = -5.0 * (1.0 - trade_ratio / 0.15)
                for ag in self.agents:
                    self.rewards[ag] = float(self.rewards.get(ag, 0.0) + inactivity_slam)

        # Rebuild observations for the next cycle
        self._generate_observations()

        # Update governance log
        gov_record = {
            "step": self._current_step,
            "proposed": {"direction": trade["original_direction"], "size": trade["original_size"]},
            "executed": {"direction": direction, "size": size, "sl": sl_input, "tp": tp_input},
            "interventions": trade["interventions"],
            "was_compliant": len(trade["interventions"]) == 0,
            "rm_message": self._rm_message.tolist(),
            "pm_message": self._pm_message.tolist(),
        }
        self._governance_log.append(gov_record)

        # Expose info for the Trader (most info-rich agent)
        self.infos[TRADER] = {
            "step": self._current_step,
            "portfolio_value": float(new_value),
            "cash": float(self._portfolio.cash),
            "pnl": float(new_value - self.initial_cash),
            "pnl_pct": float(profit),
            "max_drawdown": float(self._risk.max_drawdown),
            "sharpe_ratio": float(self._risk.sharpe_ratio()),
            "grade": grade,
            "governance": gov_record,
            "rewards": dict(self.rewards),
        }
        self.infos[RISK_MANAGER]  = {"step": self._current_step, "drawdown": float(self._risk.max_drawdown)}
        self.infos[PORTFOLIO_MGR] = {"step": self._current_step, "grade": grade}

        self._prev_portfolio_value = new_value
        self._pending_trade = None
        self._rm_cycle_reward = 0.0

    # ───────────────────────────────────────────────────────────────────────────
    # Observation Generation
    # ───────────────────────────────────────────────────────────────────────────

    def _generate_observations(self):
        base_obs = get_observation(self._market, self._portfolio, self._risk, self.ticker)

        # Append regime indicator to observation
        step = min(self._current_step, len(self._regime_labels) - 1)
        if 0 <= step < len(self._regime_labels):
            self._current_regime_label = self._regime_labels[step]
        from env.regime_engine import REGIME_TO_ID, NUM_REGIMES
        regime_id = REGIME_TO_ID.get(self._current_regime_label, 0)
        regime_indicator = np.array([regime_id / max(NUM_REGIMES, 1)], dtype=np.float32)
        base_obs_r = np.concatenate([base_obs, regime_indicator])

        self._observations = {
            RISK_MANAGER:  base_obs_r.copy(),
            PORTFOLIO_MGR: np.concatenate([base_obs_r, self._rm_message]),
            TRADER:        np.concatenate([base_obs_r, self._rm_message, self._pm_message]),
        }

    # ───────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ───────────────────────────────────────────────────────────────────────────

    def _reset_internal_state(self):
        self._market    = MarketState(prices=self.df, current_step=0)
        self._portfolio = PortfolioState(initial_cash=self.initial_cash, cash=self.initial_cash)
        self._risk      = RiskState(peak_value=self.initial_cash)
        self._current_step = 0

        # Inter-agent messages (start neutral)
        self._rm_message = np.array([0.5, 1.0, 0.0], dtype=np.float32)  # [size_limit=50%, allow=yes, force_reduce=no]
        self._pm_message = np.array([0.5, 0.0], dtype=np.float32)        # [cap_alloc=50%, override_strength=0]
        self._pm_capital_allocation = 0.5
        self._pm_override_strength  = 0.0

        self._failure_tax = GovernanceFailureTaxonomy()

        self._pending_trade  = None
        self._rm_cycle_reward = 0.0
        self._trader_compliance_bonus = 0.0

        # Anti-reward-hacking tracking
        self._consecutive_holds = 0
        self._trades_executed = 0

        self._episode_values  = [self.initial_cash]
        self._episode_rewards = []
        self._governance_log: List[Dict] = []
        self._prev_portfolio_value = self.initial_cash

        # PZ state dictionaries
        self._observations = {ag: np.zeros(self.observation_spaces[ag].shape, dtype=np.float32)
                              for ag in ALL_AGENTS}

    def _accumulate_rewards(self):
        """Add the current step rewards into PettingZoo cumulative tracking."""
        for ag in self.agents:
            self._cumulative_rewards[ag] += self.rewards[ag]

    def _execute_trade(self, direction: int, size: float, sl: float, tp: float, current_price: float) -> bool:
        """Execute trade, applying commissions and updating portfolio. Return True if trade executed."""
        if direction == 0 or size <= 0:
            return False

        if size < MIN_TRADE_SIZE:
            return False  # Trade rejected as noise; counts as a Hold

        traded = False

        if direction == 1:  # BUY / Cover Short
            pos = self._portfolio.positions.get(self.ticker, 0.0)
            if pos < 0:
                # Cover short
                abs_qty = abs(pos)
                cover_cost = abs_qty * current_price * (1 + self.commission) + self.ticket_fee
                margin_return = abs_qty * self._portfolio.avg_costs.get(self.ticker, current_price)
                self._portfolio.cash += margin_return - cover_cost
                self._portfolio.positions[self.ticker] = 0.0
                self._portfolio.avg_costs[self.ticker] = 0.0
                self._portfolio.stop_losses[self.ticker] = None
                self._portfolio.take_profits[self.ticker] = None
                traded = True
            else:
                budget = self._portfolio.cash * size
                if budget <= self.ticket_fee:
                    return False
                trade_qty = (budget - self.ticket_fee) / (current_price * (1 + self.commission) + 1e-10)
                if trade_qty > 1e-8:
                    cost = trade_qty * current_price * (1 + self.commission) + self.ticket_fee
                    self._portfolio.cash -= cost
                    prev_qty = pos
                    prev_avg  = self._portfolio.avg_costs.get(self.ticker, 0.0)
                    new_qty  = prev_qty + trade_qty
                    new_avg  = ((prev_qty * prev_avg) + (trade_qty * current_price)) / (new_qty + 1e-10)
                    self._portfolio.positions[self.ticker]   = new_qty
                    self._portfolio.avg_costs[self.ticker]   = new_avg
                    if sl > 0: self._portfolio.stop_losses[self.ticker]  = sl
                    if tp > 0: self._portfolio.take_profits[self.ticker] = tp
                    traded = True

        elif direction == 2:  # SELL / Short
            pos = self._portfolio.positions.get(self.ticker, 0.0)
            if pos > 0:
                sell_qty = min(pos, pos * size)
                if sell_qty > 1e-8:
                    revenue = sell_qty * current_price * (1 - self.commission) - self.ticket_fee
                    self._portfolio.cash += revenue
                    remaining = pos - sell_qty
                    self._portfolio.positions[self.ticker] = max(remaining, 0.0)
                    if remaining <= 1e-8:
                        self._portfolio.avg_costs[self.ticker] = 0.0
                        self._portfolio.stop_losses[self.ticker] = None
                        self._portfolio.take_profits[self.ticker] = None
                    traded = True
            else:
                margin = self._portfolio.cash * size
                if margin <= self.ticket_fee:
                    return False
                short_qty = (margin - self.ticket_fee) / (current_price * (1 + self.commission) + 1e-10)
                if short_qty > 1e-8:
                    self._portfolio.cash -= (short_qty * current_price * (1 + self.commission) + self.ticket_fee)
                    prev_qty  = abs(pos)
                    prev_avg  = self._portfolio.avg_costs.get(self.ticker, 0.0)
                    new_qty   = prev_qty + short_qty
                    new_avg   = ((prev_qty * prev_avg) + (short_qty * current_price)) / (new_qty + 1e-10)
                    self._portfolio.positions[self.ticker]   = -new_qty
                    self._portfolio.avg_costs[self.ticker]   = new_avg
                    if sl > 0: self._portfolio.stop_losses[self.ticker]  = sl
                    if tp > 0: self._portfolio.take_profits[self.ticker] = tp
                    traded = True

        if traded:
            self._risk.trade_count += 1
        return traded

    def _check_sl_tp(self, current_price: float) -> bool:
        """Check and execute SL/TP orders. Return True if hit."""
        ticker  = self.ticker
        pos_qty = self._portfolio.positions.get(ticker, 0.0)
        sl      = self._portfolio.stop_losses.get(ticker)
        tp      = self._portfolio.take_profits.get(ticker)
        if abs(pos_qty) < 1e-8:
            return False

        hit = False
        if pos_qty > 0:
            if sl and current_price <= sl: hit = True
            if tp and current_price >= tp: hit = True
            if hit:
                revenue = pos_qty * current_price * (1 - self.commission) - self.ticket_fee
                self._portfolio.cash += revenue
                self._portfolio.positions[ticker] = 0.0
                self._portfolio.avg_costs[ticker] = 0.0
                self._portfolio.stop_losses[ticker] = None
                self._portfolio.take_profits[ticker] = None
                self._risk.trade_count += 1
        elif pos_qty < 0:
            abs_qty = abs(pos_qty)
            if sl and current_price >= sl: hit = True
            if tp and current_price <= tp: hit = True
            if hit:
                avg_cost   = self._portfolio.avg_costs.get(ticker, current_price)
                cover_cost = abs_qty * current_price * (1 + self.commission) + self.ticket_fee
                margin_ret = abs_qty * avg_cost
                self._portfolio.cash += margin_ret - cover_cost
                self._portfolio.positions[ticker] = 0.0
                self._portfolio.avg_costs[ticker] = 0.0
                self._portfolio.stop_losses[ticker] = None
                self._portfolio.take_profits[ticker] = None
                self._risk.trade_count += 1
        return hit

    def _make_dummy_data(self, n: int = 500, difficulty: str = "hard",
                        seed=None, forced_regime=None):
        """Generate market data using the MarketRegimeEngine."""
        from env.regime_engine import MarketRegimeEngine
        engine = MarketRegimeEngine(seed=seed)
        df, labels = engine.generate(n_steps=n, difficulty=difficulty, forced_regime=forced_regime)
        return df, labels

    # ───────────────────────────────────────────────────────────────────────────
    # Convenience
    # ───────────────────────────────────────────────────────────────────────────

    @functools.lru_cache(maxsize=None)
    def _obs_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def _act_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def _clear_rewards(self):
        self.rewards = {ag: 0.0 for ag in self.agents}

    def _was_dead_step(self, action):
        self._clear_rewards()
        self._accumulate_rewards()

    def state(self) -> Dict:
        """Return the full shared environment state (for visualization)."""
        price = self._market.current_price()
        return {
            "step":            self._current_step,
            "price":           float(price),
            "portfolio_value": float(self._portfolio.total_value(price, self.ticker)),
            "cash":            float(self._portfolio.cash),
            "positions":       {k: float(v) for k, v in self._portfolio.positions.items()},
            "max_drawdown":    float(self._risk.max_drawdown),
            "sharpe_ratio":    float(self._risk.sharpe_ratio()),
            "trade_count":     self._risk.trade_count,
            "rm_message":      self._rm_message.tolist(),
            "pm_message":      self._pm_message.tolist(),
            "governance_log":  self._governance_log[-10:],
        }
