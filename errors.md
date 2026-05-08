---
  Reward Hacking & Logical Error Analysis                                                                                                                           
   
  CRITICAL ISSUES                                                                                                                                                   
                  
  1. RM Reward is Structurally Negative-Biased (multi_agent_env.py:583-590)

  The Risk Manager's reward is never normalized via tanh, unlike the Trader and PM. Its per-step reward is:

  rm_reward = drawdown_penalty (-0.20 max) + profit_share (±0.01-0.03) - restrictiveness_penalty (0 to 0.45)

  This means the RM lives in roughly [-0.70, +0.03] per step — almost always negative. The profit share (profit * 100 * 0.1 = profit * 10) is too small to offset
  the penalties. The RM will learn that any action leads to negative reward, which causes learned helplessness — it cannot distinguish good from bad actions.

  Fix: Apply tanh normalization to RM reward, or scale profit sharing to profit * 100 * 0.5 to balance against penalties.

  2. Asymmetric RM Incentives Favor Excessive Conservatism (multi_agent_env.py:583-586)

  The RM gets 20% of downside but only 10% of upside:
  if profit_signal < 0:
      rm_pain = profit_signal * 0.2   # 20% downside sharing
  else:
      rm_pain = profit_signal * 0.1   # 10% upside sharing

  Combined with a low restrictiveness penalty during high drawdown (line 521: * 0.10 multiplier), the optimal RM strategy is to permanently clamp size_limit to
  near-zero. The RM can trivially avoid drawdown penalties while giving up very little upside participation. This is a classic reward-hacking path.

  Fix: Symmetrize the sharing ratio (both 15%), or make the restrictiveness penalty scale with missed profit opportunities.

  3. Trader Can Bypass Hold Penalty Indefinitely (multi_agent_env.py:448-463)

  A trader can open a position at >10% portfolio exposure, then output direction=0, size=0 forever. The hold penalty is waived when is_invested and
  _steps_since_last_trade < 20, but _steps_since_last_trade resets only on meaningful trades. Once invested, the trader can ride indefinitely — the 20-step counter
  only matters for the "safely riding" exemption, after which _consecutive_holds still resets... wait, no. Let me re-read:

  if is_invested and self._steps_since_last_trade < 20:
      pass  # no penalty
  else:
      self._consecutive_holds += 1

  After 20 steps of inaction (even while invested), _consecutive_holds starts incrementing, costing 0.001 * min(holds, 20) per step. At max that's 0.02 per step —
  negligible compared to typical profit signals of 0.1-1.0. The inactivity gate at episode end (line 618-621) only triggers if trade_ratio < 0.15. A trader
  executing 75 trades in 500 steps (15%) passes the gate while holding >80% of the time. This is not a tight constraint.

  Fix: Increase the progressive hold cost to 0.005 or scale it with portfolio volatility.

  4. Trader Can Game PM Override Visibility (multi_agent_env.py:345)

  The trader observes pm_override_strength directly. If it learns that override > 0.7 means "PM will veto new positions," it can preemptively output direction=0
  whenever it sees high override, avoiding the -0.10 intervention penalty while the PM takes the veto penalty instead. The trader externalizes costs to the PM.

  Fix: Add a small penalty for "preemptive holding" when the trader holds in the face of favorable signals but high override, or mask the raw override value behind
  a discrete signal.

  ---
  HIGH-SEVERITY ISSUES

  5. Terminal Penalties Create Massive Reward Discontinuities (multi_agent_env.py:605-621)

  Bankruptcy applies reward -= 100.0 and inactivity applies reward -= 10.0 to a reward already tanh-normalized to [-1, 1]. This creates a 100:1 spike on the final
  step. While GAE partially propagates this backward, the credit assignment is poor — the last step's action gets blamed for cumulative errors. A trader that slowly
   bleeds the portfolio to 12% then does one good trade avoiding bankruptcy gets no bankruptcy penalty, while one that holds at 11% then has a bad final step gets
  hit with -100. The boundary is a hard cliff at 10%.

  6. Profit Denominator Uses Initial Capital, Not Current NAV (multi_agent_env.py:476)

  profit = (new_value - prev_value) / (self.initial_cash + 1e-10)

  A portfolio that doubled to $200K gets the same reward for a $1K gain as it did at $100K. This penalizes compounding and doesn't reflect risk-adjusted returns.
  The agent is incentivized to take larger risks as the portfolio grows because the reward per unit of risk stays constant.

  Fix: Use prev_value as the denominator.

  7. SL/TP Assist Penalty is Too Cheap (multi_agent_env.py:413)

  The trader pays only -0.02 per assist (auto-SL or auto-TP). It can learn to always output sl=0, tp=0 and pay 0.04/step to offload all risk management to the
  environment. The auto-SL uses a fixed 2% distance regardless of volatility — in high-vol regimes this triggers prematurely; in low-vol it's too wide.

  Fix: Increase assist penalty to -0.05 and make it scale with the deviation from a reasonable SL/TP.

  8. GRPO Profit Reward Uses Future Look-Ahead (prompt_utils.py:169)

  future_step = min(current_step + 5, len(env.df) - 1)
  future_return = (future_price - current_price) / (current_price + 1e-10)

  The profit_reward_func rewards the LLM for predicting 5-step-ahead returns. This is a look-ahead cheat — the policy may learn spurious correlations rather than
  causally predictive features. In live trading, this signal doesn't exist.

  9. Inconsistent Reward Normalization Across Agents

  ┌────────┬─────────────────────────┬────────────────────┬────────────────────┐
  │ Agent  │      Normalization      │ Terminal Penalties │   Typical Range    │
  ├────────┼─────────────────────────┼────────────────────┼────────────────────┤
  │ Trader │ tanh                    │ Yes (-100/-10)     │ [-1, 1] then spike │
  ├────────┼─────────────────────────┼────────────────────┼────────────────────┤
  │ RM     │ None                    │ Yes (-60)          │ [-0.70, +0.03]     │
  ├────────┼─────────────────────────┼────────────────────┼────────────────────┤
  │ PM     │ tanh then raw penalties │ Yes (-30)          │ [-2.2, +1.0]       │
  └────────┴─────────────────────────┴────────────────────┴────────────────────┘

  This inconsistency means the three agents operate on fundamentally different reward scales, making multi-agent learning dynamics unstable.

  ---
  MEDIUM-SEVERITY ISSUES

  10. PM Oscillation Exploit (multi_agent_env.py:345, 515-517)

  The PM could oscillate override_strength between 0.71 and 0.69 at high frequency. When >0.7, new positions are vetoed. When <0.7, they pass. The utilization
  penalty (line 511) and veto penalty (line 516) apply per-step, but the PM could learn to alternate, averaging the penalty while maximizing control. The
  override_thrashing failure detector (failure_taxonomy.py:179-193) catches this for logging but doesn't feed back into the reward.

  11. Short Sale Position Limit Not Applied to Existing Positions

  In _step_trader(), RM constraints like size_limit and allow_new are checked against the proposed trade direction and size. But the RM's force_reduce flag (line
  325-331) only flips BUY to SELL for long positions. There's no symmetric handling for short positions — if the trader is short and RM signals force_reduce, the
  trader can still add to shorts (direction=2, pos <= 0 means is_opening_or_adding = True, which is blocked by allow_new but force_reduce doesn't flip direction=2
  to direction=1 for shorts).

  12. Regime Noise is Too Weak (multi_agent_env.py:675)

  regime_val += np.random.normal(0, 0.02)

  With regimes spaced ~0.083 apart and std=0.02, the signal-to-noise ratio is ~4:1. An agent can denoise this by averaging over ~10 steps, perfectly identifying the
   regime. This defeats the purpose of making regime identification non-trivial.

  13. GRPO Verifier Reward Scale Problem

  The GRPO reward functions produce values in [0, 0.5] range, which are then summed across 5 verifiers for a max of ~2.6. But individual verifier signals are very
  weak — a trader that gets the direction right but wrong size gets 0.4 (profit) + 0.0 (risk) = 0.4, while one that picks the right size but wrong direction gets
  0.0 + 0.7 = 0.7. The relative weighting means governance compliance dominates profitability in GRPO training.

  ---
  MINOR / DESIGN OBSERVATIONS

  14. Trader PPO Policy Never Sets SL/TP (ppo_trainer.py:385-386)

  "sl": np.array([0.0], dtype=np.float32),
  "tp": np.array([0.0], dtype=np.float32),

  The LearnedTrader network output doesn't include SL/TP heads. The trader always delegates risk management to the environment, paying a small assist penalty. This
  prevents learning sophisticated exit strategies.

  15. Bankruptcy Terminal Penalties Are Not in Failure Taxonomy

  The COLLAPSE_CASCADE failure (30% drop in 10 steps) is detected for logging but doesn't independently penalize agents. Only the 10%-of-initial bankruptcy trigger
  applies penalties. A 25% drawdown that recovers is scored identically whether it happened gradually or in a cascade.

  ---
  VERDICT

  The environment is NOT safe for training without modifications. The three critical issues (RM negative bias, asymmetric RM incentives, and the hold-penalty
  bypass) create clear reward-hacking paths. The terminal penalty discontinuities and inconsistent normalization will cause unstable multi-agent learning.

  However, the architecture is thoughtfully designed and the anti-hacking measures (progressive hold penalty, activity gate, failure taxonomy, governance penalties)
   show awareness of these problems. With the fixes noted above — particularly normalizing the RM reward, symmetrizing the profit share, and closing the
  hold-penalty loophole — the environment would be substantially more robust for training neural networks toward positive results