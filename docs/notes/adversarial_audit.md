# QuantHive Adversarial Audit: Reward Hacking Potentials

This document outlines potential vulnerabilities where RL agents might exploit mathematical loopholes in the reward functions or environment logic to achieve high scores without fulfilling the research objective of "learned adaptive governance."

---

### 1. The "Small Trend Neutrality" Farming (GRPO)
* **Loopholes:** In `profit_reward_func`, if the future return is between `-0.001` and `0.001`, `direction=0` (Hold) receives a maximum reward of `1.0`.
* **Adversarial Case:** The agent might learn to become "hyper-passive" during any sideways market, farming a perfect score by doing nothing. This reduces the model's incentive to find entries in low-volatility regimes that precede break-outs.

### 2. Alignment Keyword Stuffing (GRPO)
* **Loopholes:** `alignment_reward_func` rewards the agent if the `<thought>` block contains words like "buy", "bull", "bear", etc., matching the action.
* **Adversarial Case:** The agent may learn a "Sybil Thought" pattern: dumping a dictionary of keywords (e.g., "Market is BULLISH, will BUY, LONG, UP") into the thought block regardless of the actual chart indicators, just to guarantee the consistency bonus.

### 3. Conservative "Size Squatting" (GRPO)
* **Loopholes:** `governance_reward_func` grants a `+0.20` bonus for using `≤ 80%` of the limit.
* **Adversarial Case:** An agent could learn to trade at exactly `0.05` (the minimum size) every single time. By taking minimal size, it guarantees the conservative bonus and ensures that even if it's wrong, the drawdown is too small to trigger penalties.

### 4. Sharpe Ratio "Ghosting" (PPO)
* **Loopholes:** `sharpe_bonus = w["sharpe"] * np.tanh(sharpe)`. 
* **Adversarial Case:** An agent might take one extremely lucky trade early in the episode and then "squat" on that performance by holding forever. Since the Sharpe ratio is calculated on the return stream, a single high-return step followed by zero-volatility steps can maintain an artificially high Sharpe ratio until the lookback window fully decays the early win.

### 5. The "Invested Squat" (Environment)
* **Loopholes:** The environment treats an agent as "active" if it is invested in >10% of the portfolio (`is_invested`). 
* **Adversarial Case:** A trader could buy exactly `10.1%` of an asset and hold it for the entire 500-step episode. This bypasses the `progressive_hold_cost` and the `inactivity_slam` without ever making another trade, effectively turning the "Active Trader" into a "Buy and Hold" script that avoids all governance scrutiny.

### 6. The "Gov-Block Collusion" Loophole (Environment)
* **Loopholes:** `consecutive_holds` does NOT increment if a trade is blocked by governance (`gov_blocked`).
* **Adversarial Case:** The agents could learn a "Mutual Sabotage" equilibrium. The Risk Manager sets a 0% limit, and the Trader continuously proposes 100% size trades. The environment sees the Trader "trying" and the RM "restricting," so neither is hit by inactivity penalties, yet zero market activity occurs.

### 7. Boundary Surfing (Environment)
* **Loopholes:** Penalties like `utilization_penalty = max(0.0, 0.20 - cap_alloc)` have hard binary-like floors.
* **Adversarial Case:** Agents will almost always learn to "surf" exactly at the `0.20` threshold. While this satisfies the constraint, it suggests the agents aren't learning *nuanced* allocation, but are merely learning to find the "minimum effort" point that zeros out the penalty function.
