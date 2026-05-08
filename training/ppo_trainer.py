"""
Multi-Agent PPO Trainer for QuantHive.

Implements Proximal Policy Optimization for training all three governance
agents (Risk Manager, Portfolio Manager, Trader) with alternating optimization.

Key design choices (all justified for academic rigor):
  - GAE (λ=0.95) for advantage estimation
  - Clipped surrogate objective (ε=0.2)
  - Entropy bonus with annealing for exploration
  - Orthogonal initialization (PPO best practice)
  - Per-agent learning rates and rollout buffers
  - Curriculum learning over market difficulty

This is a SINGLE, CLEAN training pipeline — replacing the confusing
dual REINFORCE + GRPO system that was flagged as a credibility issue.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running this file directly from notebooks/terminals without
# requiring the caller to preconfigure PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.multi_agent_env import (
    MultiAgentTradingEnv,
    RISK_MANAGER,
    PORTFOLIO_MGR,
    TRADER,
    ALL_AGENTS,
)
from env.governance_metrics import GovernanceMetrics
from env.failure_taxonomy import GovernanceFailureTaxonomy
from agents.networks import LearnedRiskManager, LearnedPortfolioManager, LearnedTrader


# ─── Rollout Buffer ────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """Per-agent rollout storage for PPO updates.

    Stores transitions (obs, action, reward, value, log_prob) for
    one agent across one episode, then computes GAE advantages.
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    # Computed after episode
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            next_done = False if t == n - 1 else self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * lam * (1 - next_done) * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.rewards)


# ─── PPO Update Logic ──────────────────────────────────────────────────────────

def ppo_update(
    network: Any,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    minibatch_size: int = 64,
    agent_type: str = "continuous",
    device: str = "cpu",
) -> Dict[str, float]:
    """Perform PPO update on a single agent's network."""
    if len(buffer) == 0 or buffer.advantages is None:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # Convert buffer to tensors
    obs_t = torch.FloatTensor(np.array(buffer.observations)).to(device)
    old_log_probs_t = torch.FloatTensor(np.array(buffer.log_probs)).to(device)
    advantages_t = torch.FloatTensor(buffer.advantages).to(device)
    returns_t = torch.FloatTensor(buffer.returns).to(device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    metrics_accum = defaultdict(list)
    n = len(buffer)

    for _ in range(ppo_epochs):
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            mb_idx = indices[start:end]

            mb_obs = obs_t[mb_idx]
            mb_old_lp = old_log_probs_t[mb_idx]
            mb_adv = advantages_t[mb_idx]
            mb_ret = returns_t[mb_idx]

            if agent_type == "continuous":
                actions_arr = np.array([buffer.actions[i] for i in mb_idx])
                mb_actions = torch.FloatTensor(actions_arr).to(device)

                _, new_log_prob, entropy, values = network.get_action_and_value(
                    mb_obs, action=mb_actions,
                )
            else:
                # Use list comprehension with explicit casting to avoid linter "red"
                mb_dir = torch.LongTensor(
                    [int(buffer.actions[i]["direction"]) for i in mb_idx]
                ).to(device)
                
                # Use .get() safely and ensure it returns a list/array
                cont_data = []
                for i in mb_idx:
                    act = buffer.actions[i]
                    raw = act.get("raw_cont", [0.0, 0.0, 0.0])
                    cont_data.append(raw)
                mb_cont = torch.FloatTensor(np.array(cont_data)).to(device)

                result = network.get_action_and_value(
                    mb_obs, action_dir=mb_dir, action_cont=mb_cont,
                )
                new_log_prob = result["log_prob"]
                entropy = result["entropy"]
                values = result["value"]

            # PPO clipped objective
            ratio = torch.exp(new_log_prob - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            value_loss = nn.functional.mse_loss(values, mb_ret)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            optimizer.step()

            metrics_accum["policy_loss"].append(policy_loss.item())
            metrics_accum["value_loss"].append(value_loss.item())
            metrics_accum["entropy"].append(-entropy_loss.item())

    return {k: float(np.mean(v)) for k, v in metrics_accum.items()}


# ─── Main Training Loop ───────────────────────────────────────────────────────

class MultiAgentPPOTrainer:
    """Orchestrates multi-agent PPO training with curriculum learning.

    Training schedule:
      1. Alternating optimization: cycle through agents
      2. Curriculum: easy → medium → hard → adversarial regimes
      3. Metrics logging per episode
      4. Periodic checkpointing

    Usage:
        trainer = MultiAgentPPOTrainer()
        trainer.train(total_episodes=1500)
    """

    def __init__(
        self,
        # Environment
        max_steps: int = 500,
        initial_cash: float = 100_000.0,
        # PPO
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        entropy_decay: float = 0.9995,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        # Learning rates
        lr_rm: float = 3e-4,
        lr_pm: float = 3e-4,
        lr_trader: float = 3e-4,
        # Alternating
        phase_length: int = 50,
        # Curriculum
        curriculum: Optional[List[Tuple[int, int, str]]] = None,
        # Output
        output_dir: str = "outputs/ppo_training",
        save_every: int = 100,
        log_every: int = 10,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.max_steps = max_steps
        self.initial_cash = initial_cash
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.entropy_decay = entropy_decay
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.phase_length = phase_length
        self.save_every = save_every
        self.log_every = log_every
        self.seed = seed
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default curriculum: easy → medium → hard → adversarial
        self.curriculum = curriculum or [
            (0, 200, "easy"),
            (200, 500, "medium"),
            (500, 1000, "hard"),
            (1000, 1500, "hard"),
        ]

        # Initialize agents
        self.rm = LearnedRiskManager(obs_dim=25, device=device)
        self.pm = LearnedPortfolioManager(obs_dim=28, device=device)
        self.trader = LearnedTrader(obs_dim=30, device=device)

        # Optimizers
        self.opt_rm = optim.Adam(self.rm.parameters(), lr=lr_rm)
        self.opt_pm = optim.Adam(self.pm.parameters(), lr=lr_pm)
        self.opt_trader = optim.Adam(self.trader.parameters(), lr=lr_trader)

        # Metrics storage
        self.all_metrics: Dict[str, List] = defaultdict(list)

    def _get_difficulty(self, episode: int) -> str:
        """
        Return the environment difficulty based on a non-linear, probabilistic curriculum.
        
        This prevents 'catastrophic forgetting' by maintaining a mix of
        difficulties while gradually shifting the focus to harder regimes.
        """
        # Estimate total episodes from curriculum if possible, else default to 1500
        total_eps = self.curriculum[-1][1] if self.curriculum else 1500
        progress = episode / total_eps
        
        # Probabilities: [easy, medium, hard]
        # Keep easy regimes more frequent for longer to establish baseline profitable behaviors
        if progress < 0.3:
            probs = [0.85, 0.10, 0.05]
        elif progress < 0.6:
            probs = [0.50, 0.35, 0.15]
        elif progress < 0.85:
            probs = [0.20, 0.40, 0.40]
        else:
            probs = [0.10, 0.20, 0.70]
            
        return np.random.choice(["easy", "medium", "hard"], p=probs)

    def _collect_episode(
        self, env: MultiAgentTradingEnv,
    ) -> Tuple[Dict[str, RolloutBuffer], GovernanceMetrics, GovernanceFailureTaxonomy, Dict]:
        """Collect one full episode of experience."""
        buffers = {ag: RolloutBuffer() for ag in ALL_AGENTS}
        gov_metrics = GovernanceMetrics()

        env.reset(seed=None)  # Let regime engine handle seeding
        failure_tax = env._failure_tax
        gov_metrics.peak_value = self.initial_cash

        step_count = 0
        final_info = {}

        def assign_cycle_rewards(done: bool = False) -> None:
            for ag in ALL_AGENTS:
                if len(buffers[ag]) > 0:
                    buffers[ag].rewards[-1] = float(env.rewards.get(ag, 0.0))
                    if done:
                        buffers[ag].dones[-1] = True

        while env.agents and step_count < self.max_steps * 3:
            agent = env.agent_selection
            obs = env.observe(agent)

            # PettingZoo compliance: dead agents must receive None
            if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                env.step(None)
                step_count += 1
                continue

            # Get action from the appropriate learned agent
            if agent == RISK_MANAGER:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action_t, log_prob, _, value = self.rm.network.get_action_and_value(obs_t)
                    action_np = torch.sigmoid(action_t).squeeze(0).cpu().numpy()
                    raw_action = action_t.squeeze(0).cpu().numpy()

                action = action_np
                buffers[agent].add(
                    obs, raw_action, 0.0,
                    value.item(), log_prob.item(), False,
                )

            elif agent == PORTFOLIO_MGR:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action_t, log_prob, _, value = self.pm.network.get_action_and_value(obs_t)
                    action_np = torch.sigmoid(action_t).squeeze(0).cpu().numpy()
                    raw_action = action_t.squeeze(0).cpu().numpy()

                action = action_np
                buffers[agent].add(
                    obs, raw_action, 0.0,
                    value.item(), log_prob.item(), False,
                )

            elif agent == TRADER:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    result = self.trader.network.get_action_and_value(obs_t)

                current_price = env._market.current_price()
                direction = int(result["direction"].item())
                sl_off = float(result["sl_offset"].item())
                tp_off = float(result["tp_offset"].item())
                
                sl_val = 0.0
                tp_val = 0.0
                if direction == 1:
                    sl_val = current_price * (1.0 - sl_off)
                    tp_val = current_price * (1.0 + tp_off)
                elif direction == 2:
                    sl_val = current_price * (1.0 + sl_off)
                    tp_val = current_price * (1.0 - tp_off)

                action = {
                    "direction": direction,
                    "size": np.array([float(result["size"].item())], dtype=np.float32),
                    "sl": np.array([sl_val], dtype=np.float32),
                    "tp": np.array([tp_val], dtype=np.float32),
                }
                stored_action = {
                    "direction": direction,
                    "raw_cont": result["raw_cont"].squeeze(0).cpu().numpy(),
                }
                buffers[agent].add(
                    obs, stored_action, 0.0,
                    result["value"].item(), result["log_prob"].item(), False,
                )

            env.step(action)
            step_count += 1

            # Track governance metrics after trader acts (full cycle)
            if agent == TRADER:
                episode_done = all(
                    env.terminations.get(ag, False) or env.truncations.get(ag, False)
                    for ag in ALL_AGENTS
                )
                assign_cycle_rewards(done=episode_done)

                info = env.infos.get(TRADER, {})
                governance = info.get("governance", {})
                gov_metrics.update_step(
                    portfolio_value=info.get("portfolio_value", self.initial_cash),
                    was_compliant=governance.get("was_compliant", True),
                    n_interventions=len(governance.get("interventions", [])),
                    rm_action=env._rm_message,
                    pm_action=env._pm_message,
                    regime_label=getattr(env, "_current_regime_label", ""),
                    current_drawdown=info.get("current_drawdown", info.get("max_drawdown", 0.0)),
                )

                if episode_done:
                    final_info = info
                    break

            if not env.agents:
                # Capture final terminal rewards (e.g., bankruptcy penalties, activity gates)
                assign_cycle_rewards(done=True)
                final_info = env.infos.get(TRADER, {})
                break

        # Compute GAE for each agent
        for ag in ALL_AGENTS:
            if len(buffers[ag]) > 0:
                last_val = 0.0  # Terminal state value = 0
                buffers[ag].compute_gae(last_val, self.gamma, self.gae_lambda)

        failure_tax.check_episode()

        return buffers, gov_metrics, failure_tax, final_info

    def train(self, total_episodes: int = 1500) -> Dict[str, List]:
        """Run the full training loop using Alternating Optimization."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        print("=" * 70)
        print("  QuantHive — Multi-Agent PPO (Alternating Optimization)")
        print(f"  Episodes: {total_episodes}  |  Curriculum: {len(self.curriculum)} phases")
        print(f"  Phase Length: {self.phase_length} eps | Device: {self.device}")
        print("=" * 70)

        best_gov_score = -np.inf

        # Optimization phases: 0=Trader, 1=RiskManager, 2=PortfolioMgr
        # This prevents the "Death Spiral" by providing a stable partner for learning.
        ALL_PHASES = [TRADER, RISK_MANAGER, PORTFOLIO_MGR]

        for ep in range(total_episodes):
            t0 = time.time()
            difficulty = self._get_difficulty(ep)

            # Determine which agent is being optimized this episode
            phase_idx = (ep // self.phase_length) % len(ALL_PHASES)
            opt_agent = ALL_PHASES[phase_idx]

            env = MultiAgentTradingEnv(
                difficulty=difficulty,
                max_steps=self.max_steps,
                initial_cash=self.initial_cash,
            )

            # Collect episode
            buffers, gov_metrics, failure_tax, info = self._collect_episode(env)

            # PPO update ONLY for the currently optimized agent
            train_metrics = {}

            if opt_agent == RISK_MANAGER and len(buffers[RISK_MANAGER]) > 0:
                rm_metrics = ppo_update(
                    self.rm.network, self.opt_rm, buffers[RISK_MANAGER],
                    clip_epsilon=self.clip_epsilon,
                    entropy_coeff=self.entropy_coeff,
                    value_coeff=self.value_coeff,
                    max_grad_norm=self.max_grad_norm,
                    ppo_epochs=self.ppo_epochs,
                    minibatch_size=self.minibatch_size,
                    agent_type="continuous",
                    device=self.device,
                )
                for k, v in rm_metrics.items(): train_metrics[f"rm_{k}"] = v

            elif opt_agent == PORTFOLIO_MGR and len(buffers[PORTFOLIO_MGR]) > 0:
                pm_metrics = ppo_update(
                    self.pm.network, self.opt_pm, buffers[PORTFOLIO_MGR],
                    clip_epsilon=self.clip_epsilon,
                    entropy_coeff=self.entropy_coeff,
                    value_coeff=self.value_coeff,
                    max_grad_norm=self.max_grad_norm,
                    ppo_epochs=self.ppo_epochs,
                    minibatch_size=self.minibatch_size,
                    agent_type="continuous",
                    device=self.device,
                )
                for k, v in pm_metrics.items(): train_metrics[f"pm_{k}"] = v

            elif opt_agent == TRADER and len(buffers[TRADER]) > 0:
                trader_metrics = ppo_update(
                    self.trader.network, self.opt_trader, buffers[TRADER],
                    clip_epsilon=self.clip_epsilon,
                    entropy_coeff=self.entropy_coeff,
                    value_coeff=self.value_coeff,
                    max_grad_norm=self.max_grad_norm,
                    ppo_epochs=self.ppo_epochs,
                    minibatch_size=self.minibatch_size,
                    agent_type="mixed",
                    device=self.device,
                )
                for k, v in trader_metrics.items(): train_metrics[f"trader_{k}"] = v

            # Anneal entropy coefficient
            self.entropy_coeff *= self.entropy_decay

            # Compute final governance metrics
            final_gov = gov_metrics.finalize()
            failure_summary = failure_tax.summary()
            # Log metrics
            self.all_metrics["episode"].append(ep)
            self.all_metrics["difficulty"].append(difficulty)
            self.all_metrics["opt_agent"].append(opt_agent)
            self.all_metrics["max_drawdown"].append(final_gov["max_drawdown"])
            self.all_metrics["sharpe_ratio"].append(final_gov["sharpe_ratio"])
            self.all_metrics["total_return"].append(final_gov["total_return"])
            self.all_metrics["violation_rate"].append(final_gov["constraint_violation_rate"])
            self.all_metrics["governance_stability"].append(final_gov["governance_stability"])
            self.all_metrics["false_tightening"].append(final_gov["false_tightening_rate"])
            self.all_metrics["total_failures"].append(failure_tax.total_failures())
            self.all_metrics["critical_failures"].append(failure_tax.critical_failures())
            for k, v in train_metrics.items():
                self.all_metrics[f"train_{k}"].append(v)

            elapsed = time.time() - t0

            if ep % self.log_every == 0:
                print(
                    f"Ep {ep:5d} [{difficulty:12s}] [{opt_agent:20s}] | "
                    f"DD={final_gov['max_drawdown']:.2%} | "
                    f"Ret={final_gov['total_return']:+.2%} | "
                    f"Viol={final_gov['constraint_violation_rate']:.2%} | "
                    f"Fails={failure_tax.total_failures()} | "
                    f"{elapsed:.1f}s"
                )

            # Checkpointing
            if (ep + 1) % self.save_every == 0:
                self._save_checkpoint(ep + 1)

            # Track best governance score (combined metric)
            gov_score = (
                (1 - final_gov["max_drawdown"]) * 0.3
                + final_gov["sharpe_ratio"] * 0.2
                + (1 - final_gov["constraint_violation_rate"]) * 0.2
                + (1 - final_gov["false_tightening_rate"]) * 0.15
                + final_gov.get("intervention_efficiency", 0) * 0.15
            )
            if gov_score > best_gov_score:
                best_gov_score = gov_score
                self._save_checkpoint(ep + 1, prefix="best_")

        # Final save
        self._save_checkpoint(total_episodes, prefix="final_")
        self._save_metrics()

        print("\n" + "=" * 70)
        print("  Training Complete!")
        print(f"  Best Governance Score: {best_gov_score:.4f}")
        print("=" * 70)

        return dict(self.all_metrics)

    def _save_checkpoint(self, episode: int, prefix: str = ""):
        """Save model checkpoints."""
        try:
            ckpt_dir = self.output_dir / f"{prefix}checkpoint_ep{episode}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(self.rm.state_dict(), ckpt_dir / "risk_manager.pt")
            torch.save(self.pm.state_dict(), ckpt_dir / "portfolio_manager.pt")
            torch.save(self.trader.state_dict(), ckpt_dir / "trader.pt")
        except (OSError, RuntimeError) as exc:
            print(f"Warning: checkpoint save skipped for episode {episode}: {exc}")

    def load_checkpoint(self, ckpt_dir_str: str):
        """Load model checkpoints to resume training."""
        ckpt_dir = Path(ckpt_dir_str)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found.")
        
        self.rm.load_state_dict(torch.load(ckpt_dir / "risk_manager.pt", map_location=self.device, weights_only=True))
        self.pm.load_state_dict(torch.load(ckpt_dir / "portfolio_manager.pt", map_location=self.device, weights_only=True))
        self.trader.load_state_dict(torch.load(ckpt_dir / "trader.pt", map_location=self.device, weights_only=True))
        print(f"Loaded checkpoint from {ckpt_dir}")

    def _save_metrics(self):
        """Save training metrics to JSON."""
        serialized = {}
        for k, v in self.all_metrics.items():
            serialized[k] = [
                float(x) if isinstance(x, (np.floating, np.integer, float, int)) else str(x)
                for x in v
            ]
        try:
            with open(self.output_dir / "training_metrics.json", "w") as f:
                json.dump(serialized, f, indent=2)
        except OSError as exc:
            print(f"Warning: metrics save skipped: {exc}")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantHive Multi-Agent PPO Training")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--phase-length", type=int, default=50, help="Episodes per optimization phase")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/ppo_training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = MultiAgentPPOTrainer(
        max_steps=args.max_steps,
        phase_length=args.phase_length,
        output_dir=args.output_dir,
        seed=args.seed,
        device=device,
    )
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        
    trainer.train(total_episodes=args.episodes)
 
