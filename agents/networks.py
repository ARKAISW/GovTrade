"""
Neural network architectures for QuantHive governance agents.

All three agents (Risk Manager, Portfolio Manager, Trader) use the same
Actor-Critic MLP architecture with agent-specific input/output dimensions.

Architecture:
  - 3 hidden layers × 256 units
  - LayerNorm + GELU activation
  - Separate actor (policy) and critic (value) heads
  - Continuous actions via Gaussian distribution (mean + log_std)

This is the core infrastructure that enables LEARNED governance policies,
replacing the rule-based heuristics that were the project's fatal flaw.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from typing import Tuple, Dict, Optional


class ActorCriticMLP(nn.Module):
    """Shared Actor-Critic MLP for governance agents.

    The actor head outputs parameters of the action distribution.
    The critic head outputs a scalar value estimate V(s).

    For continuous actions (RM, PM): Gaussian with learnable std.
    For mixed actions (Trader): Categorical direction + Gaussian size.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 256),
        activation: str = "gelu",
        use_layer_norm: bool = True,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Actor head: outputs action mean
        self.actor_mean = nn.Linear(in_dim, act_dim)

        # Learnable log standard deviation (state-independent)
        self.actor_log_std = nn.Parameter(
            torch.full((act_dim,), init_log_std)
        )

        # Critic head: outputs V(s)
        self.critic = nn.Linear(in_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (PPO best practice)."""
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and value."""
        features = self.backbone(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        return action_mean, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, compute log_prob, entropy, and value.

        Args:
            obs: Observation tensor [batch, obs_dim].
            action: If provided, evaluate this action instead of sampling.

        Returns:
            (action, log_prob, entropy, value)
        """
        action_mean, value = self.forward(obs)
        action_std = self.actor_log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value estimate only (for bootstrapping)."""
        features = self.backbone(obs)
        return self.critic(features).squeeze(-1)


class TraderActorCritic(nn.Module):
    """Specialized Actor-Critic for the Trader agent.

    The Trader has a mixed action space:
      - direction: Categorical(3) — Hold/Buy/Sell
      - size: Continuous [0, 1]
      - sl_offset: Continuous [0, ∞) — stop-loss distance as fraction of price
      - tp_offset: Continuous [0, ∞) — take-profit distance as fraction of price

    This network handles the mixed discrete/continuous action space.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 256),
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # Shared backbone
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Direction head: Categorical(3)
        self.direction_logits = nn.Linear(in_dim, 3)

        # Continuous heads: size, sl_offset, tp_offset
        self.continuous_mean = nn.Linear(in_dim, 3)
        self.continuous_log_std = nn.Parameter(torch.full((3,), -0.5))

        # Critic
        self.critic = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.direction_logits.weight, gain=0.01)
        nn.init.zeros_(self.direction_logits.bias)
        nn.init.orthogonal_(self.continuous_mean.weight, gain=0.01)
        nn.init.zeros_(self.continuous_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        dir_logits = self.direction_logits(features)
        cont_mean = self.continuous_mean(features)
        value = self.critic(features).squeeze(-1)
        return dir_logits, cont_mean, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_dir: Optional[torch.Tensor] = None,
        action_cont: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample or evaluate mixed actions.

        Returns dict with keys:
            direction, size, sl_offset, tp_offset,
            log_prob, entropy, value
        """
        dir_logits, cont_mean, value = self.forward(obs)

        # Direction distribution
        dir_dist = Categorical(logits=dir_logits)
        if action_dir is None:
            direction = dir_dist.sample()
        else:
            direction = action_dir
        dir_log_prob = dir_dist.log_prob(direction)
        dir_entropy = dir_dist.entropy()

        # Continuous distribution
        cont_std = self.continuous_log_std.exp().expand_as(cont_mean)
        cont_dist = Normal(cont_mean, cont_std)
        if action_cont is None:
            raw_cont = cont_dist.rsample()
        else:
            raw_cont = action_cont
        cont_log_prob = cont_dist.log_prob(raw_cont).sum(dim=-1)
        cont_entropy = cont_dist.entropy().sum(dim=-1)

        # Apply activation to bound continuous actions
        size = torch.sigmoid(raw_cont[..., 0])         # [0, 1]
        sl_offset = torch.relu(raw_cont[..., 1]) * 0.05  # [0, ~0.25] as price fraction
        tp_offset = torch.relu(raw_cont[..., 2]) * 0.10  # [0, ~0.50] as price fraction

        total_log_prob = dir_log_prob + cont_log_prob
        total_entropy = dir_entropy + cont_entropy

        return {
            "direction": direction,
            "size": size,
            "sl_offset": sl_offset,
            "tp_offset": tp_offset,
            "raw_cont": raw_cont,
            "log_prob": total_log_prob,
            "entropy": total_entropy,
            "value": value,
        }

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.critic(features).squeeze(-1)


# ─── Agent Wrappers ────────────────────────────────────────────────────────────

class LearnedRiskManager:
    """Wraps ActorCriticMLP for the Risk Manager agent.

    Input:  observation (25,) — base_obs + regime_indicator
    Output: action (3,)       — [size_limit, allow_new, force_reduce]
    """

    def __init__(self, obs_dim: int = 25, device: str = "cpu"):
        self.network = ActorCriticMLP(obs_dim=obs_dim, act_dim=3)
        self.device = device
        self.network.to(device)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.network(obs_t)
                action = torch.sigmoid(mean)  # Bound to [0, 1]
            else:
                action, _, _, _ = self.network.get_action_and_value(obs_t)
                action = torch.sigmoid(action)  # Bound to [0, 1]
        return action.squeeze(0).cpu().numpy()

    def parameters(self):
        return self.network.parameters()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)


class LearnedPortfolioManager:
    """Wraps ActorCriticMLP for the Portfolio Manager agent.

    Input:  observation (28,) — base_obs + regime + rm_message
    Output: action (2,)       — [capital_allocation, override_strength]
    """

    def __init__(self, obs_dim: int = 28, device: str = "cpu"):
        self.network = ActorCriticMLP(obs_dim=obs_dim, act_dim=2)
        self.device = device
        self.network.to(device)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.network(obs_t)
                action = torch.sigmoid(mean)
            else:
                action, _, _, _ = self.network.get_action_and_value(obs_t)
                action = torch.sigmoid(action)
        return action.squeeze(0).cpu().numpy()

    def parameters(self):
        return self.network.parameters()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)


class LearnedTrader:
    """Wraps TraderActorCritic for the Trader agent.

    Input:  observation (30,) — base_obs + regime + rm_message + pm_message
    Output: action dict        — {direction, size, sl, tp}
    """

    def __init__(self, obs_dim: int = 30, device: str = "cpu"):
        self.network = TraderActorCritic(obs_dim=obs_dim)
        self.device = device
        self.network.to(device)

    def act(
        self, obs: np.ndarray, deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.network.get_action_and_value(obs_t)

        direction = int(result["direction"].item())
        size = float(result["size"].item())

        return {
            "direction": direction,
            "size": np.array([size], dtype=np.float32),
            "sl": np.array([0.0], dtype=np.float32),  # SL/TP computed by env
            "tp": np.array([0.0], dtype=np.float32),
        }

    def parameters(self):
        return self.network.parameters()

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)
