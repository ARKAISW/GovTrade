"""
Governance quality metrics for QuantHive.

These metrics measure the QUALITY of learned governance, NOT trading profit.
This is the key conceptual shift: the primary evaluation object is whether
supervisory agents learn better institutional controls than static rules.

Metrics are organized into three tiers:
  1. Capital Protection   — drawdown, tail risk, loss containment
  2. Governance Behavior  — intervention efficiency, response latency, stability
  3. Counterfactual Value — loss prevented by governance vs. no-governance world
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GovernanceMetrics:
    """Accumulated governance metrics over an evaluation episode.

    All metrics are computed incrementally during episode execution
    and finalized at episode end.
    """

    # ─── Capital Protection ─────────────────────────────────────────────
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_value: float = 0.0
    returns: List[float] = field(default_factory=list)

    # ─── Governance Behavior ────────────────────────────────────────────
    intervention_count: int = 0
    compliant_count: int = 0
    total_actions: int = 0
    rm_actions: List[np.ndarray] = field(default_factory=list)
    pm_actions: List[np.ndarray] = field(default_factory=list)
    regime_labels: List[str] = field(default_factory=list)
    drawdown_at_intervention: List[float] = field(default_factory=list)
    steps_to_first_intervention: Optional[int] = None

    # ─── Counterfactual Tracking ────────────────────────────────────────
    governed_values: List[float] = field(default_factory=list)
    ungoverned_values: List[float] = field(default_factory=list)

    def update_step(
        self,
        portfolio_value: float,
        was_compliant: bool,
        n_interventions: int,
        rm_action: Optional[np.ndarray] = None,
        pm_action: Optional[np.ndarray] = None,
        regime_label: str = "",
        current_drawdown: float = 0.0,
    ):
        """Update metrics with data from one step."""
        # Capital protection
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        self.current_drawdown = current_drawdown
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        if len(self.governed_values) > 0:
            ret = (portfolio_value - self.governed_values[-1]) / (self.governed_values[-1] + 1e-10)
            self.returns.append(ret)

        self.governed_values.append(portfolio_value)

        # Governance behavior
        self.total_actions += 1
        if was_compliant:
            self.compliant_count += 1
        if n_interventions > 0:
            self.intervention_count += n_interventions
            self.drawdown_at_intervention.append(current_drawdown)
            if self.steps_to_first_intervention is None:
                self.steps_to_first_intervention = self.total_actions

        if rm_action is not None:
            self.rm_actions.append(rm_action.copy())
        if pm_action is not None:
            self.pm_actions.append(pm_action.copy())

        self.regime_labels.append(regime_label)

    def finalize(self) -> Dict[str, float]:
        """Compute all final metrics at episode end."""
        metrics: Dict[str, float] = {}

        # ── Capital Protection ──────────────────────────────────────────
        metrics["max_drawdown"] = self.max_drawdown
        metrics["tail_loss_containment"] = self._tail_loss_containment()
        metrics["sortino_ratio"] = self._sortino_ratio()
        metrics["calmar_ratio"] = self._calmar_ratio()

        # ── Governance Behavior ─────────────────────────────────────────
        metrics["constraint_violation_rate"] = self._violation_rate()
        metrics["intervention_efficiency"] = self._intervention_efficiency()
        metrics["intervention_latency"] = self._intervention_latency()
        metrics["governance_stability"] = self._governance_stability()
        metrics["false_tightening_rate"] = self._false_tightening_rate()
        metrics["capital_utilization"] = self._capital_utilization()

        # ── Recovery ────────────────────────────────────────────────────
        metrics["regime_recovery_time"] = self._regime_recovery_time()

        # ── Counterfactual ──────────────────────────────────────────────
        metrics["counterfactual_loss_prevented"] = self._counterfactual_loss()

        # ── Standard Performance ────────────────────────────────────────
        metrics["total_return"] = self._total_return()
        metrics["sharpe_ratio"] = self._sharpe_ratio()

        return metrics

    # ─── Metric Implementations ─────────────────────────────────────────

    def _tail_loss_containment(self) -> float:
        """Ratio of VaR_95 to VaR_99. Higher = better tail containment."""
        if len(self.returns) < 20:
            return 0.0
        returns = np.array(self.returns)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        if abs(var_99) < 1e-10:
            return 1.0
        return float(np.clip(var_95 / (var_99 + 1e-10), 0.0, 1.0))

    def _sortino_ratio(self) -> float:
        """Sortino ratio: return / downside deviation."""
        if len(self.returns) < 2:
            return 0.0
        returns = np.array(self.returns)
        downside = returns[returns < 0]
        if len(downside) < 2 or np.std(downside) < 1e-10:
            return 0.0
        return float(np.mean(returns) / (np.std(downside) + 1e-10))

    def _calmar_ratio(self) -> float:
        """Calmar ratio: annualized return / max drawdown."""
        if self.max_drawdown < 1e-10 or len(self.returns) < 2:
            return 0.0
        total_ret = self._total_return()
        return float(total_ret / (self.max_drawdown + 1e-10))

    def _violation_rate(self) -> float:
        """Fraction of actions that triggered governance intervention."""
        if self.total_actions == 0:
            return 0.0
        return 1.0 - (self.compliant_count / self.total_actions)

    def _intervention_efficiency(self) -> float:
        """Ratio: interventions that happened during real stress vs total.
        Higher = RM only intervenes when needed (less over-regulation).
        """
        if self.intervention_count == 0:
            return 1.0  # No interventions = perfectly efficient (or inactive)
        # Count interventions during genuine stress (drawdown > 10%)
        stress_interventions = sum(1 for dd in self.drawdown_at_intervention if dd > 0.10)
        return float(stress_interventions / (self.intervention_count + 1e-10))

    def _intervention_latency(self) -> float:
        """Average steps between drawdown exceeding 10% and first RM tightening.
        Lower = faster response.
        """
        if not self.rm_actions or len(self.rm_actions) < 2:
            return float("inf")

        rm_arr = np.array(self.rm_actions)
        latencies: List[int] = []
        stress_started = None

        for i in range(len(self.governed_values)):
            dd = 0.0
            if self.peak_value > 0:
                dd = (self.peak_value - self.governed_values[i]) / (self.peak_value + 1e-10)

            if dd > 0.10 and stress_started is None:
                stress_started = i

            # Check if RM tightened (size_limit < 0.3)
            if stress_started is not None and i < len(rm_arr):
                if rm_arr[i][0] < 0.3:
                    latencies.append(i - stress_started)
                    stress_started = None

        return float(np.mean(latencies)) if latencies else float("inf")

    def _governance_stability(self) -> float:
        """Standard deviation of RM actions in stable (non-stress) regimes.
        Lower = more consistent governance.
        """
        if not self.rm_actions:
            return 0.0

        rm_arr = np.array(self.rm_actions)
        stable_mask = np.array([
            r in ("bull_steady", "mean_revert", "sideways_choppy", "")
            for r in self.regime_labels[:len(rm_arr)]
        ])

        if stable_mask.sum() < 5:
            return 0.0

        stable_actions = rm_arr[stable_mask]
        return float(np.mean(np.std(stable_actions, axis=0)))

    def _false_tightening_rate(self) -> float:
        """Fraction of RM tightening events (size_limit < 0.3) during bull regimes."""
        if not self.rm_actions:
            return 0.0

        rm_arr = np.array(self.rm_actions)
        false_tighten = 0
        total_tighten = 0

        for i, action in enumerate(rm_arr):
            if action[0] < 0.3:  # RM is restricting
                total_tighten += 1
                regime = self.regime_labels[i] if i < len(self.regime_labels) else ""
                if regime in ("bull_steady", "mean_revert"):
                    false_tighten += 1  # Restricting during safe regime

        if total_tighten == 0:
            return 0.0
        return float(false_tighten / total_tighten)

    def _capital_utilization(self) -> float:
        """Average capital allocation from PM actions."""
        if not self.pm_actions:
            return 0.5
        pm_arr = np.array(self.pm_actions)
        return float(np.mean(pm_arr[:, 0]))

    def _regime_recovery_time(self) -> float:
        """Average steps to recover 50% of crash loss after crash regimes."""
        if len(self.governed_values) < 10:
            return float("inf")

        recovery_times: List[int] = []
        values = np.array(self.governed_values)

        # Find crash periods
        i = 0
        while i < len(values) - 1:
            regime = self.regime_labels[i] if i < len(self.regime_labels) else ""
            if regime in ("crash", "flash_crash", "cascading_liquidation", "correlated_selloff"):
                # Find the trough
                crash_start_val = values[i]
                trough_val = values[i]
                trough_idx = i

                j = i + 1
                while j < len(values) and self.regime_labels[j] == regime:
                    if values[j] < trough_val:
                        trough_val = values[j]
                        trough_idx = j
                    j += 1

                # Find 50% recovery
                loss = crash_start_val - trough_val
                recovery_target = trough_val + loss * 0.5

                k = trough_idx + 1
                while k < len(values):
                    if values[k] >= recovery_target:
                        recovery_times.append(k - trough_idx)
                        break
                    k += 1

                i = j
            else:
                i += 1

        return float(np.mean(recovery_times)) if recovery_times else float("inf")

    def _counterfactual_loss(self) -> float:
        """PnL difference: governed world vs ungoverned world."""
        if not self.governed_values or not self.ungoverned_values:
            return 0.0
        gov_return = (self.governed_values[-1] - self.governed_values[0]) / (self.governed_values[0] + 1e-10)
        ungov_return = (self.ungoverned_values[-1] - self.ungoverned_values[0]) / (self.ungoverned_values[0] + 1e-10)
        return float(gov_return - ungov_return)

    def _total_return(self) -> float:
        if len(self.governed_values) < 2:
            return 0.0
        return float((self.governed_values[-1] - self.governed_values[0]) / (self.governed_values[0] + 1e-10))

    def _sharpe_ratio(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        returns = np.array(self.returns)
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        return float(np.mean(returns) / std)
