"""
Governance Failure Taxonomy for QuantHive.

Classifies governance failures into named, measurable categories.
This transforms vague "the system didn't work" into precise
institutional failure modes — critical for research credibility.

Each failure type has:
  - A formal definition (trigger condition)
  - A severity level
  - A measurable count per episode

Reporting failure distributions (not just aggregate reward) is
what makes this project publishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

import numpy as np


class FailureSeverity(Enum):
    """Severity classification for governance failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureEvent:
    """A single governance failure occurrence."""
    failure_type: str
    step: int
    severity: FailureSeverity
    details: Dict[str, Any] = field(default_factory=dict)


class GovernanceFailureTaxonomy:
    """Detects and classifies governance failures during episode execution.

    Failure Types:
    ─────────────────────────────────────────────────────────────
    1. OVERREACTION
       RM reduces size_limit > 80% during < 5% drawdown.
       Severity: MEDIUM
       Impact: Missed profit opportunities, capital starvation.

    2. DELAYED_INTERVENTION
       Drawdown exceeds 20% before RM tightens constraints.
       Severity: HIGH
       Impact: Preventable capital loss.

    3. FALSE_CONSTRAINT_TIGHTENING
       RM tightens (size_limit < 0.2) during bull regime with
       drawdown < 3%.
       Severity: LOW
       Impact: Reduced capital utilization.

    4. CAPITAL_STARVATION
       PM allocation < 10% for > 20 consecutive steps.
       Severity: HIGH
       Impact: Agent unable to trade despite good conditions.

    5. COLLAPSE_CASCADE
       Portfolio value drops > 30% in < 10 steps despite
       governance being active.
       Severity: CRITICAL
       Impact: Total governance failure under stress.

    6. OVERRIDE_THRASHING
       PM override flag flips > 5 times in 10 consecutive steps.
       Severity: MEDIUM
       Impact: Policy instability, inconsistent signaling.

    7. INACTION_UNDER_STRESS
       RM keeps size_limit > 0.5 while drawdown > 15%.
       Severity: HIGH
       Impact: Failure to protect capital during clear stress.

    8. TRIVIAL_SAFETY
       RM always outputs size_limit < 0.05 regardless of conditions.
       Severity: MEDIUM
       Impact: Renders governance meaningless (always blocking).

    9. BANKRUPTCY
       Portfolio value falls below 10% of initial capital.
       Severity: CRITICAL
       Impact: Total loss of capital, definitive governance failure.
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self):
        self.failures: List[FailureEvent] = []
        self._recent_rm_actions: List[np.ndarray] = []
        self._recent_pm_overrides: List[float] = []
        self._recent_values: List[float] = []
        self._low_alloc_streak: int = 0
        self._stress_unresponded_steps: int = 0
        self._initial_value: float = 0.0

    def check_step(
        self,
        step: int,
        rm_action: np.ndarray,
        pm_action: np.ndarray,
        portfolio_value: float,
        drawdown: float,
        regime_label: str,
    ):
        """Check for governance failures at this step."""
        if not self._initial_value and portfolio_value > 0:
            self._initial_value = portfolio_value

        self._recent_rm_actions.append(rm_action.copy())
        self._recent_pm_overrides.append(float(pm_action[1]))
        self._recent_values.append(portfolio_value)

        size_limit = float(rm_action[0])
        cap_alloc = float(pm_action[0])
        override = float(pm_action[1])

        # 9. BANKRUPTCY
        if self._initial_value > 0 and portfolio_value < self._initial_value * 0.10:
            self.failures.append(FailureEvent(
                failure_type="bankruptcy",
                step=step,
                severity=FailureSeverity.CRITICAL,
                details={"portfolio_value": portfolio_value, "initial_value": self._initial_value},
            ))

        # 1. OVERREACTION
        if size_limit < 0.2 and drawdown < 0.05:
            self.failures.append(FailureEvent(
                failure_type="overreaction",
                step=step,
                severity=FailureSeverity.MEDIUM,
                details={"size_limit": size_limit, "drawdown": drawdown},
            ))

        # 2. DELAYED_INTERVENTION
        if drawdown > 0.20 and size_limit > 0.3:
            self._stress_unresponded_steps += 1
            if self._stress_unresponded_steps >= 3:
                self.failures.append(FailureEvent(
                    failure_type="delayed_intervention",
                    step=step,
                    severity=FailureSeverity.HIGH,
                    details={"drawdown": drawdown, "size_limit": size_limit,
                             "unresponded_steps": self._stress_unresponded_steps},
                ))
        else:
            self._stress_unresponded_steps = 0

        # 3. FALSE_CONSTRAINT_TIGHTENING
        if (size_limit < 0.2 and drawdown < 0.03 and
                regime_label in ("bull_steady", "mean_revert")):
            self.failures.append(FailureEvent(
                failure_type="false_constraint_tightening",
                step=step,
                severity=FailureSeverity.LOW,
                details={"size_limit": size_limit, "regime": regime_label},
            ))

        # 4. CAPITAL_STARVATION
        if cap_alloc < 0.10:
            self._low_alloc_streak += 1
            if self._low_alloc_streak > 20:
                self.failures.append(FailureEvent(
                    failure_type="capital_starvation",
                    step=step,
                    severity=FailureSeverity.HIGH,
                    details={"allocation": cap_alloc, "streak": self._low_alloc_streak},
                ))
        else:
            self._low_alloc_streak = 0

        # 5. COLLAPSE_CASCADE
        if len(self._recent_values) >= 10:
            val_10_ago = self._recent_values[-10]
            if val_10_ago > 0 and portfolio_value < val_10_ago * 0.70:
                self.failures.append(FailureEvent(
                    failure_type="collapse_cascade",
                    step=step,
                    severity=FailureSeverity.CRITICAL,
                    details={
                        "value_10_ago": val_10_ago,
                        "current_value": portfolio_value,
                        "drop_pct": (val_10_ago - portfolio_value) / val_10_ago,
                    },
                ))

        # 6. OVERRIDE_THRASHING
        if len(self._recent_pm_overrides) >= 10:
            recent = self._recent_pm_overrides[-10:]
            # Count flips (override > 0.5 → < 0.5 or vice versa)
            flips = sum(
                1 for i in range(1, len(recent))
                if (recent[i] > 0.5) != (recent[i - 1] > 0.5)
            )
            if flips > 5:
                self.failures.append(FailureEvent(
                    failure_type="override_thrashing",
                    step=step,
                    severity=FailureSeverity.MEDIUM,
                    details={"flips_in_10_steps": flips},
                ))

        # 7. INACTION_UNDER_STRESS
        if drawdown > 0.15 and size_limit > 0.5:
            self.failures.append(FailureEvent(
                failure_type="inaction_under_stress",
                step=step,
                severity=FailureSeverity.HIGH,
                details={"drawdown": drawdown, "size_limit": size_limit},
            ))

    def check_episode(self):
        """Check for episode-level governance failures."""
        # 8. TRIVIAL_SAFETY
        if len(self._recent_rm_actions) > 50:
            rm_arr = np.array(self._recent_rm_actions)
            mean_limit = np.mean(rm_arr[:, 0])
            std_limit = np.std(rm_arr[:, 0])
            if mean_limit < 0.05 and std_limit < 0.02:
                self.failures.append(FailureEvent(
                    failure_type="trivial_safety",
                    step=-1,
                    severity=FailureSeverity.MEDIUM,
                    details={"mean_size_limit": float(mean_limit),
                             "std_size_limit": float(std_limit)},
                ))

    def summary(self) -> Dict[str, int]:
        """Count failures by type."""
        counts: Dict[str, int] = {}
        for f in self.failures:
            counts[f.failure_type] = counts.get(f.failure_type, 0) + 1
        return counts

    def severity_counts(self) -> Dict[str, int]:
        """Count failures by severity level."""
        counts: Dict[str, int] = {s.value: 0 for s in FailureSeverity}
        for f in self.failures:
            counts[f.severity.value] += 1
        return counts

    def total_failures(self) -> int:
        return len(self.failures)

    def critical_failures(self) -> int:
        return sum(1 for f in self.failures if f.severity == FailureSeverity.CRITICAL)
