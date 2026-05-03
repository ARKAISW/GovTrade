"""Tests for governance logic — interventions, compliance tracking, failure taxonomy."""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.governance_metrics import GovernanceMetrics
from env.failure_taxonomy import GovernanceFailureTaxonomy, FailureSeverity


class TestGovernanceMetrics:
    def test_initial_state(self):
        gm = GovernanceMetrics()
        assert gm.max_drawdown == 0.0
        assert gm.total_actions == 0
        assert gm.intervention_count == 0

    def test_step_tracking(self):
        gm = GovernanceMetrics()
        gm.peak_value = 100000
        gm.update_step(100000, True, 0, np.array([0.5, 1.0, 0.0]),
                        np.array([0.6, 0.0]), "bull_steady", 0.0)
        assert gm.total_actions == 1
        assert gm.compliant_count == 1
        assert gm.intervention_count == 0

    def test_intervention_tracking(self):
        gm = GovernanceMetrics()
        gm.peak_value = 100000
        gm.update_step(95000, False, 2, np.array([0.3, 1.0, 0.0]),
                        np.array([0.6, 0.0]), "bear_steady", 0.05)
        assert gm.intervention_count == 2
        assert gm.compliant_count == 0

    def test_finalize_returns_all_keys(self):
        gm = GovernanceMetrics()
        gm.peak_value = 100000
        for i in range(50):
            val = 100000 - i * 100
            gm.update_step(val, True, 0, np.array([0.5, 1.0, 0.0]),
                            np.array([0.6, 0.0]), "bull_steady", i * 0.001)
        result = gm.finalize()
        expected_keys = {"max_drawdown", "tail_loss_containment", "sortino_ratio",
                         "constraint_violation_rate", "total_return", "sharpe_ratio"}
        assert expected_keys.issubset(set(result.keys()))

    def test_drawdown_computation(self):
        gm = GovernanceMetrics()
        gm.peak_value = 100000
        gm.update_step(100000, True, 0, regime_label="bull_steady", current_drawdown=0.0)
        gm.update_step(90000, True, 0, regime_label="bear_steady", current_drawdown=0.10)
        gm.update_step(85000, True, 0, regime_label="crash", current_drawdown=0.15)
        assert gm.max_drawdown == 0.15


class TestFailureTaxonomy:
    def test_overreaction_detection(self):
        ft = GovernanceFailureTaxonomy()
        # RM restricts heavily (size_limit=0.1) with low drawdown (0.02)
        ft.check_step(1, np.array([0.1, 1.0, 0.0]), np.array([0.6, 0.0]),
                       100000, 0.02, "bull_steady")
        summary = ft.summary()
        assert "overreaction" in summary

    def test_no_false_positive_in_stress(self):
        ft = GovernanceFailureTaxonomy()
        # RM restricts during genuine stress — NOT an overreaction
        ft.check_step(1, np.array([0.1, 0.0, 1.0]), np.array([0.6, 0.0]),
                       80000, 0.20, "crash")
        summary = ft.summary()
        assert summary.get("overreaction", 0) == 0

    def test_capital_starvation_detection(self):
        ft = GovernanceFailureTaxonomy()
        for i in range(25):
            ft.check_step(i, np.array([0.5, 1.0, 0.0]), np.array([0.05, 0.0]),
                           100000, 0.0, "bull_steady")
        assert ft.summary().get("capital_starvation", 0) > 0

    def test_collapse_cascade_detection(self):
        ft = GovernanceFailureTaxonomy()
        for i in range(15):
            val = 100000 - i * 5000  # Drops 50k in 10 steps
            ft.check_step(i, np.array([0.5, 1.0, 0.0]), np.array([0.6, 0.0]),
                           val, (100000 - val) / 100000, "crash")
        assert ft.summary().get("collapse_cascade", 0) > 0

    def test_severity_classification(self):
        ft = GovernanceFailureTaxonomy()
        # Trigger an overreaction (MEDIUM)
        ft.check_step(1, np.array([0.1, 1.0, 0.0]), np.array([0.6, 0.0]),
                       100000, 0.02, "bull_steady")
        counts = ft.severity_counts()
        assert counts["medium"] >= 1

    def test_trivial_safety_episode_check(self):
        ft = GovernanceFailureTaxonomy()
        for i in range(60):
            ft.check_step(i, np.array([0.02, 1.0, 0.0]), np.array([0.6, 0.0]),
                           100000, 0.0, "bull_steady")
        ft.check_episode()
        assert ft.summary().get("trivial_safety", 0) > 0
