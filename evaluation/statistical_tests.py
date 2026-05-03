"""
Statistical significance testing for QuantHive benchmark results.

Implements:
  - Welch's t-test for baseline comparisons
  - Bootstrap confidence intervals (95%)
  - Cohen's d effect sizes
  - Bonferroni correction for multiple comparisons
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's t-test for unequal variance samples."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d effect size."""
    na, nb = np.array(a), np.array(b)
    if len(na) < 2 or len(nb) < 2:
        return 0.0
    pooled_std = np.sqrt(((len(na)-1)*np.var(na, ddof=1) + (len(nb)-1)*np.var(nb, ddof=1)) /
                         (len(na) + len(nb) - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(na) - np.mean(nb)) / pooled_std)


def bootstrap_ci(data: List[float], confidence: float = 0.95,
                 n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, lower, upper)."""
    rng = np.random.default_rng(seed)
    arr = np.array(data)
    means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True))
                      for _ in range(n_bootstrap)])
    alpha = (1 - confidence) / 2
    return float(np.mean(arr)), float(np.percentile(means, alpha*100)), float(np.percentile(means, (1-alpha)*100))


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def compare_configs(results_a: List[float], results_b: List[float],
                    label_a: str = "A", label_b: str = "B") -> Dict[str, float]:
    """Full statistical comparison between two configurations."""
    t_stat, p_val = welch_ttest(results_a, results_b)
    d = cohens_d(results_a, results_b)
    mean_a, lo_a, hi_a = bootstrap_ci(results_a)
    mean_b, lo_b, hi_b = bootstrap_ci(results_b)
    return {
        f"{label_a}_mean": mean_a, f"{label_a}_ci_lo": lo_a, f"{label_a}_ci_hi": hi_a,
        f"{label_b}_mean": mean_b, f"{label_b}_ci_lo": lo_b, f"{label_b}_ci_hi": hi_b,
        "t_statistic": t_stat, "p_value": p_val, "cohens_d": d,
        "significant_005": p_val < 0.05, "significant_001": p_val < 0.01,
    }
