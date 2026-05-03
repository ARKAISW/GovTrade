# QuantHive

**A Benchmark for Learned Adaptive Governance in Multi-Agent Financial Systems**

[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Abstract

QuantHive is a research benchmark that studies whether **learned governance policies** (trained via PPO) can outperform static, rule-based oversight in multi-agent financial environments. The project formalizes the governance problem as a Dec-POMDP with three heterogeneous agents — a Risk Manager, a Portfolio Manager, and a Trader — who must coordinate under regime-varying market conditions.

The benchmark provides:
- **12 market regimes** (8 standard + 4 adversarial) with Markov transitions
- **5 baseline configurations** (B0–B4) ranging from random to fully learned
- **6 ablation studies** (A1–A6) isolating individual governance components
- **Counterfactual evaluation**: same-seed comparison of governed vs. ungoverned worlds
- **Out-of-distribution testing**: evaluating trained policies on unseen market regimes
- **A formal governance failure taxonomy** with 8 named failure modes

> **Research Question**: *Can neural governance agents learn adaptive institutional controls that demonstrably outperform static rules, as measured by drawdown prevention, intervention efficiency, and counterfactual loss avoidance?*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Market Regime Engine                          │
│   8 standard regimes + 4 adversarial │ Markov transitions      │
└────────────────────────┬────────────────────────────────────────┘
                         │ OHLCV + regime_label
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PettingZoo AEC Environment                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Risk Manager │───▶│ Portfolio Mgr│───▶│   Trader     │      │
│  │  obs: (25,)  │    │  obs: (28,)  │    │  obs: (30,)  │      │
│  │  act: (3,)   │    │  act: (2,)   │    │  act: mixed  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │ rm_message        │ pm_message        │              │
│         └───────────────────┴───────────────────┘              │
│                    Inter-Agent Messages                         │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Governance Evaluation Layer                        │
│  GovernanceMetrics │ FailureTaxonomy │ Counterfactual Tracker   │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Observation | Action | Objective |
|-------|------------|--------|-----------|
| **Risk Manager** | Market + regime indicator (25,) | `[size_limit, allow_new, force_reduce]` | Minimize tail risk while maintaining capital utilization |
| **Portfolio Manager** | Market + RM message (28,) | `[capital_allocation, override_strength]` | Maximize risk-adjusted portfolio growth |
| **Trader** | Market + RM + PM messages (30,) | `{direction, size, sl, tp}` | Maximize PnL within governance constraints |

### Market Regimes

| Category | Regimes | Calibration |
|----------|---------|-------------|
| **Standard** | `bull_steady`, `bull_volatile`, `bear_steady`, `crash`, `sideways_choppy`, `mean_revert`, `bubble_pop`, `flash_crash` | S&P 500, BTC, historical events |
| **Adversarial** | `spoofing`, `delayed_signal`, `correlated_selloff`, `cascading_liquidation` | Designed for OOD evaluation |

---

## Governance Failure Taxonomy

QuantHive defines **8 named governance failure modes** with formal detection criteria:

| Failure | Severity | Trigger |
|---------|----------|---------|
| `overreaction` | Medium | RM restricts > 80% during < 5% drawdown |
| `delayed_intervention` | High | Drawdown exceeds 20% before RM responds |
| `false_constraint_tightening` | Low | RM tightens during bull regime |
| `capital_starvation` | High | PM allocation < 10% for 20+ steps |
| `collapse_cascade` | Critical | Portfolio drops 30% in < 10 steps despite governance |
| `override_thrashing` | Medium | PM override flips > 5 times in 10 steps |
| `inaction_under_stress` | High | RM keeps loose limits during 15%+ drawdown |
| `trivial_safety` | Medium | RM always outputs near-zero limits |

---

## Evaluation Framework

### Baselines (B0–B4)

| ID | Configuration | Purpose |
|----|--------------|---------|
| B0 | Random policies (all agents) | Lower bound |
| B1 | All rule-based | Static governance baseline |
| B2 | Learned Trader + rule-based RM/PM | Single-agent RL baseline |
| B3 | Learned RM + rule-based PM + learned Trader | Partial governance |
| B4 | All learned | Full learned governance |

### Ablation Studies (A1–A6)

| ID | Ablation | Tests |
|----|----------|-------|
| A1 | Remove RM constraints | Value of risk management |
| A2 | Static RM (fixed 50% limit) | Value of adaptive limits |
| A3 | Remove PM constraints | Value of portfolio oversight |
| A4 | Zero inter-agent messages | Value of communication |
| A5 | Zero regime indicator | Value of regime awareness |
| A6 | Adversarial Trader | Governance robustness |

### Counterfactual Analysis

Same-seed evaluation comparing:
- **World A**: Governed (learned RM + PM + Trader)
- **World B**: Ungoverned (no constraints, same Trader)

Reports: drawdown prevented, tail risk contained, loss avoidance.

---

## Quick Start

### Installation

```bash
pip install torch gymnasium pettingzoo numpy pandas scipy matplotlib
```

### Training

```bash
# Full training (1500 episodes with curriculum)
python training/ppo_trainer.py --episodes 1500 --device auto

# Quick smoke test
python training/ppo_trainer.py --episodes 50 --max-steps 100 --device cpu
```

### Evaluation

```bash
# Run full benchmark suite
python -m evaluation.benchmark_suite --seeds 50 --checkpoint-dir outputs/ppo_training/best_checkpoint_ep*

# Run tests
python -m pytest tests/test_regime_engine.py tests/test_governance_logic.py tests/test_environment.py -v
```

---

## Project Structure

```
QuantHive/
├── agents/
│   ├── networks.py              # Actor-Critic MLP for all 3 agents
│   ├── risk_model.py            # Rule-based RM (legacy baseline)
│   └── portfolio_manager.py     # Rule-based PM (legacy baseline)
├── env/
│   ├── multi_agent_env.py       # PettingZoo AEC environment
│   ├── regime_engine.py         # Market regime engine (12 regimes)
│   ├── governance_metrics.py    # 15+ governance quality metrics
│   ├── failure_taxonomy.py      # 8 named failure modes
│   ├── reward.py                # Reward functions
│   ├── state.py                 # Market/Portfolio/Risk state
│   └── trading_env.py           # Single-agent Gymnasium wrapper
├── training/
│   └── ppo_trainer.py           # Multi-Agent PPO with curriculum
├── evaluation/
│   ├── benchmark_suite.py       # 5 baselines + 6 ablations + counterfactual + OOD
│   └── statistical_tests.py     # Welch's t-test, Cohen's d, bootstrap CI
├── visualization/
│   └── publication_plots.py     # Academic-quality figures
├── configs/
│   ├── env_default.yaml         # Environment parameters
│   ├── training_ppo.yaml        # PPO hyperparameters
│   └── benchmark.yaml           # Evaluation configuration
├── paper/
│   ├── main.tex                 # LaTeX paper skeleton
│   └── references.bib           # Bibliography
├── tests/
│   ├── test_regime_engine.py    # 11 tests for regime generation
│   ├── test_governance_logic.py # 11 tests for metrics + failure taxonomy
│   ├── test_reward_determinism.py # 9 tests for reward correctness
│   └── test_environment.py      # 18 tests for env integration
├── Makefile                     # Reproducible workflows
└── README.md                    # This file
```

---

## Key Metrics

QuantHive evaluates governance quality across three tiers:

**Capital Protection**: Max drawdown, tail-loss containment (VaR₉₅/VaR₉₉), Sortino ratio, Calmar ratio

**Governance Behavior**: Constraint violation rate, intervention efficiency, intervention latency, governance stability, false tightening rate, capital utilization

**Counterfactual Value**: Loss prevented by governance vs. no-governance world, regime recovery time

---

## Citation

```bibtex
@article{sarkar2026quanthive,
  title={QuantHive: A Benchmark for Learned Adaptive Governance in Multi-Agent Financial Systems},
  author={Sarkar, Arka},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
