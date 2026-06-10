# GovTrade

**A Benchmark for Learned Adaptive Governance in Multi-Agent Financial Systems**

[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)]()
[![PettingZoo](https://img.shields.io/badge/PettingZoo-AEC-purple)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> *Can neural agents learn to be better institutional overseers than static rules — and can we prove it rigorously?*

---

## Overview

GovTrade is a research-grade benchmark for evaluating **learned adaptive governance** in multi-agent financial systems. It formalizes financial oversight as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** with three heterogeneous agents — a Risk Manager, a Portfolio Manager, and a Trader — who must coordinate under regime-varying market conditions.

The primary research question is not "how much profit did the system make?" but rather: **does neural governance demonstrably outperform static, rule-based oversight at preventing capital loss, containing tail risk, and responding correctly under market stress?**

### What the benchmark provides

| Component | Details |
|---|---|
| **Market Regimes** | 12 regimes (8 standard + 4 adversarial) with Markov transitions, calibrated to real events |
| **Baselines** | B0–B4: random → rule-based → partial learned → fully learned |
| **Ablation Studies** | A1–A6: isolating the value of each governance component |
| **Counterfactual Eval** | Same-seed comparison of governed vs. ungoverned worlds |
| **OOD Evaluation** | Testing on adversarial regimes never seen during training |
| **Failure Taxonomy** | 9 formally-defined, measurable governance failure modes |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Market Regime Engine                          │
│   8 standard regimes + 4 adversarial │ Markov transitions      │
└────────────────────────┬────────────────────────────────────────┘
                         │ OHLCV + noised regime indicator
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

Each agent acts sequentially per market step. An agent's output message becomes part of the next agent's observation, creating an emergent negotiation dynamic.

| Agent | Obs Shape | Action Space | Objective |
|-------|-----------|--------------|-----------|
| **Risk Manager** | `(25,)` | `[size_limit, allow_new, force_reduce]` | Minimize tail risk while maintaining capital utilization |
| **Portfolio Manager** | `(28,)` | `[capital_allocation, override_strength]` | Maximize risk-adjusted portfolio growth |
| **Trader** | `(30,)` | `{direction, size, sl, tp}` | Maximize PnL within active governance constraints |

All three agents share the same **Actor-Critic MLP** backbone: 3 hidden layers × 256 units, LayerNorm + GELU activation, trained via PPO. The Trader uses a specialized mixed discrete/continuous head (Categorical direction + Gaussian size/SL/TP).

### Market Regimes

| Category | Regimes | Real-World Calibration |
|----------|---------|------------------------|
| **Standard** | `bull_steady`, `bull_volatile`, `bear_steady`, `crash`, `sideways_choppy`, `mean_revert`, `bubble_pop`, `flash_crash` | S&P 500 2017, COVID crash Mar 2020, BTC Nov 2021, Flash Crash May 2010 |
| **Adversarial** | `spoofing`, `delayed_signal`, `correlated_selloff`, `cascading_liquidation` | Designed for OOD generalization testing |

Regimes follow a **Markov transition matrix** with configurable persistence. The regime indicator in the observation is intentionally noised (σ=0.15) so agents must infer regime from market features, not just the label.

---

## Training: Alternating PPO with Curriculum

> ⚠️ **Joint training is explicitly forbidden.** It causes a "Death Spiral" where the Trader stops acting and the Risk Manager imposes 0% limits. Agents are trained using **alternating optimization** — one agent updates per phase while the others act as frozen, stable partners.

### Algorithm

```
for episode in range(1500):
    phase = (episode // 50) % 3          # Cycle: Trader → RM → PM
    collect full episode (all 3 agents act)
    PPO update only the agent for current phase
    anneal entropy coefficient
```

### Curriculum

Training difficulty increases probabilistically to prevent catastrophic forgetting:

| Progress | Easy | Medium | Hard |
|----------|------|--------|------|
| 0–30% | 85% | 10% | 5% |
| 30–60% | 50% | 35% | 15% |
| 60–85% | 20% | 40% | 40% |
| 85–100% | 10% | 20% | 70% |

---

## Governance Failure Taxonomy

GovTrade defines **9 formally-detectable governance failure modes.** These are research logging instruments — they do **not** influence the reward signal.

| Failure | Severity | Trigger Condition |
|---------|----------|-------------------|
| `overreaction` | 🟡 Medium | RM restricts >80% during <5% drawdown |
| `delayed_intervention` | 🔴 High | Drawdown exceeds 20% for 3+ steps before RM responds |
| `false_constraint_tightening` | 🟢 Low | RM tightens during `bull_steady` with <3% drawdown |
| `capital_starvation` | 🔴 High | PM allocation <10% for 20+ consecutive steps |
| `collapse_cascade` | ⛔ Critical | Portfolio drops >30% in <10 steps despite active governance |
| `override_thrashing` | 🟡 Medium | PM override flips >5 times in 10 steps |
| `inaction_under_stress` | 🔴 High | RM keeps loose limits during 15%+ drawdown |
| `trivial_safety` | 🟡 Medium | RM consistently outputs near-zero limits (mean <0.05, std <0.02) |
| `bankruptcy` | ⛔ Critical | Portfolio falls below 10% of initial capital |

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

| ID | Ablation | What It Isolates |
|----|----------|-----------------|
| A1 | Remove RM constraints | Value of risk management |
| A2 | Static RM (fixed 50% limit) | Value of adaptive limits |
| A3 | Remove PM constraints | Value of portfolio oversight |
| A4 | Zero inter-agent messages | Value of agent communication |
| A5 | Zero regime indicator | Value of regime awareness |
| A6 | Adversarial Trader | Governance robustness under adversarial actors |

### Counterfactual Analysis

Same random seed, two worlds:
- **World A** — Governed: learned RM + PM constrain the Trader
- **World B** — Ungoverned: same Trader, no governance layer

Reports: drawdown prevented, VaR contained, total loss avoidance attributed to governance.

### Key Metrics

| Tier | Metrics |
|------|---------|
| **Capital Protection** | Max drawdown, VaR₉₅/VaR₉₉, Sortino ratio, Calmar ratio |
| **Governance Behavior** | Constraint violation rate, intervention efficiency, intervention latency, false tightening rate, capital utilization, governance stability |
| **Counterfactual Value** | Loss prevented vs. ungoverned, regime recovery time |

---

## Quick Start

### Local Installation

```bash
git clone https://github.com/ARKAISW/govtrade.git
cd govtrade
pip install -r requirements.txt
```

### Training (Local)

```bash
# Full 1500-episode run (CPU, ~2–4 hours)
python training/ppo_trainer.py --episodes 1500 --device auto

# Quick smoke test (verify everything works, ~2 minutes)
python training/ppo_trainer.py --episodes 50 --max-steps 100 --device cpu

# Resume from checkpoint
python training/ppo_trainer.py --episodes 1500 --resume-from outputs/ppo_training/best_checkpoint_ep500
```

### Evaluation

```bash
# Full benchmark suite (baselines + ablations + counterfactual + OOD)
python -m evaluation.benchmark_suite --seeds 50 --checkpoint-dir outputs/ppo_training/best_checkpoint_ep*

# Run unit tests
python -m pytest tests/ -v

# Governance-specific tests only
python -m pytest tests/test_governance_logic.py tests/test_regime_engine.py -v
```

---

## ☁️ Training on Kaggle (Free GPU)

You can train on a free Kaggle notebook GPU (Tesla P100 / T4) without keeping your machine on. A full 1500-episode run takes roughly **45–90 minutes** on a P100.

### Step-by-step

**1. Create a new Kaggle Notebook**
- Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
- Set **Accelerator → GPU T4 x2** (or P100) in the right-hand panel
- Set **Persistence → Files** so outputs survive session end

**2. Clone the repo and install dependencies**

```python
# Cell 1 — Setup
!git clone https://github.com/ARKAISW/govtrade.git
%cd govtrade
!pip install -q pettingzoo>=1.24.0 gymnasium
```

**3. Run training**

```python
# Cell 2 — Train
!python training/ppo_trainer.py \
    --episodes 1500 \
    --device cuda \
    --output-dir /kaggle/working/govtrade_outputs \
    --seed 42
```

**4. Save checkpoints as a Kaggle Dataset**
- After training, go to **Data → Upload** in your notebook
- Point it at `/kaggle/working/govtrade_outputs/`
- This lets you reuse checkpoints in future sessions without re-training

**5. Download results**
- Click the output folder in the right panel → Download as zip
- Or use the Kaggle API: `kaggle kernels output <username>/<notebook-slug> -p ./outputs`

> **Tip:** Kaggle sessions time out after ~12 hours but the notebook keeps running even if you close the browser tab. For 1500 episodes you're well within the limit.

---

## Project Structure

```
govtrade/
├── agents/
│   └── networks.py              # Actor-Critic MLP for all 3 agents (RM, PM, Trader)
├── env/
│   ├── multi_agent_env.py       # PettingZoo AEC environment (~990 lines)
│   ├── regime_engine.py         # Market regime engine (12 regimes, Markov transitions)
│   ├── governance_metrics.py    # 15+ governance quality metrics
│   ├── failure_taxonomy.py      # 9 named, formally-detectable failure modes
│   ├── reward.py                # Reward functions + GRPO verifier functions
│   └── state.py                 # MarketState / PortfolioState / RiskState
├── training/
│   └── ppo_trainer.py           # Multi-Agent PPO with alternating optimization + curriculum
├── evaluation/
│   ├── benchmark_suite.py       # 5 baselines + 6 ablations + counterfactual + OOD
│   └── statistical_tests.py     # Welch's t-test, Cohen's d, bootstrap CI
├── visualization/
│   └── publication_plots.py     # Academic-quality matplotlib figures
├── configs/
│   ├── env_default.yaml         # Environment parameters
│   ├── training_ppo.yaml        # PPO hyperparameters
│   └── benchmark.yaml           # Evaluation configuration
├── paper/
│   ├── main.tex                 # LaTeX paper skeleton
│   └── references.bib           # Bibliography
├── tests/
│   ├── test_environment.py      # 18 integration tests
│   ├── test_governance_logic.py # 11 tests for metrics + failure taxonomy
│   ├── test_regime_engine.py    # 11 tests for regime generation
│   ├── test_reward_determinism.py # 9 tests for reward correctness
│   └── smoke_test.py            # End-to-end sanity check
├── Makefile                     # Reproducible workflow commands
└── README.md                    # This file
```

### Makefile Targets

```bash
make train          # Full 1500-episode PPO run
make train-quick    # 50-episode smoke test
make benchmark      # Full benchmark suite
make test           # All unit tests
make test-governance # Governance-specific tests only
make plots          # Generate publication figures
```

---

## Anti-Reward-Hacking Design

The environment actively combats common reward-hacking failure modes:

| Hack | Fix |
|------|-----|
| Wash-trading (fake micro-trades to avoid inactivity penalty) | Fixed ticket fee ($5/trade) + min trade size 1% |
| Cooperative strangulation (RM+PM freeze Trader, avoid penalty) | Inactivity penalty shared across all three agents |
| RM rewarded for crises (perverse incentive to let Trader crash) | RM gets only 5% of upside but 30% of downside (asymmetric) |
| Lookahead bias in directional bonus | Lagged price trend used at decision time |
| Phantom policy updates from saturated sigmoid | Full Jacobian correction for change-of-variables |
| Over-restriction strangulation (vol→0 → penalty→0) | Constant minimum floor on all governance penalties |

---

## Citation

```bibtex
@article{sarkar2026govtrade,
  title={GovTrade: A Benchmark for Learned Adaptive Governance in Multi-Agent Financial Systems},
  author={Sarkar, Arka},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
