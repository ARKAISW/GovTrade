# GovTrade

**Learned Adaptive Governance in Multi-Agent Financial Systems**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)]()
[![PettingZoo](https://img.shields.io/badge/PettingZoo-AEC-purple)]()
[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> **Paper:** *GovTrade: Learned Adaptive Governance in Multi-Agent Financial Systems* — Arka Sarkar (2026)
> [[PDF]](paper/main.tex) · [[arXiv]](#) *(link added on upload)*

---

## What is this?

GovTrade is an open benchmark environment for studying **learned financial governance** — the question of whether neural agents can learn to impose *better* institutional controls than static rules under market stress.

The environment models oversight as a **three-agent Dec-POMDP**: a profit-maximizing Trader, a risk-constraining Risk Manager (RM), and a strategic Portfolio Manager (PM). The primary research object is not trading performance — it is **governance quality**: whether the supervisory agents learn to intervene appropriately, not too early, not too late, and without paralyzing the system they oversee.

The paper's central finding is a failure mode we call **Cooperative Strangulation**: a degenerate Nash-like equilibrium where both supervisory agents converge to a permanent shutdown policy, achieving zero constraint violations at the cost of complete portfolio inactivity. This is documented, characterized in relation to the specification gaming and Goodhart's Law literature, and partially mitigated via Institutional Action Floors.

---

## Key Results

Across **nine independent training seeds** (5,000 episodes each, PPO with Alternating Optimization):

| Metric | Learned (B4) | Random (B0) |
|--------|-------------|-------------|
| Constraint violation rate | **0.72%** | 56.4% |
| Reduction in violations vs. random | **78×** | — |
| Sharpe ratio | 0.143 | negative |
| Returns | positive | negative |
| Governance Score (mean ± std) | **0.7406 ± 0.0464** | 0.21 |

95% CI for Governance Score: **[0.7103, 0.7709]** (t-distribution); **[0.7099, 0.7709]** (non-parametric bootstrap, n=10,000), confirming robustness of the uncertainty estimate.

**Counterfactual evaluation** (50 deterministic episodes, identical trajectories):  
Governed system → **3.38% lower maximum drawdown** than ungoverned, on the same market trajectories.

---

## The Cooperative Strangulation Failure Mode

The most important finding is a MARL failure mode that is structurally distinct from canonical single-agent reward hacking:

| Property | Single-agent reward hacking | Cooperative Strangulation |
|---|---|---|
| Agents | 1 | 2 (must converge simultaneously) |
| Strategy type | Unintended shortcut | Globally optimal for the safety objective *as specified* |
| Missing constraint | Implicit | Operational viability is absent from the reward function |
| Escape gradient | Exists | **None** — RM sees 0 drawdown, PM sees 0 activity |
| Equilibrium type | Suboptimal local | Consistent with Nash equilibrium under specified rewards |

Once the system enters this state, gradient-based training provides no signal to escape: the RM observes zero drawdown (no reason to relax limits), the PM observes zero activity (no PnL to protect). This makes it structurally more stable than typical specification gaming failures.

**Mitigation:** Institutional Action Floors — a hard lower bound `ℓ_min = 0.20` on the RM's size limit and `c_min = 0.05` on the PM's capital allocation. This forces viability to be an observable constraint rather than an implicit hope.

---

## Architecture

```
Market Regime Engine (12 regimes: 8 standard + 4 adversarial, Markov transitions)
         │
         ▼
┌────────────────────────────────────────────────────┐
│ Risk Manager (RM)   → obs: (25,)  act: (3,)         │
│   ↓ size limit ℓ, block flag b, message m_RM        │
│ Portfolio Manager (PM) → obs: (28,)  act: (2,)      │
│   ↓ capital alloc c, veto v, message m_PM           │
│ Trader              → obs: (30,)  act: mixed         │
│   ↓ direction d, size s, SL, TP                     │
│ Portfolio / Environment                             │
│   ↓ reward Rⁱ (decoupled), next obs                │
└────────────────────────────────────────────────────┘
```

All agents use a shared **Actor-Critic MLP**: 3 hidden layers × 256 units, LayerNorm, GELU, orthogonal init. The Trader uses a mixed discrete/continuous output head. Each agent receives only its own reward signal — institutional independence is enforced by design.

---

## Governance Failure Taxonomy

GovTrade defines **9 formally-detectable, measurable failure modes**. These are research instruments — they do not influence the reward signal.

| ID | Failure Mode | Trigger | Severity |
|----|---|---|---|
| F1 | Overreaction | RM restricts >80% at <5% drawdown | Medium |
| F2 | Delayed Intervention | DD >20% for 3+ steps before RM responds | High |
| F3 | False Tightening | ℓ <10% in bull regime, DD <3% | Low |
| F4 | Capital Starvation | PM alloc <10% for >20 steps | High |
| F5 | Collapse Cascade | >30% portfolio drop in <10 steps | Critical |
| F6 | Override Thrashing | >5 PM override flips in 10 steps | Medium |
| F7 | Inaction Under Stress | ℓ >60% while DD >15% | High |
| F8 | Trivial Safety | Mean ℓ <0.05, std <0.02 (*Cooperative Strangulation*) | Medium |
| F9 | Bankruptcy | NAV <10% of initial capital | Critical |

---

## Benchmark Design

### Baselines (B0–B4)

| ID | Configuration | Purpose |
|----|---|---|
| B0 | Random policies (all agents) | Lower bound |
| B1 | All rule-based | Static governance baseline |
| B2 | Learned Trader + rule-based RM/PM | Single-agent RL |
| B3 | Learned RM + rule-based PM + learned Trader | Partial governance |
| B4 | All learned | Full learned governance (**main result**) |

### Ablation Studies (A1–A6)

| ID | Ablation | Isolates |
|----|---|---|
| A1 | Remove RM constraints entirely | Value of risk management |
| A2 | Static RM (fixed 50% limit) | Value of *adaptive* limits |
| A3 | Remove PM constraints | Value of portfolio oversight |
| A4 | Zero inter-agent messages | Value of agent communication |
| A5 | Zero regime indicator | Value of regime awareness |
| A6 | Adversarial Trader | Governance robustness under adversarial actor |

### Counterfactual Analysis

Same deterministic seed, two conditions:
- **World A** — Governed: learned RM + PM constrain the Trader
- **World B** — Ungoverned: same Trader policy, no governance layer

Reports: maximum drawdown difference, attributed to governance intervention.

---

## Reproducing the Results

### Installation

```bash
git clone https://github.com/ARKAISW/GovTrade.git
cd GovTrade
pip install -r requirements.txt
```

### Quick smoke test (~2 min)

```bash
python training/ppo_trainer.py --episodes 50 --max-steps 100 --device cpu
```

### Full training run (5,000 episodes)

```bash
# Local (CPU, ~6–12 hours)
python training/ppo_trainer.py --episodes 5000 --device auto --seed 7

# On Kaggle GPU (recommended — free T4, ~90 min per seed)
# See the Kaggle section below
```

### Benchmark evaluation (from checkpoint)

```bash
# Full benchmark: baselines + ablations + counterfactual + OOD
python -m evaluation.benchmark_suite \
    --checkpoint-dir outputs/ppo_training/best_checkpoint_ep* \
    --seeds 50

# Unit tests
python -m pytest tests/ -v
```

### Makefile shortcuts

```bash
make train          # 5000-episode PPO run
make train-quick    # 50-episode smoke test
make benchmark      # Full benchmark suite
make test           # All 49 unit tests
make test-governance # Governance-specific tests
make plots          # Reproduce publication figures
```

---

## Training on Kaggle (Free GPU)

The paper's results were produced on Kaggle T4 GPUs. A full 5,000-episode run takes ~90 minutes per seed.

```python
# In a Kaggle notebook:
!git clone https://github.com/ARKAISW/GovTrade.git
%cd GovTrade
!pip install -q pettingzoo>=1.24.0 gymnasium

!python training/ppo_trainer.py \
    --episodes 5000 \
    --device cuda \
    --output-dir /kaggle/working/govtrade_outputs \
    --seed 7
```

After training, save the `/kaggle/working/govtrade_outputs/` directory as a Kaggle Dataset for reuse.

---

## Training Methodology

**Joint training is explicitly not used.** It consistently produces a "Death Spiral": the RM explores by tightening limits to zero, severing the causal link between the Trader's actions and environment transitions. The Trader's advantage function collapses, causing it to stop acting. This is documented in Section 5.1 of the paper.

**Alternating Optimization** cycles through agents in 50-episode phases (Trader → RM → PM), providing each agent with a pseudo-stationary environment:

```
for episode in range(5000):
    phase = (episode // 50) % 3      # 0=Trader, 1=RM, 2=PM
    collect_episode(all_agents_act)
    ppo_update(only agents[phase])
    decay_entropy()
```

**Curriculum** increases difficulty probabilistically to prevent catastrophic forgetting of stable-market behaviors:

| Training progress | Easy | Medium | Hard |
|---|---|---|---|
| 0–30% | 85% | 10% | 5% |
| 30–60% | 50% | 35% | 15% |
| 60–85% | 20% | 40% | 40% |
| 85–100% | 10% | 20% | 70% |

---

## Project Structure

```
GovTrade/
├── agents/
│   └── networks.py              # Actor-Critic MLP (shared backbone for all agents)
├── env/
│   ├── multi_agent_env.py       # PettingZoo AEC environment
│   ├── regime_engine.py         # 12-regime Markov market engine
│   ├── governance_metrics.py    # 15+ governance quality metrics
│   ├── failure_taxonomy.py      # 9 formally-detectable failure modes
│   ├── reward.py                # Decoupled reward functions
│   └── state.py                 # MarketState / PortfolioState / RiskState
├── training/
│   └── ppo_trainer.py           # Alternating PPO + curriculum
├── evaluation/
│   ├── benchmark_suite.py       # Baselines + ablations + counterfactual + OOD
│   └── statistical_tests.py     # Bootstrap CI, Welch's t-test, Cohen's d
├── visualization/
│   └── publication_plots.py     # Matplotlib figures for paper
├── configs/
│   ├── env_default.yaml
│   ├── training_ppo.yaml        # Full PPO hyperparameter table
│   └── benchmark.yaml
├── paper/
│   ├── main.tex                 # LaTeX source
│   └── references.bib
├── tests/                       # 49 unit + integration tests
├── Makefile
└── requirements.txt
```

---

## PPO Hyperparameters

| Parameter | Value |
|---|---|
| Clip ε | 0.2 |
| GAE λ | 0.95 |
| Discount γ | 0.99 |
| Entropy coefficient (initial) | 0.01 |
| Entropy decay | 0.9998/episode |
| Learning rate | 3×10⁻⁴ |
| Batch size | full episode |
| Hidden layers | 3 × 256 |
| Activation | GELU + LayerNorm |
| Weight init | Orthogonal |
| Phase length | 50 episodes |
| Total episodes | 5,000 |

---

## Citation

If you use GovTrade in your research, please cite:

```bibtex
@article{sarkar2026govtrade,
  title   = {GovTrade: Learned Adaptive Governance in Multi-Agent Financial Systems},
  author  = {Sarkar, Arka},
  year    = {2026},
  note    = {arXiv preprint}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
