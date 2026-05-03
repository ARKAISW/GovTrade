"""
Benchmark Suite for QuantHive.

Provides rigorous, deterministic evaluation with:
  - 5 baseline configurations (B0-B4)
  - 6 ablation studies (A1-A6)
  - Counterfactual evaluation (World A vs World B)
  - Out-of-distribution testing
"""

from __future__ import annotations
import json, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ─── Wrapper Agents ────────────────────────────────────────────────────────────

class RandomRiskManager:
    def act(self, obs, deterministic=False):
        return np.random.uniform(0, 1, size=3).astype(np.float32)

class RandomPortfolioManager:
    def act(self, obs, deterministic=False):
        return np.random.uniform(0, 1, size=2).astype(np.float32)

class RandomTraderWrapper:
    def act(self, obs, deterministic=False):
        return {"direction": np.random.randint(0, 3),
                "size": np.array([np.random.uniform(0, 0.5)], dtype=np.float32),
                "sl": np.array([0.0], dtype=np.float32),
                "tp": np.array([0.0], dtype=np.float32)}

class RuleBasedRiskManagerWrapper:
    def act(self, obs, deterministic=False):
        dd = float(obs[22]) if len(obs) > 22 else 0.0
        if dd > 0.20: return np.array([0.1, 0.0, 1.0], dtype=np.float32)
        elif dd > 0.10: return np.array([0.3, 0.5, 0.0], dtype=np.float32)
        return np.array([0.8, 1.0, 0.0], dtype=np.float32)

class RuleBasedPortfolioManagerWrapper:
    def act(self, obs, deterministic=False):
        return np.array([0.6, 0.0], dtype=np.float32)

class RuleBasedTraderWrapper:
    def act(self, obs, deterministic=False):
        rsi = float(obs[5]) if len(obs) > 5 else 0.5
        if rsi < 0.3: d, s = 1, 0.3
        elif rsi > 0.7: d, s = 2, 0.3
        else: d, s = 0, 0.0
        return {"direction": d, "size": np.array([s], dtype=np.float32),
                "sl": np.array([0.0], dtype=np.float32),
                "tp": np.array([0.0], dtype=np.float32)}

class NoConstraintRM:
    def act(self, obs, deterministic=False):
        return np.array([1.0, 1.0, 0.0], dtype=np.float32)

class StaticConstraintRM:
    def act(self, obs, deterministic=False):
        return np.array([0.5, 1.0, 0.0], dtype=np.float32)

class NoConstraintPM:
    def act(self, obs, deterministic=False):
        return np.array([1.0, 0.0], dtype=np.float32)


def make_agent_set(rm_type="learned", pm_type="learned", trader_type="learned",
                   checkpoint_dir=None, device="cpu"):
    agents = {}
    if rm_type == "random": agents["rm"] = RandomRiskManager()
    elif rm_type == "rule_based": agents["rm"] = RuleBasedRiskManagerWrapper()
    elif rm_type == "learned":
        from agents.networks import LearnedRiskManager
        rm = LearnedRiskManager(obs_dim=25, device=device)
        if checkpoint_dir:
            import torch
            p = Path(checkpoint_dir) / "risk_manager.pt"
            if p.exists(): rm.load_state_dict(torch.load(p, map_location=device))
        agents["rm"] = rm

    if pm_type == "random": agents["pm"] = RandomPortfolioManager()
    elif pm_type == "rule_based": agents["pm"] = RuleBasedPortfolioManagerWrapper()
    elif pm_type == "learned":
        from agents.networks import LearnedPortfolioManager
        pm = LearnedPortfolioManager(obs_dim=28, device=device)
        if checkpoint_dir:
            import torch
            p = Path(checkpoint_dir) / "portfolio_manager.pt"
            if p.exists(): pm.load_state_dict(torch.load(p, map_location=device))
        agents["pm"] = pm

    if trader_type == "random": agents["trader"] = RandomTraderWrapper()
    elif trader_type == "rule_based": agents["trader"] = RuleBasedTraderWrapper()
    elif trader_type == "learned":
        from agents.networks import LearnedTrader
        trader = LearnedTrader(obs_dim=30, device=device)
        if checkpoint_dir:
            import torch
            p = Path(checkpoint_dir) / "trader.pt"
            if p.exists(): trader.load_state_dict(torch.load(p, map_location=device))
        agents["trader"] = trader
    return agents


# ─── Evaluation Runner ─────────────────────────────────────────────────────────

def evaluate_episode(env, agents, max_steps=500, zero_messages=False, zero_regime=False):
    from env.multi_agent_env import RISK_MANAGER, PORTFOLIO_MGR, TRADER
    from env.governance_metrics import GovernanceMetrics
    from env.failure_taxonomy import GovernanceFailureTaxonomy

    gov = GovernanceMetrics()
    ftax = GovernanceFailureTaxonomy()
    env.reset()
    gov.peak_value = env._initial_cash
    sc = 0
    while env.agents and sc < max_steps * 3:
        ag = env.agent_selection
        obs = env.observe(ag)
        if zero_messages and ag == TRADER: obs[-5:] = 0.0
        if zero_regime and len(obs) > 24: obs[24] = 0.0
        if ag == RISK_MANAGER: action = agents["rm"].act(obs, True)
        elif ag == PORTFOLIO_MGR: action = agents["pm"].act(obs, True)
        elif ag == TRADER: action = agents["trader"].act(obs, True)
        env.step(action)
        sc += 1
        if ag == TRADER:
            info = env.infos.get(TRADER, {})
            gv = info.get("governance", {})
            gov.update_step(
                info.get("portfolio_value", env._initial_cash),
                gv.get("was_compliant", True),
                len(gv.get("interventions", [])),
                env._rm_message, env._pm_message,
                getattr(env, "_current_regime_label", ""),
                info.get("max_drawdown", 0.0))
            ftax.check_step(env._current_step, env._rm_message, env._pm_message,
                           info.get("portfolio_value", env._initial_cash),
                           info.get("max_drawdown", 0.0),
                           getattr(env, "_current_regime_label", ""))
    ftax.check_episode()
    return gov.finalize(), ftax


# ─── Benchmark Suite ───────────────────────────────────────────────────────────

class BenchmarkSuite:
    BASELINES = {
        "B0": {"rm": "random", "pm": "random", "trader": "random"},
        "B1": {"rm": "rule_based", "pm": "rule_based", "trader": "rule_based"},
        "B2": {"rm": "rule_based", "pm": "rule_based", "trader": "learned"},
        "B3": {"rm": "learned", "pm": "rule_based", "trader": "learned"},
        "B4": {"rm": "learned", "pm": "learned", "trader": "learned"},
    }

    def __init__(self, num_seeds=50, max_steps=500, checkpoint_dir=None,
                 output_dir="results/benchmark", device="cpu"):
        self.num_seeds = num_seeds
        self.max_steps = max_steps
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def _run_eval_set(self, agents, difficulty="hard", zero_messages=False,
                      zero_regime=False, forced_regime=None):
        from env.multi_agent_env import MultiAgentTradingEnv
        all_m = defaultdict(list)
        for seed in range(10000, 10000 + self.num_seeds):
            env = MultiAgentTradingEnv(difficulty=difficulty, max_steps=self.max_steps,
                                       seed=seed, forced_regime=forced_regime)
            m, _ = evaluate_episode(env, agents, self.max_steps, zero_messages, zero_regime)
            for k, v in m.items():
                if not np.isinf(v): all_m[k].append(v)
        return {f"{k}_mean": float(np.mean(v)) for k, v in all_m.items()}

    def run_baselines(self):
        print("\n" + "=" * 60 + "\n  BASELINE EVALUATION\n" + "=" * 60)
        results = {}
        for bid, cfg in self.BASELINES.items():
            agents = make_agent_set(**cfg, checkpoint_dir=self.checkpoint_dir, device=self.device)
            results[bid] = self._run_eval_set(agents)
            dd = results[bid].get("max_drawdown_mean", 0)
            sr = results[bid].get("sharpe_ratio_mean", 0)
            print(f"  {bid}: DD={dd:.2%} Sharpe={sr:.3f}")
        self._save("baseline_results.json", results)
        return results

    def run_ablations(self):
        print("\n" + "=" * 60 + "\n  ABLATION STUDIES\n" + "=" * 60)
        results = {}
        full = make_agent_set("learned","learned","learned",self.checkpoint_dir,self.device)

        # A1-A6
        for aid, mod in [("A1", lambda a: a.update({"rm": NoConstraintRM()}) or a),
                         ("A2", lambda a: a.update({"rm": StaticConstraintRM()}) or a),
                         ("A3", lambda a: a.update({"pm": NoConstraintPM()}) or a)]:
            a = make_agent_set("learned","learned","learned",self.checkpoint_dir,self.device)
            mod(a)
            results[aid] = self._run_eval_set(a)
            print(f"  {aid}: DD={results[aid].get('max_drawdown_mean',0):.2%}")

        a = make_agent_set("learned","learned","learned",self.checkpoint_dir,self.device)
        results["A4"] = self._run_eval_set(a, zero_messages=True)
        results["A5"] = self._run_eval_set(a, zero_regime=True)
        a6 = make_agent_set("learned","learned","random",self.checkpoint_dir,self.device)
        results["A6"] = self._run_eval_set(a6)
        self._save("ablation_results.json", results)
        return results

    def run_counterfactual(self):
        from env.multi_agent_env import MultiAgentTradingEnv
        print("\n" + "=" * 60 + "\n  COUNTERFACTUAL ANALYSIS\n" + "=" * 60)
        gov_a = make_agent_set("learned","learned","learned",self.checkpoint_dir,self.device)
        ungov = make_agent_set("learned","learned","learned",self.checkpoint_dir,self.device)
        ungov["rm"], ungov["pm"] = NoConstraintRM(), NoConstraintPM()
        g, u = [], []
        for s in range(10000, 10000 + self.num_seeds):
            ea = MultiAgentTradingEnv(difficulty="hard", max_steps=self.max_steps, seed=s)
            eb = MultiAgentTradingEnv(difficulty="hard", max_steps=self.max_steps, seed=s)
            ma, _ = evaluate_episode(ea, gov_a, self.max_steps)
            mb, _ = evaluate_episode(eb, ungov, self.max_steps)
            g.append(ma); u.append(mb)
        dd_g = np.mean([m["max_drawdown"] for m in g])
        dd_u = np.mean([m["max_drawdown"] for m in u])
        print(f"  Governed DD: {dd_g:.2%}  Ungoverned DD: {dd_u:.2%}  Prevented: {dd_u-dd_g:.2%}")
        res = {"governed_dd": float(dd_g), "ungoverned_dd": float(dd_u)}
        self._save("counterfactual_results.json", res)
        return res

    def run_ood(self):
        print("\n" + "=" * 60 + "\n  OOD EVALUATION\n" + "=" * 60)
        agents = make_agent_set("learned","learned","learned",self.checkpoint_dir,self.device)
        results = {}
        for regime in ["bull_steady","bear_steady","flash_crash","bubble_pop","cascading_liquidation"]:
            results[regime] = self._run_eval_set(agents, forced_regime=regime)
            print(f"  {regime}: DD={results[regime].get('max_drawdown_mean',0):.2%}")
        self._save("ood_results.json", results)
        return results

    def run_full(self):
        self.run_baselines()
        self.run_ablations()
        self.run_counterfactual()
        self.run_ood()

    def _save(self, fn, data):
        with open(self.output_dir / fn, "w") as f:
            json.dump(data, f, indent=2, default=float)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--output", type=str, default="results/benchmark")
    args = p.parse_args()
    BenchmarkSuite(num_seeds=args.seeds, checkpoint_dir=args.checkpoint_dir,
                   output_dir=args.output).run_full()
