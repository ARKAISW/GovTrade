# QuantHive: A Benchmark for Learned Adaptive Governance
# in Multi-Agent Financial Systems
#
# Makefile for reproducible research workflows

.PHONY: train eval benchmark plots test clean

# Train all agents via Multi-Agent PPO
train:
	python training/ppo_trainer.py --episodes 1500 --device auto --output-dir outputs/ppo_training

# Quick training (for testing pipeline)
train-quick:
	python training/ppo_trainer.py --episodes 50 --max-steps 100 --device cpu --output-dir outputs/quick

# Run full benchmark suite (baselines + ablations + counterfactual + OOD)
benchmark:
	python -m evaluation.benchmark_suite --seeds 50 --output results/benchmark

# Generate publication figures
plots:
	python -m visualization.publication_plots --input results/benchmark --output figures

# Run full test suite
test:
	python -m pytest tests/ -v --tb=short

# Run only new governance/regime tests
test-governance:
	python -m pytest tests/test_regime_engine.py tests/test_governance_logic.py tests/test_reward_determinism.py -v

# Clean outputs
clean:
	rm -rf outputs/ results/ figures/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
