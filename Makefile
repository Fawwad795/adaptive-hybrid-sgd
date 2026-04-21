PYTHON   ?= python
PIP      ?= pip
PYTEST   ?= pytest

.PHONY: setup phase1 phase1-cnn test-phase1 test run-all plots clean help

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "import torchvision; torchvision.datasets.MNIST('data', download=True); print('MNIST ready')"
	@echo "Setup complete."

# ── Phase 1 ───────────────────────────────────────────────────────────────────
phase1:
	$(PYTHON) run.py --mode single --model logreg --dataset mnist --epochs 10 --seed 42

phase1-cnn:
	$(PYTHON) run.py --mode single --model cnn --dataset cifar10 --epochs 10 --seed 42

phase1-both: phase1 phase1-cnn

# ── Tests ─────────────────────────────────────────────────────────────────────
test-phase1:
	$(PYTEST) tests/test_trainer.py -v

test:
	$(PYTEST) tests/ -v

# ── Experiments (Phase 6) ─────────────────────────────────────────────────────
run-e1:
	$(PYTHON) run.py --exp e1

run-e2:
	$(PYTHON) run.py --exp e2

run-e3:
	$(PYTHON) run.py --exp e3

run-e4:
	$(PYTHON) run.py --exp e4

run-e5:
	$(PYTHON) run.py --exp e5

run-e6:
	$(PYTHON) run.py --exp e6

run-all: run-e1 run-e2 run-e3 run-e4 run-e5 run-e6

# ── Plots ─────────────────────────────────────────────────────────────────────
plots:
	$(PYTHON) analysis/plot_loss_curves.py --log results/raw/*.jsonl
	$(PYTHON) analysis/plot_speedup.py
	$(PYTHON) analysis/plot_straggler.py
	$(PYTHON) analysis/plot_recovery.py
	$(PYTHON) analysis/plot_bandwidth.py
	$(PYTHON) analysis/plot_switching.py

report:
	$(PYTHON) analysis/report_generator.py

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf results/raw/* results/tables/* results/plots/* results/report/* checkpoints/*
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make setup        Install dependencies + download MNIST"
	@echo "  make phase1       Run single-worker LogReg on MNIST"
	@echo "  make phase1-cnn   Run single-worker CNN on CIFAR-10"
	@echo "  make test-phase1  Run Phase 1 unit tests"
	@echo "  make test         Run all tests"
	@echo "  make run-all      Run all 6 experiments (Phases 1–5 must be complete)"
	@echo "  make plots        Regenerate all figures from existing logs"
	@echo "  make report       Generate EVALUATION_REPORT.md"
	@echo "  make clean        Remove all generated files"
	@echo ""
