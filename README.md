# Adaptive Hybrid Distributed SGD

**NUST CS-347 Parallel & Distributed Computing — Course Project**  
Muhammad Jahanzeb Babar · Syed Fawwad Ahmed · Muhammad Shaheer Saleh

An adaptive distributed SGD framework that dynamically switches between
**Parameter Server (PS)** and **Ring AllReduce (RAR)** topologies at runtime
based on observed system conditions (bandwidth, straggler lag, node liveness).

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Run Phase 1 — single-worker LogReg on MNIST
python run.py --mode single --model logreg --dataset mnist --epochs 10 --seed 42

# 3. Run Phase 1 — single-worker CNN on CIFAR-10
python run.py --mode single --model cnn --dataset cifar10 --epochs 10 --seed 42
```

Or use the Makefile (requires `make`):

```bash
make setup        # install deps + download MNIST
make phase1       # LogReg on MNIST
make phase1-cnn   # CNN on CIFAR-10
make test-phase1  # run Phase 1 unit tests
```

## Judge Demo App

The repository now includes a modern demo UI for live judge-facing runs.

Backend:

```bash
python -m uvicorn demo_api.app:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` and use the **Hybrid Straggler Demo** preset for the
best live presentation flow. The frontend connects to the FastAPI backend, streams
epoch/iteration events live, and surfaces controller switch decisions plus
checkpoint artifacts.

---

## Directory Structure

```
adaptive-hybrid-sgd/
├── workers/          # Local SGD kernel, models, data loading, logger
├── ps_engine/        # Parameter Server (ZeroMQ) — Phase 2
├── rar_engine/       # Ring AllReduce (mpi4py) — Phase 3
├── controller/       # Adaptive Controller + Policy Engine — Phase 5
├── checkpoint/       # Atomic versioned checkpoints — Phase 4+
├── monitor/          # Heartbeat + telemetry collector — Phase 4
├── experiments/      # E1–E6 experiment drivers — Phase 6
├── analysis/         # Plot + report generators
├── results/          # raw logs, tables, plots, report
├── tests/            # pytest test suite
├── config/           # default.yaml (all hyperparams + thresholds)
├── run.py            # Unified CLI entry point
└── Makefile          # setup / phase1 / run-all / plots / test / clean
```

---

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | **DONE** | Single-worker SGD baseline (LogReg + CNN, MNIST + CIFAR-10) |
| 2 | Pending | Static Parameter Server (ZeroMQ, BSP/SSP/Async) |
| 3 | Pending | Static Ring AllReduce (mpi4py, BSP) |
| 4 | Pending | Monitoring layer (per-round telemetry) |
| 5 | Pending | Adaptive Controller (threshold rules + safe switching) |
| 6 | Pending | Full experimental campaign (E1–E6) |

---

## Phase 1 Results

**LogReg on MNIST (10 epochs, seed=42):**

| Epoch | Train Loss | Val Acc |
|-------|-----------|---------|
| 1     | 0.5063    | 90.2%   |
| 5     | 0.2977    | 91.9%   |
| 10    | 0.2785    | 92.0%   |

Throughput: ~1500–1700 samples/sec (single CPU core).  
Log: `results/raw/single_logreg_mnist_seed42.jsonl`  
Plots: `results/plots/loss_single_logreg_mnist_seed42.png`

---

## Reproducibility

All experiments use deterministic seeding:

```bash
python run.py --mode single --seed 42   # always produces the same loss curve
python run.py --mode single --seed 123  # different seed = different curve
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Parallelism | `multiprocessing`, `threading`, `mpi4py` |
| PS messaging | ZeroMQ (push/pull, req/reply) |
| RAR messaging | MPI collective ops (reduce-scatter + allgather) |
| Numerical backend | NumPy + PyTorch (CPU) |
| Metrics | JSON-lines logs + Matplotlib |

---

## Running All Experiments (Phase 6)

```bash
make run-all    # runs E1–E6 (requires Phases 1–5 complete)
make plots      # regenerate all figures
make report     # generate EVALUATION_REPORT.md
```
