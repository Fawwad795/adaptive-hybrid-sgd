"""
Phase 1 correctness tests.

Run with:  pytest tests/test_trainer.py -v
"""

import numpy as np
import pytest
import torch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_config(model="logreg", dataset="mnist", epochs=3, seed=42):
    return {
        "model": model,
        "dataset": dataset,
        "lr": 0.01,
        "batch_size": 64,
        "epochs": epochs,
        "seed": seed,
        "data_dir": "data",
        "log_dir": "results/raw",
    }


# ── Model unit tests ───────────────────────────────────────────────────────────

class TestLogReg:
    def setup_method(self):
        from workers.models import LogReg
        self.model = LogReg(seed=0)

    def test_forward_shape(self):
        X = np.random.randn(16, 784).astype(np.float32)
        logits = self.model.forward(X)
        assert logits.shape == (16, 10), f"Expected (16,10), got {logits.shape}"

    def test_gradient_shapes(self):
        X = np.random.randn(16, 784).astype(np.float32)
        y = np.random.randint(0, 10, 16)
        grads = self.model.gradients(X, y)
        assert grads["W"].shape == (784, 10)
        assert grads["b"].shape == (10,)

    def test_loss_decreases_after_update(self):
        X = np.random.randn(64, 784).astype(np.float32)
        y = np.random.randint(0, 10, 64)
        loss_before = self.model.loss(X, y)
        grads = self.model.gradients(X, y)
        self.model.update(grads, lr=0.1)
        loss_after = self.model.loss(X, y)
        assert loss_after < loss_before, "Loss should decrease after one gradient step"

    def test_get_set_params_roundtrip(self):
        params = self.model.get_params()
        W_orig = params["W"].copy()
        # Corrupt params
        self.model.W[:] = 999.0
        # Restore
        self.model.set_params(params)
        assert np.allclose(self.model.W, W_orig)

    def test_predict_proba_sums_to_one(self):
        X = np.random.randn(8, 784).astype(np.float32)
        probs = self.model.predict_proba(X)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)


class TestSmallCNN:
    def setup_method(self):
        from workers.models import SmallCNN
        torch.manual_seed(0)
        self.model = SmallCNN()

    def test_forward_shape(self):
        X = torch.randn(8, 3, 32, 32)
        logits = self.model(X)
        assert logits.shape == (8, 10)

    def test_param_count(self):
        count = self.model.param_count()
        # LeNet on CIFAR-10 should be roughly 60K params
        assert 50_000 < count < 80_000, f"Unexpected param count: {count}"

    def test_gradient_shapes_after_backward(self):
        X = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model(X)
        loss = criterion(logits, y)
        loss.backward()
        grads = self.model.gradients()
        for name, g in grads.items():
            param_shape = dict(self.model.named_parameters())[name].shape
            assert g.shape == tuple(param_shape), \
                f"Grad shape mismatch for {name}: {g.shape} vs {param_shape}"

    def test_get_set_params_roundtrip(self):
        params = self.model.get_params()
        # Corrupt params
        with torch.no_grad():
            for p in self.model.parameters():
                p.fill_(999.0)
        # Restore
        self.model.set_params(params)
        restored = self.model.get_params()
        for name in params:
            assert np.allclose(params[name], restored[name]), \
                f"Param {name} not restored correctly"


# ── Data loader tests ──────────────────────────────────────────────────────────

class TestDataLoader:
    def test_shard_deterministic(self):
        """Same (rank, world_size, seed) must produce the same shard every time."""
        from workers.data_loader import get_shard
        cfg = _base_config()
        loader1 = get_shard(0, 2, cfg)
        loader2 = get_shard(0, 2, cfg)
        # Compare first-batch indices by loading two batches
        batch1 = next(iter(loader1))[1].tolist()
        batch2 = next(iter(loader2))[1].tolist()
        assert batch1 == batch2, "Shard must be deterministic"

    def test_shards_cover_dataset(self):
        """All shard indices combined must cover the entire dataset exactly once."""
        from workers.data_loader import _shard_indices
        total, world_size, seed = 1000, 4, 7
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(_shard_indices(total, rank, world_size, seed))
        assert sorted(all_indices) == list(range(total)), \
            "Shards must partition the full index range"

    def test_batch_shape_mnist(self):
        from workers.data_loader import get_shard
        cfg = _base_config()
        loader = get_shard(0, 1, cfg)
        Xb, yb = next(iter(loader))
        assert Xb.shape[1:] == (1, 28, 28)
        assert yb.shape[0] == Xb.shape[0]

    def test_batch_shape_cifar10(self):
        from workers.data_loader import get_shard
        cfg = _base_config(model="cnn", dataset="cifar10")
        loader = get_shard(0, 1, cfg)
        Xb, yb = next(iter(loader))
        assert Xb.shape[1:] == (3, 32, 32)


# ── Trainer tests ──────────────────────────────────────────────────────────────

class TestTrainer:
    def test_reproducibility(self):
        """Two runs with the same seed must produce identical loss sequences."""
        from workers.trainer import train_single
        cfg = _base_config(epochs=1)
        m1 = train_single(cfg, run_id="_test_repro_run1")
        m2 = train_single(cfg, run_id="_test_repro_run2")

        losses1 = [r["loss"] for r in m1 if r.get("phase") == "train"]
        losses2 = [r["loss"] for r in m2 if r.get("phase") == "train"]
        assert losses1 == losses2, "Training must be deterministic given the same seed"

    def test_loss_decreases(self):
        """Train loss at epoch 3 must be lower than epoch 1."""
        from workers.trainer import train_single
        cfg = _base_config(epochs=3)
        metrics = train_single(cfg, run_id="_test_loss_dec")
        summaries = [r for r in metrics if r.get("phase") == "epoch_summary"]
        assert len(summaries) >= 2
        assert summaries[-1]["train_loss"] < summaries[0]["train_loss"], \
            "Train loss should decrease over epochs"

    def test_logreg_mnist_convergence(self):
        """LogReg should reach >85% val accuracy on MNIST within 5 epochs."""
        from workers.trainer import train_single
        cfg = _base_config(epochs=5)
        metrics = train_single(cfg, run_id="_test_convergence")
        summaries = [r for r in metrics if r.get("phase") == "epoch_summary"]
        final_val_acc = summaries[-1]["val_acc"]
        assert final_val_acc > 0.85, \
            f"Expected val acc > 85%, got {final_val_acc*100:.1f}%"

    def test_metrics_all_present(self):
        """Each iteration row must have the required keys."""
        from workers.trainer import train_single
        cfg = _base_config(epochs=1)
        metrics = train_single(cfg, run_id="_test_keys")
        required = {"phase", "epoch", "iter", "loss", "acc", "compute_ms"}
        for row in metrics:
            if row.get("phase") == "train":
                missing = required - set(row.keys())
                assert not missing, f"Missing keys in metrics row: {missing}"

    def test_log_file_created(self):
        """A .jsonl log file must be written after training."""
        from workers.trainer import train_single
        from pathlib import Path
        cfg = _base_config(epochs=1)
        run_id = "_test_log_file"
        train_single(cfg, run_id=run_id)
        log_path = Path(cfg["log_dir"]) / f"{run_id}.jsonl"
        assert log_path.exists(), f"Log file not found: {log_path}"
        assert log_path.stat().st_size > 0, "Log file is empty"
