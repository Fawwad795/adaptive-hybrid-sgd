"""
Tests for the checkpoint store (Phase 5).

All tests are single-process / in-memory — no ZeroMQ, no multiprocessing.

Run with:  pytest tests/test_checkpoint.py -v
"""

import numpy as np
import pytest
import tempfile
import time
from pathlib import Path

from checkpoint.store import save_checkpoint, load_checkpoint


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sample_params(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W": rng.standard_normal((784, 10)).astype(np.float32),
        "b": rng.standard_normal((10,)).astype(np.float32),
    }


def _sample_meta(version: int = 1) -> dict:
    return {
        "round":    version,
        "from":     "rar",
        "to":       "ps",
        "ts":       time.time(),
        "telemetry": {"lag_ratio": 1.8},
    }


# ── save / load round-trip ─────────────────────────────────────────────────────

class TestCheckpointRoundTrip:

    def test_params_restored_exactly(self, tmp_path):
        params = _sample_params(seed=42)
        save_checkpoint(params, _sample_meta(1), str(tmp_path), version=1)
        loaded_params, _ = load_checkpoint(str(tmp_path))
        for key in params:
            assert np.allclose(params[key], loaded_params[key], atol=1e-6), \
                f"Param '{key}' not restored correctly"

    def test_meta_restored(self, tmp_path):
        meta = _sample_meta(version=7)
        params = _sample_params()
        save_checkpoint(params, meta, str(tmp_path), version=7)
        _, loaded_meta = load_checkpoint(str(tmp_path))
        assert loaded_meta["round"] == 7
        assert loaded_meta["from"]  == "rar"
        assert loaded_meta["to"]    == "ps"

    def test_file_created(self, tmp_path):
        save_checkpoint(_sample_params(), _sample_meta(1), str(tmp_path), version=1)
        ckpt_files = list(tmp_path.glob("ckpt_v*.npz"))
        assert len(ckpt_files) == 1

    def test_no_tmp_file_left_behind(self, tmp_path):
        """Atomic write must not leave a .tmp file after success."""
        save_checkpoint(_sample_params(), _sample_meta(1), str(tmp_path), version=1)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Orphaned .tmp files: {tmp_files}"

    def test_multiple_versions_saved(self, tmp_path):
        for v in [1, 2, 3]:
            save_checkpoint(_sample_params(seed=v), _sample_meta(v),
                            str(tmp_path), version=v)
        ckpt_files = list(tmp_path.glob("ckpt_v*.npz"))
        assert len(ckpt_files) == 3

    def test_load_latest_returns_highest_version(self, tmp_path):
        for v in [1, 2, 5]:
            save_checkpoint(_sample_params(seed=v), _sample_meta(v),
                            str(tmp_path), version=v)
        _, meta = load_checkpoint(str(tmp_path))
        assert meta["round"] == 5

    def test_load_specific_version(self, tmp_path):
        for v in [1, 2, 3]:
            save_checkpoint(_sample_params(seed=v), _sample_meta(v),
                            str(tmp_path), version=v)
        _, meta = load_checkpoint(str(tmp_path), version=2)
        assert meta["round"] == 2

    def test_different_seeds_produce_different_params(self, tmp_path):
        p1 = _sample_params(seed=0)
        p2 = _sample_params(seed=1)
        assert not np.allclose(p1["W"], p2["W"]), \
            "Different seeds should produce different params"

    def test_large_params_round_trip(self, tmp_path):
        """CNN-sized params (~62K floats) must survive round-trip."""
        rng = np.random.default_rng(99)
        params = {
            "conv1.weight": rng.standard_normal((32, 3, 3, 3)).astype(np.float32),
            "conv2.weight": rng.standard_normal((64, 32, 3, 3)).astype(np.float32),
            "fc.weight":    rng.standard_normal((10, 64 * 8 * 8)).astype(np.float32),
            "fc.bias":      rng.standard_normal((10,)).astype(np.float32),
        }
        save_checkpoint(params, _sample_meta(1), str(tmp_path), version=1)
        loaded_params, _ = load_checkpoint(str(tmp_path))
        for key in params:
            assert np.allclose(params[key], loaded_params[key], atol=1e-6)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_checkpoint(str(tmp_path))   # empty dir → no checkpoints

    def test_checkpoint_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "ckpts"
        save_checkpoint(_sample_params(), _sample_meta(1), str(new_dir), version=1)
        assert new_dir.exists()
        assert list(new_dir.glob("ckpt_v*.npz"))

    def test_overwrite_same_version(self, tmp_path):
        """Saving twice with the same version must not leave duplicates."""
        params_v1 = _sample_params(seed=1)
        params_v2 = _sample_params(seed=2)
        save_checkpoint(params_v1, _sample_meta(1), str(tmp_path), version=1)
        save_checkpoint(params_v2, _sample_meta(1), str(tmp_path), version=1)
        ckpt_files = list(tmp_path.glob("ckpt_v*.npz"))
        assert len(ckpt_files) == 1
        loaded_params, _ = load_checkpoint(str(tmp_path), version=1)
        # Should be the second write
        assert np.allclose(params_v2["W"], loaded_params["W"], atol=1e-6)
