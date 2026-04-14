"""
Tests for scripts/make_splits.py — the train/val split fix.

These ensure the regression that made our original train.parquet identical to
test.parquet cannot silently return.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent


def _toy_parquet(path: Path, n_identities: int, images_per_id: list[int]):
    """Build a small labeled parquet where identity i has images_per_id[i] images."""
    assert len(images_per_id) == n_identities
    rows = []
    for pid in range(n_identities):
        for k in range(images_per_id[pid]):
            rows.append({
                "image_path": f"images/fake/id{pid:03d}_img{k:03d}.png",
                "split": "test",
                "identity": pid,
                "camera_id": 0,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _run_splits(source: Path, out_dir: Path, val_frac=0.2, seed=42):
    """Invoke make_splits.py as a subprocess and check it exits cleanly."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_splits.py"),
        "--source", str(source),
        "--out_dir", str(out_dir),
        "--val_frac", str(val_frac),
        "--seed", str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"make_splits.py failed:\n{result.stderr}"
    return result


class TestMakeSplits:
    def test_identity_sets_are_disjoint(self, tmp_path):
        # 20 identities, all with 3 images → all eligible for val
        source = tmp_path / "source.parquet"
        _toy_parquet(source, n_identities=20, images_per_id=[3] * 20)
        _run_splits(source, tmp_path, val_frac=0.25)

        tr = pd.read_parquet(tmp_path / "train.parquet")
        va = pd.read_parquet(tmp_path / "val.parquet")

        tr_ids = set(tr["identity"])
        va_ids = set(va["identity"])
        assert tr_ids.isdisjoint(va_ids), "identity overlap between train and val"

    def test_image_paths_are_disjoint(self, tmp_path):
        source = tmp_path / "source.parquet"
        _toy_parquet(source, n_identities=20, images_per_id=[3] * 20)
        _run_splits(source, tmp_path)

        tr = pd.read_parquet(tmp_path / "train.parquet")
        va = pd.read_parquet(tmp_path / "val.parquet")
        assert set(tr["image_path"]).isdisjoint(set(va["image_path"]))

    def test_singletons_go_to_train_only(self, tmp_path):
        """Identities with only 1 image can't participate in ReID eval
        (no query+gallery pair possible). They should land in train."""
        source = tmp_path / "source.parquet"
        images_per_id = [1] * 10 + [3] * 10  # 10 singletons, 10 with 3 images each
        _toy_parquet(source, n_identities=20, images_per_id=images_per_id)
        _run_splits(source, tmp_path, val_frac=0.5)

        va = pd.read_parquet(tmp_path / "val.parquet")
        singleton_ids = set(range(10))
        assert singleton_ids.isdisjoint(set(va["identity"])), (
            "singletons leaked into val — they should all be in train"
        )

    def test_split_is_deterministic_for_same_seed(self, tmp_path):
        source = tmp_path / "source.parquet"
        _toy_parquet(source, n_identities=50, images_per_id=[2] * 50)

        # First split
        out1 = tmp_path / "run1"
        out1.mkdir()
        _run_splits(source, out1, val_frac=0.2, seed=42)

        # Second split with same seed
        out2 = tmp_path / "run2"
        out2.mkdir()
        _run_splits(source, out2, val_frac=0.2, seed=42)

        v1 = pd.read_parquet(out1 / "val.parquet").sort_values("image_path").reset_index(drop=True)
        v2 = pd.read_parquet(out2 / "val.parquet").sort_values("image_path").reset_index(drop=True)
        pd.testing.assert_frame_equal(v1, v2)

    def test_val_frac_approximately_honored(self, tmp_path):
        """With 100 eligible identities and val_frac=0.2, val should have ≈20 IDs."""
        source = tmp_path / "source.parquet"
        _toy_parquet(source, n_identities=100, images_per_id=[2] * 100)
        _run_splits(source, tmp_path, val_frac=0.2)

        va = pd.read_parquet(tmp_path / "val.parquet")
        n_val_ids = va["identity"].nunique()
        assert 18 <= n_val_ids <= 22, f"expected ≈20 val ids, got {n_val_ids}"
