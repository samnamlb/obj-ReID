"""Regression tests for evaluation integrity, data loading, and both training stages."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from datasets.loader import get_train_loader
from evaluate import compute_cmc_map, evaluate_dataset
from reid_data import split_metadata, validate_images
from losses import TripletLossHardMining


@pytest.fixture
def data(tmp_path):
    rows = []
    for identity in (10, 30):
        for view in range(3):
            name = f"{identity}_{view}.png"
            Image.new("RGB", (32, 32), (identity * 5, view * 40, 20)).save(tmp_path / name)
            rows.append(dict(image_path=name, identity=identity, camera_id=view))
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "train.parquet")
    frame.to_parquet(tmp_path / "test.parquet")
    return tmp_path


def test_pk_loader_maps_sparse_ids(data):
    loader, count = get_train_loader(data, pk_sampler=True, P=2, K=4)
    images, labels, _ = next(iter(loader))
    assert count == 2 and images.shape == (8, 3, 224, 224)
    assert labels.bincount().tolist() == [4, 4]
    with pytest.raises(ValueError, match="PK sampling"):
        get_train_loader(data, pk_sampler=True, P=3, K=2)


def test_missing_images_fail_instead_of_changing_indices(data):
    with pytest.raises(FileNotFoundError, match="indices must not change"):
        validate_images(data, ["missing.png"])


def test_self_match_excluded_and_ragged_camera_filters():
    cmc, ap, inp = compute_cmc_map(
        np.array([[0, 2, 1], [1, 0, 2]]), np.array([1, 2]), np.array([1, 2, 1]),
        np.array([0, 0]), np.array([0, 0, 1]), max_rank=5, exclude_same_camera=False,
        query_paths=["a", "b"], gallery_paths=["a", "b", "c"],
    )
    assert ap == 1 and inp == 1 and len(cmc) == 5
    # The second query has no eligible positive and is excluded, as in standard ReID.
    assert np.all(cmc == 1)


def test_bad_rankings_rejected(data):
    query, gallery = split_metadata(data, "dataset_a")
    path = data / "pred.csv"
    pd.DataFrame({"query_index": range(len(query)), "ranked_gallery_indices": ["0,0"] * len(query)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="full gallery permutation"):
        evaluate_dataset(path, data, "dataset_a")


def test_triplet_duplicate_gradients_finite():
    embeddings = torch.ones(4, 8, requires_grad=True)
    loss = TripletLossHardMining()(embeddings, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_full_gallery_ap_and_inp():
    # Correct matches at ranks 2 and 4: AP=(1/2+2/4)/2, INP=2/4.
    cmc, ap, inp = compute_cmc_map(
        np.array([[0, 1, 2, 3]]), np.array([7]), np.array([0, 7, 0, 7]),
        np.array([0]), np.array([1, 1, 1, 1]), max_rank=5,
    )
    np.testing.assert_array_equal(cmc, [0, 1, 1, 1, 1])
    assert ap == .5 and inp == .5


def test_camera_exclusion_and_short_curves():
    cmc, ap, inp = compute_cmc_map(
        np.array([[0, 1, 2], [2, 0, 1]]), np.array([1, 2]), np.array([1, 1, 2]),
        np.array([0, 0]), np.array([0, 1, 1]), max_rank=5,
    )
    assert ap == 1 and inp == 1 and len(cmc) == 5


def test_dataset_b_explicit_order(data):
    frame = pd.read_parquet(data / "test.parquet")
    frame["split"] = ["gallery", "query", "gallery"] * 2
    frame.to_parquet(data / "test.parquet")
    query, gallery = split_metadata(data, "dataset_b")
    assert query.image_path.tolist() == ["10_1.png", "30_1.png"]
    assert gallery.image_path.tolist() == ["10_0.png", "10_2.png", "30_0.png", "30_2.png"]


class CumulativeTransformer(nn.Module):
    def forward(self, x, attn_mask=None):
        return x.cumsum(dim=1)


class TinyCLIP(nn.Module):
    """Small differentiable substitute; tests wiring, not CLIP quality."""
    def __init__(self):
        super().__init__()
        self.context_length = 77
        self.visual = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512))
        self.token_embedding = nn.Embedding(49408, 8)
        self.positional_embedding = nn.Parameter(torch.zeros(77, 8))
        self.transformer = CumulativeTransformer()
        self.ln_final = nn.LayerNorm(8)
        self.text_projection = nn.Parameter(torch.randn(8, 512) * 0.02)


def test_two_stage_training_and_checkpoint_roundtrip(data, monkeypatch):
    import open_clip
    import train_stage1
    import train_stage2
    from models.model import StudentModel

    torch.set_num_threads(2)
    monkeypatch.setattr(open_clip, "create_model_and_transforms", lambda *a, **kw: (TinyCLIP(), None, None))
    common = dict(dataset_root=str(data), parquet_path=None, num_identities=None,
                  category="animal", num_tokens=4, epochs=1, checkpoint_dir=str(data),
                  debug=False, debug_samples=200, device="cpu", seed=42, num_workers=0)
    train_stage1.train_stage1(SimpleNamespace(**common, batch_size=4, lr=0.001))
    train_stage2.train_stage2(SimpleNamespace(**common, P=2, K=2, proj_dim=16,
        stage1_checkpoint=str(data / "stage1_prompts.pt"), lambda_id=1., lambda_tri=1.,
        lambda_i2t=.5, triplet_margin=.3, lr_encoder=1e-4, lr_head=1e-3, warmup_epochs=1))
    model = StudentModel(device="cpu", checkpoint=str(data / "stage2_last.pt"))
    encoded = model.encode(torch.randn(2, 3, 224, 224))
    assert encoded.shape == (2, 16) and torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded.norm(dim=1), torch.ones(2))
    with pytest.raises(ValueError, match="proj_dim"):
        StudentModel(device="cpu", checkpoint=str(data / "stage2_last.pt"), proj_dim=32)
