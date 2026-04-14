"""
Unit tests for losses/__init__.py.

These verify that each loss computes the *mathematically expected* value on
inputs where we can work out the answer by hand. They are independent of any
model, dataset, or training pipeline.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from losses import (
    ContrastiveLoss,
    IDClassificationLoss,
    ImageToTextCELoss,
    StageTwoLoss,
    TripletLossHardMining,
)


# ---------------------------------------------------------------------------
# ContrastiveLoss — symmetric i2t + t2i cross-entropy
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_random_features_hit_log_batch_size(self):
        """Uniform-random features ≈ random guessing ≈ log(B) per CE direction."""
        B, D = 64, 64
        img = F.normalize(torch.randn(B, D), dim=-1)
        txt = F.normalize(torch.randn(B, D), dim=-1)
        crit = ContrastiveLoss(temperature=1.0, learnable_temp=False)
        loss = crit(img, txt).item()
        # With temperature=1 and random unit vectors, CE should sit near log(B)=4.159.
        # Give a generous slack — this is a stochastic sanity check.
        assert 3.5 < loss < 5.0, f"expected loss near log({B})≈{math.log(B):.3f}, got {loss:.3f}"

    def test_perfectly_aligned_pairs_give_low_loss(self):
        """When image_i == text_i exactly, loss should be near 0 at high temperature."""
        B, D = 16, 32
        shared = F.normalize(torch.randn(B, D), dim=-1)
        crit = ContrastiveLoss(temperature=0.01, learnable_temp=False)
        loss = crit(shared, shared).item()
        # temperature=0.01 → logits scale by 100 → diagonal dominates → loss ≈ 0
        assert loss < 0.1, f"perfectly aligned pairs should give near-zero loss, got {loss:.4f}"

    def test_symmetric_in_i2t_and_t2i(self):
        """Swapping image and text features should give the same loss."""
        B, D = 8, 16
        a = F.normalize(torch.randn(B, D), dim=-1)
        b = F.normalize(torch.randn(B, D), dim=-1)
        crit = ContrastiveLoss(temperature=0.1, learnable_temp=False)
        assert torch.allclose(crit(a, b), crit(b, a))

    def test_learnable_temperature_is_parameter(self):
        crit = ContrastiveLoss(temperature=0.07, learnable_temp=True)
        assert crit.log_temp.requires_grad is True
        assert isinstance(crit.log_temp, torch.nn.Parameter)

    def test_fixed_temperature_is_buffer(self):
        crit = ContrastiveLoss(temperature=0.07, learnable_temp=False)
        assert not isinstance(crit.log_temp, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# IDClassificationLoss — CE with label smoothing
# ---------------------------------------------------------------------------

class TestIDClassificationLoss:
    def test_confident_correct_logits_give_low_loss(self):
        num_classes = 10
        logits = torch.full((4, num_classes), -10.0)
        labels = torch.tensor([0, 3, 7, 9])
        for i, lbl in enumerate(labels):
            logits[i, lbl] = 10.0
        crit = IDClassificationLoss(label_smoothing=0.0)
        assert crit(logits, labels).item() < 0.01

    def test_label_smoothing_adds_floor(self):
        """With label smoothing > 0, loss never reaches exactly 0 even for perfect logits."""
        num_classes = 10
        logits = torch.full((4, num_classes), -100.0)
        labels = torch.tensor([0, 3, 7, 9])
        for i, lbl in enumerate(labels):
            logits[i, lbl] = 100.0
        crit = IDClassificationLoss(label_smoothing=0.1)
        assert crit(logits, labels).item() > 0.01


# ---------------------------------------------------------------------------
# TripletLossHardMining — batch-hard mining
# ---------------------------------------------------------------------------

class TestTripletLossHardMining:
    def test_known_hardest_triple(self):
        """Hand-constructed 3 embeddings, 2 identities — verify exact margin."""
        # id 0: two close points; id 1: one faraway point
        embs = torch.tensor([
            [1.0, 0.0, 0.0],   # id 0, a
            [0.9, 0.1, 0.0],   # id 0, b  — close positive
            [0.0, 1.0, 0.0],   # id 1     — faraway negative
        ])
        # L2-normalize like the real pipeline does
        embs = F.normalize(embs, p=2, dim=-1)
        labels = torch.tensor([0, 0, 1])

        margin = 0.3
        crit = TripletLossHardMining(margin=margin)
        loss = crit(embs, labels).item()

        # Compute expected manually
        d = torch.cdist(embs, embs, p=2)
        # For anchor 0: hardest pos = 1, hardest neg = 2
        # For anchor 1: hardest pos = 0, hardest neg = 2
        # For anchor 2: no positives — skipped
        ap0 = d[0, 1].item(); an0 = d[0, 2].item()
        ap1 = d[1, 0].item(); an1 = d[1, 2].item()
        expected = (max(0.0, ap0 - an0 + margin) + max(0.0, ap1 - an1 + margin)) / 2
        assert abs(loss - expected) < 1e-5, f"expected {expected:.5f}, got {loss:.5f}"

    def test_zero_loss_when_negatives_are_far_enough(self):
        """If neg - pos > margin, triplet loss is 0."""
        embs = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],   # identical positive — ap=0
            [-1.0, 0.0],  # antipodal negative — an=2
        ])
        labels = torch.tensor([0, 0, 1])
        crit = TripletLossHardMining(margin=0.3)
        assert crit(embs, labels).item() == pytest.approx(0.0, abs=1e-6)

    def test_no_positives_returns_zero(self):
        """If no identity has ≥2 images in the batch, loss is 0 (not NaN/crash)."""
        embs = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2])  # all unique
        crit = TripletLossHardMining(margin=0.3)
        loss = crit(embs, labels)
        assert loss.item() == pytest.approx(0.0)
        assert not torch.isnan(loss).item()

    def test_backward_through_duplicate_pairs_is_finite(self):
        """PK sampler oversamples with replacement when an identity has < K
        images, producing batches with literally identical embeddings.
        The euclidean distance between them is zero, and d/dx sqrt(x) at
        x=0 is infinite. The triplet loss must use an epsilon so backward
        doesn't inject NaN/Inf into the model weights."""
        # Leaf tensor first so .grad is populated after backward.
        raw = torch.tensor(
            [[1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],   # identical to above
             [0.0, 1.0, 0.0]],
            requires_grad=True,
        )
        embs = F.normalize(raw, p=2, dim=-1)
        labels = torch.tensor([0, 0, 1])

        crit = TripletLossHardMining(margin=0.3)
        loss = crit(embs, labels)
        loss.backward()

        assert torch.isfinite(loss).item(), f"loss is non-finite: {loss}"
        assert torch.isfinite(raw.grad).all().item(), (
            f"backward produced non-finite gradients: {raw.grad}"
        )


# ---------------------------------------------------------------------------
# ImageToTextCELoss
# ---------------------------------------------------------------------------

class TestImageToTextCELoss:
    def test_aligned_image_text_gives_low_loss(self):
        """Image feature matches its own text feature exactly → CE near 0."""
        num_ids, D = 10, 8
        text_feats = F.normalize(torch.randn(num_ids, D), dim=-1)
        # Take 4 images whose features exactly equal their identity's text feature
        labels = torch.tensor([0, 3, 7, 2])
        img_feats = text_feats[labels].clone()

        crit = ImageToTextCELoss(temperature=0.01)
        assert crit(img_feats, text_feats, labels).item() < 0.1

    def test_wrong_label_gives_high_loss(self):
        num_ids, D = 10, 8
        text_feats = F.normalize(torch.randn(num_ids, D), dim=-1)
        labels_correct = torch.tensor([0, 3, 7, 2])
        img_feats = text_feats[labels_correct].clone()
        labels_wrong = torch.tensor([5, 5, 5, 5])  # mismatched

        crit = ImageToTextCELoss(temperature=0.01)
        loss_right = crit(img_feats, text_feats, labels_correct).item()
        loss_wrong = crit(img_feats, text_feats, labels_wrong).item()
        assert loss_wrong > loss_right * 10, "wrong labels should be much worse"


# ---------------------------------------------------------------------------
# StageTwoLoss — combined 3-loss wrapper
# ---------------------------------------------------------------------------

class TestStageTwoLoss:
    def test_components_are_weighted_sum(self):
        """Total loss must equal λ_id * l_id + λ_tri * l_tri + λ_i2t * l_i2t."""
        B, D, num_ids = 8, 16, 12
        img = F.normalize(torch.randn(B, D), dim=-1)
        logits = torch.randn(B, num_ids)
        text = F.normalize(torch.randn(num_ids, D), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # PK sampler style

        crit = StageTwoLoss(lambda_id=1.5, lambda_tri=2.0, lambda_i2t=0.5)
        total, comp = crit(img, logits, text, labels)

        expected = 1.5 * comp["id"] + 2.0 * comp["tri"] + 0.5 * comp["i2t"]
        assert abs(total.item() - expected) < 1e-5

    def test_returns_dict_with_all_components(self):
        B, D, num_ids = 4, 8, 4
        img = F.normalize(torch.randn(B, D), dim=-1)
        logits = torch.randn(B, num_ids)
        text = F.normalize(torch.randn(num_ids, D), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])

        crit = StageTwoLoss()
        _, comp = crit(img, logits, text, labels)
        assert set(comp.keys()) == {"id", "tri", "i2t", "total"}
        for v in comp.values():
            assert isinstance(v, float)  # .item() called internally
