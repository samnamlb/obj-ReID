"""
Invariant tests for CLIPReIDModel and StudentModel.

These tests actually instantiate CLIP (weights already cached locally from
prior runs), so they're slower than the loss tests. If CLIP cache is missing
they will download it — that's a one-time ~100 MB cost.
"""
import pytest
import torch

from models.clip_reid_model import CLIPReIDModel
from models.model import StudentModel


@pytest.fixture(scope="module")
def tiny_model() -> CLIPReIDModel:
    """One CLIPReIDModel instance shared across this module's tests."""
    torch.manual_seed(0)
    m = CLIPReIDModel(num_identities=5, num_tokens=4, category="animal")
    m.eval()
    return m


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    # (B, 3, 224, 224) — the input shape CLIP ViT-B/16 expects
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


class TestCLIPReIDModel:
    def test_encode_image_output_shape(self, tiny_model, dummy_batch):
        feats = tiny_model.encode_image(dummy_batch)
        assert feats.shape == (2, 512), f"expected (2, 512), got {feats.shape}"

    def test_encode_image_is_l2_normalized(self, tiny_model, dummy_batch):
        """Every embedding must be a unit vector — this is the contract the
        evaluator and similarity computation rely on."""
        feats = tiny_model.encode_image(dummy_batch)
        norms = feats.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_image_is_deterministic(self, tiny_model, dummy_batch):
        """Same input, same output — no hidden randomness in inference."""
        out1 = tiny_model.encode_image(dummy_batch)
        out2 = tiny_model.encode_image(dummy_batch)
        assert torch.allclose(out1, out2)

    def test_set_stage1_freezes_everything_but_prompts(self, tiny_model):
        tiny_model.set_stage1()
        trainable = {
            name for name, p in tiny_model.named_parameters() if p.requires_grad
        }
        assert trainable == {"text_encoder.prompt_tokens"}, (
            f"Stage 1 should train ONLY prompt_tokens, but {trainable - {'text_encoder.prompt_tokens'}} "
            f"are also unfrozen"
        )

    def test_set_stage2_freezes_text_branch(self, tiny_model):
        tiny_model.set_stage2()
        for name, p in tiny_model.named_parameters():
            if name.startswith("text_encoder"):
                assert not p.requires_grad, f"{name} should be frozen in Stage 2"
            if name.startswith("image_encoder"):
                assert p.requires_grad, f"{name} should train in Stage 2"

    def test_precompute_text_features_shape(self, tiny_model):
        tiny_model.set_stage2()
        feats = tiny_model.precompute_text_features(
            num_identities=5, batch_size=4, device=torch.device("cpu")
        )
        assert feats.shape == (5, 512)
        # Output of the text encoder is L2-normalized per its forward()
        norms = feats.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestStudentModel:
    def test_encode_returns_l2_normalized_on_cpu(self):
        """StudentModel is the inference wrapper used by predict.py — its
        output is what evaluate.py ranks on."""
        torch.manual_seed(0)
        m = StudentModel(device="cpu")
        images = torch.randn(2, 3, 224, 224)
        feats = m.encode(images)

        assert feats.shape == (2, 512)
        assert feats.device.type == "cpu", "encode() must return CPU tensors"
        norms = feats.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_embedding_dim_property(self):
        m = StudentModel(device="cpu")
        assert m.embedding_dim == 512
