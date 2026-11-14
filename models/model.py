import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


class StudentModel:

    def __init__(self, device: str = "cuda", checkpoint: str = None, proj_dim: int = None, pretrained="openai"):
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required. Install with: pip install open-clip-torch"
            )

        # Fall back to CPU only when neither CUDA nor (for MPS requests) MPS is available.
        # Previous code ignored --device mps entirely, pinning Apple Silicon runs to CPU.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # quick_gelu=True matches how OpenAI trained the ViT-B/16 CLIP weights.
        # Without it open_clip silently substitutes standard GELU, degrading embedding quality.
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=pretrained if checkpoint is None else None, force_quick_gelu=True
        )
        self._image_encoder = clip_model.visual
        self._image_encoder.to(self.device).eval()

        for p in self._image_encoder.parameters():
            p.requires_grad_(False)

        state = None
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "proj" in state:
                saved_dim = state["proj"]["weight"].shape[0]
                if proj_dim is not None and proj_dim != saved_dim:
                    raise ValueError("proj_dim does not match the checkpoint")
                proj_dim = saved_dim
            elif proj_dim is not None:
                raise ValueError("Checkpoint has no projection head")
        elif proj_dim is not None:
            raise ValueError("A projection head requires a trained checkpoint")
        self._proj = None
        if proj_dim is not None:
            self._proj = nn.Linear(512, proj_dim, bias=False).to(self.device).eval()
        if state is not None:
            self._image_encoder.load_state_dict(state["encoder"] if "encoder" in state else state)
            if self._proj is not None:
                self._proj.load_state_dict(state["proj"])
        self._embedding_dim = proj_dim if proj_dim is not None else 512

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @torch.no_grad()
    def encode(self, images: torch.Tensor, return_cpu: bool = True) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) tensor, ImageNet-normalized.
                    The evaluator handles normalization — do NOT re-normalize here.
        Returns:
            (B, D) L2-normalized float32 embeddings; on CPU by default.
        """
        images = images.to(self.device)
        feats = self._image_encoder(images)
        if self._proj is not None:
            feats = self._proj(feats)
        embeddings = F.normalize(feats.float(), p=2, dim=-1)
        return embeddings.cpu() if return_cpu else embeddings
