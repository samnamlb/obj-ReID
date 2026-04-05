# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class StudentModel:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

        # Load a pretrained ResNet-50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Projection head: 2048 -> 512
        self.embedding_layer = nn.Linear(2048, 512)

        # Put model on device
        self.backbone.to(self.device)
        self.embedding_layer.to(self.device)

        # Set eval mode (VERY IMPORTANT)
        self.backbone.eval()
        self.embedding_layer.eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W) already normalized with ImageNet stats
        returns: (B, D) L2-normalized embeddings
        """
        images = images.to(self.device)

        # CNN forward
        features = self.backbone(images)     # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)

        embeddings = self.embedding_layer(features)

        # L2 normalization (required for cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu()

    @property
    def embedding_dim(self) -> int:
        return 512