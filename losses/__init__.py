
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Symmetric image-to-text and text-to-image cross-entropy loss.

    The similarity matrix is (B x B): entry [i, j] = sim(image_i, text_j).
    Correct matches are on the diagonal.

    Args:
        temperature:    Softmax temperature. Start at 0.07.
        learnable_temp: If True, temperature becomes a trained parameter.
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer("log_temp", torch.tensor(temperature).log())

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (B, D) L2-normalized
            text_features:  (B, D) L2-normalized
        Returns:
            Scalar loss.
        """
        # CLIP convention: logits = similarity / temperature (lower tau → sharper).
        # Previously this code *multiplied* by temp, making tau=0.07 softer (not sharper),
        # which left the contrastive gradient ~200x too small and stalled prompt-token
        # learning at the log(B) random-guess floor.
        temp = self.log_temp.exp()
        logits = (image_features @ text_features.T) / temp    # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0


class IDClassificationLoss(nn.Module):
    """
    Standard cross-entropy on classifier logits with label smoothing.

    Label smoothing (0.1) prevents the model becoming overconfident and
    generally improves generalisation in ReID tasks.
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_identities) raw classifier output
            labels: (B,) ground-truth identity indices
        """
        return self.ce(logits, labels)


class TripletLossHardMining(nn.Module):
    """
    Batch-hard triplet loss.

    For each anchor:
      - Hardest positive: same identity, maximum embedding distance
      - Hardest negative: different identity, minimum embedding distance
    Loss = mean(max(0, d(a,p) - d(a,n) + margin))

    Requires the PK-sampler DataLoader so that every batch contains
    multiple images per identity (otherwise there are no valid positives).

    Args:
        margin: Triplet margin (start with 0.3).
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    @staticmethod
    def _pairwise_euclidean(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute (B, B) Euclidean distance matrix.
        Uses: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>

        Note on the epsilon: d/dx sqrt(x) is infinite at x=0, so naive
        .clamp(min=0).sqrt() produces NaN gradients whenever two embeddings
        in a batch are identical — which happens routinely when the PK
        sampler oversamples with replacement (identity has < K images).
        Clamping to a tiny positive floor keeps the gradient finite.
        """
        sq = embeddings.pow(2).sum(dim=1, keepdim=True)       # (B, 1)
        dist_sq = sq + sq.T - 2.0 * (embeddings @ embeddings.T)
        return dist_sq.clamp(min=1e-12).sqrt()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized image features
            labels:     (B,) identity labels
        Returns:
            Scalar batch-hard triplet loss.
        """
        dist = self._pairwise_euclidean(embeddings)            # (B, B)

        labels_col = labels.unsqueeze(1)                       # (B, 1)
        same_id = labels_col == labels_col.T                   # (B, B)
        diff_id = ~same_id

        # Exclude diagonal (self-distance) from positive mask
        eye = torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
        pos_mask = same_id & ~eye

        # Hardest positive (max distance among same-identity pairs)
        pos_dist = dist.masked_fill(~pos_mask, -1e9)
        hardest_pos = pos_dist.max(dim=1).values              # (B,)

        # Hardest negative (min distance among different-identity pairs)
        neg_dist = dist.masked_fill(~diff_id, 1e9)
        hardest_neg = neg_dist.min(dim=1).values              # (B,)

        triplet = F.relu(hardest_pos - hardest_neg + self.margin)

        # Only average over anchors that have at least one valid positive
        valid = pos_mask.any(dim=1)
        if valid.sum() == 0:
            return triplet.sum() * 0.0
        return triplet[valid].mean()

class ImageToTextCELoss(nn.Module):
    """
    Keeps image features aligned with their learned text representations.

    Computes cosine similarity between each image embedding and ALL identity
    text features (precomputed once at the start of Stage 2), then applies
    cross-entropy using the ground-truth identity as the label.

    Args:
        temperature: Temperature for the similarity logits (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        all_text_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features:    (B, D) L2-normalized image embeddings
            all_text_features: (num_identities, D) fixed text features
            labels:            (B,) ground-truth identity indices
        """
        logits = (image_features @ all_text_features.T) / self.temperature
        return F.cross_entropy(logits, labels)


class StageTwoLoss(nn.Module):
    """
    total = lambda_id * L_id  +  lambda_tri * L_tri  +  lambda_i2t * L_i2tce

    Starting weights from the plan: lambda_id=1.0, lambda_tri=1.0, lambda_i2t=0.5.
    Returns both the total loss and a dict of each component for logging.
    """

    def __init__(
        self,
        lambda_id: float = 1.0,
        lambda_tri: float = 1.0,
        lambda_i2t: float = 0.5,
        label_smoothing: float = 0.1,
        triplet_margin: float = 0.3,
        i2t_temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.lambda_tri = lambda_tri
        self.lambda_i2t = lambda_i2t

        self.id_loss = IDClassificationLoss(label_smoothing=label_smoothing)
        self.tri_loss = TripletLossHardMining(margin=triplet_margin)
        self.i2t_loss = ImageToTextCELoss(temperature=i2t_temperature)

    def forward(
        self,
        image_features: torch.Tensor,
        logits: torch.Tensor,
        all_text_features: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Args:
            image_features:    (B, D) L2-normalized
            logits:            (B, num_identities) classifier scores
            all_text_features: (num_identities, D) fixed text features
            labels:            (B,) ground-truth identity indices

        Returns:
            (total_loss, components)
            components: dict with keys "id", "tri", "i2t", "total"
        """
        l_id = self.id_loss(logits, labels)
        l_tri = self.tri_loss(image_features, labels)
        l_i2t = self.i2t_loss(image_features, all_text_features, labels)

        total = self.lambda_id * l_id + self.lambda_tri * l_tri + self.lambda_i2t * l_i2t

        components = {
            "id": l_id.item(),
            "tri": l_tri.item(),
            "i2t": l_i2t.item(),
            "total": total.item(),
        }
        return total, components
