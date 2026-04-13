
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip



class LearnableTextEncoder(nn.Module):
    """
    Wraps the frozen CLIP text encoder with per-identity learnable prompt tokens.

    Prompt template (at the embedding level):
        [SOS] [a] [photo] [of] [a]  <X1> <X2> ... <XM>  [category] [.] [EOS] [PAD...]

    Only the <Xi> token embeddings are learnable; everything else is frozen.

    Args:
        clip_model:      The open_clip CLIP model object.
        num_identities:  Number of training identities (one prompt set per ID).
        num_tokens:      Number of learnable tokens M per identity (default 4).
        category:        The class-level word appended after the prompts, e.g.
                         "animal", "vehicle", or "person".
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_identities: int,
        num_tokens: int = 4,
        category: str = "animal",
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.context_length = clip_model.context_length  # 77

        # Borrow CLIP text components — these will be frozen externally
        self.token_embedding = clip_model.token_embedding     # nn.Embedding
        self.positional_embedding = clip_model.positional_embedding  # (77, D)
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection     # (D, D)

        embed_dim = self.token_embedding.weight.shape[1]  # 512 for ViT-B/16

        # Shape: (num_identities, M, embed_dim); small init avoids destabilising early training
        self.prompt_tokens = nn.Parameter(
            torch.randn(num_identities, num_tokens, embed_dim) * 0.02
        )

        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        with torch.no_grad():
            prefix_tok = tokenizer(["a photo of a"])[0]          # (77,)
            suffix_tok = tokenizer([f"a {category}."])[0]         # (77,)

        # prefix: [SOS, "a", "photo", "of", "a"]  (5 tokens, no EOS)
        prefix_ids = self._trim(prefix_tok, keep_sos=True, keep_eos=False)

        # suffix: ["<category>", ".", EOS]  (skip SOS from the suffix template)
        suffix_ids = self._trim(suffix_tok, keep_sos=False, keep_eos=True)

        self.register_buffer("prefix_ids", prefix_ids)
        self.register_buffer("suffix_ids", suffix_ids)

        # EOS sits at this position in the constructed sequence
        self._eos_pos = len(prefix_ids) + num_tokens + len(suffix_ids) - 1

        total = len(prefix_ids) + num_tokens + len(suffix_ids)
        assert total <= self.context_length, (
            f"Prompt too long: {total} tokens > context_length {self.context_length}. "
            f"Reduce num_tokens or shorten the category string."
        )

    @staticmethod
    def _trim(
        tokens: torch.Tensor,
        keep_sos: bool = True,
        keep_eos: bool = True,
    ) -> torch.Tensor:
        """Return the meaningful (non-padding) portion of a tokenized sequence."""
        SOS, EOS = 49406, 49407
        t = tokens.tolist()
        eos_pos = t.index(EOS)
        seq = t[: eos_pos + 1]         # include EOS
        if not keep_eos:
            seq = seq[:-1]
        if not keep_sos and seq[0] == SOS:
            seq = seq[1:]
        return torch.tensor(seq, dtype=torch.long)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask used by CLIP's causal text transformer."""
        mask = torch.empty(size, size, device=device)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, identity_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            identity_indices: (B,) LongTensor of identity IDs in [0, num_identities).
        Returns:
            (B, 512) L2-normalized text feature vectors.
        """
        B = identity_indices.shape[0]
        device = identity_indices.device

        with torch.no_grad():
            pfx = self.token_embedding(self.prefix_ids.to(device))  # (P, D)
            sfx = self.token_embedding(self.suffix_ids.to(device))  # (S, D)

        pfx = pfx.unsqueeze(0).expand(B, -1, -1)                    # (B, P, D)
        sfx = sfx.unsqueeze(0).expand(B, -1, -1)                    # (B, S, D)
        prompts = self.prompt_tokens[identity_indices]                # (B, M, D)

        x = torch.cat([pfx, prompts, sfx], dim=1)                   # (B, P+M+S, D)
        seq_len = x.shape[1]

        if seq_len < self.context_length:
            pad = torch.zeros(
                B, self.context_length - seq_len, x.shape[-1],
                device=device, dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=1)                          # (B, 77, D)

        x = x + self.positional_embedding.to(device)                # (B, 77, D)

        # open_clip Transformer expects (B, seq_len, D) — batch first
        mask = self._causal_mask(self.context_length, device)
        x = self.transformer(x, attn_mask=mask)

        x = self.ln_final(x.to(self.ln_final.weight.dtype))
        x = x[:, self._eos_pos]                                      # (B, D)
        x = x @ self.text_projection                                 # (B, D)

        return F.normalize(x, p=2, dim=-1)


class CLIPReIDModel(nn.Module):
    """
    The full training model.

    Stage 1 — freeze image_encoder + text_encoder, train prompt_tokens only.
    Stage 2 — freeze text_encoder + prompt_tokens, train image_encoder + classifier.

    Args:
        num_identities: Number of training identities.
        num_tokens:     Learnable tokens per identity (default 4).
        category:       Class-level word for the text prompt.
        proj_dim:       If not None, add a Linear(512, proj_dim) projection head
                        after the image encoder (reduces embedding dim for efficiency).
    """

    def __init__(
        self,
        num_identities: int,
        num_tokens: int = 4,
        category: str = "animal",
        proj_dim: int = None,
    ):
        super().__init__()

        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )

        self.image_encoder = clip_model.visual    # input (B,3,H,W) → (B,512)

        self.text_encoder = LearnableTextEncoder(
            clip_model=clip_model,
            num_identities=num_identities,
            num_tokens=num_tokens,
            category=category,
        )
        embed_dim = 512
        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Linear(embed_dim, proj_dim, bias=False)
            embed_dim = proj_dim

        self.classifier = nn.Linear(embed_dim, num_identities, bias=False)

        self.embed_dim = embed_dim

    def set_stage1(self):
        """Configure for Stage 1: only prompt_tokens train."""
        # Freeze image encoder
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        # Freeze text encoder backbone (everything except prompt_tokens)
        for name, p in self.text_encoder.named_parameters():
            p.requires_grad_(name == "prompt_tokens")
        # Freeze classifier
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        if self.proj is not None:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def set_stage2(self):
        """Configure for Stage 2: image_encoder + classifier train; text branch frozen."""
        # Unfreeze image encoder
        for p in self.image_encoder.parameters():
            p.requires_grad_(True)
        # Freeze entire text branch (encoder + prompt tokens)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        # Unfreeze classifier and optional projection head
        for p in self.classifier.parameters():
            p.requires_grad_(True)
        if self.proj is not None:
            for p in self.proj.parameters():
                p.requires_grad_(True)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalized image embeddings.
        images: (B, 3, H, W)  →  (B, embed_dim)
        """
        feats = self.image_encoder(images)           # (B, 512)
        if self.proj is not None:
            feats = self.proj(feats)
        return F.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def precompute_text_features(
        self, num_identities: int, batch_size: int = 256, device: torch.device = None
    ) -> torch.Tensor:
        """
        Pre-compute all identity text features at the start of Stage 2.
        Returns a fixed (num_identities, 512) tensor stored on `device`.
        Call this ONCE before the Stage 2 training loop.
        """
        if device is None:
            device = next(self.parameters()).device
        all_ids = torch.arange(num_identities, device=device)
        features = []
        for start in range(0, num_identities, batch_size):
            batch_ids = all_ids[start : start + batch_size]
            feat = self.text_encoder(batch_ids)      # (B, 512)
            features.append(feat)
        return torch.cat(features, dim=0)            # (num_identities, 512)

    def forward(
        self,
        images: torch.Tensor,
        identity_indices: torch.Tensor,
        stage: int = 1,
    ):
        """
        Args:
            images:           (B, 3, H, W)
            identity_indices: (B,) LongTensor of ground-truth identity IDs
            stage:            1 or 2

        Stage 1 returns: (image_features, text_features)
            both (B, 512) L2-normalized — used for contrastive loss

        Stage 2 returns: (image_features, logits)
            image_features (B, embed_dim) L2-normalized
            logits         (B, num_identities) raw classifier scores
        """
        image_features = self.encode_image(images)          # (B, D), normalized

        if stage == 1:
            text_features = self.text_encoder(identity_indices)  # (B, 512), normalized
            return image_features, text_features
        else:
            logits = self.classifier(image_features)             # (B, num_ids)
            return image_features, logits
