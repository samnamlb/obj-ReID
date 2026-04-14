#!/usr/bin/env python3
"""
predict.py — Generate a submission-ready prediction CSV from a trained CLIP-ReID model.

The output CSV is evaluated directly by evaluate.py.

Usage (zero-shot CLIP baseline — no training needed):
    python predict.py \
        --dataset_root ./datasets/dataset_a \
        --dataset_name dataset_a \
        --output predictions/dataset_a.csv

Usage (fine-tuned model after Stage 2):
    python predict.py \
        --dataset_root ./datasets/dataset_a \
        --dataset_name dataset_a \
        --checkpoint checkpoints/stage2_last.pt \
        --output predictions/dataset_a.csv

Then evaluate:
    python evaluate.py \
        --student_id YOUR_ID \
        --prediction predictions/dataset_a.csv \
        --datasets dataset_a
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.model import StudentModel


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Inference dataset
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """Loads a flat list of image paths for batch encoding."""

    def __init__(self, dataset_root: str, image_paths: list, image_size: int = 224):
        self.root = dataset_root
        self.paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.paths[idx])
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_images(
    model: StudentModel,
    dataset: InferenceDataset,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Return L2-normalized embeddings ordered by dataset index."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    emb_list, idx_list = [], []
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Encoding", leave=False):
            # StudentModel.encode() handles device transfer and L2-normalization
            emb = model.encode(images)
            emb_list.append(emb.numpy())
            idx_list.append(indices.numpy())

    embeddings = np.vstack(emb_list)
    order = np.concatenate(idx_list)
    return embeddings[np.argsort(order)]   # restore sequential order


def load_query_gallery(dataset_root: str, dataset_name: str):
    """
    Return (query_paths, gallery_paths) exactly matching evaluate.py's split logic
    so that query indices align with evaluation expectations.
    """
    df = pd.read_parquet(os.path.join(dataset_root, "test.parquet"))

    if dataset_name == "dataset_a":
        query_paths, gallery_paths = [], []
        for pid, group in df.groupby("identity"):
            paths = group["image_path"].values.tolist()
            if len(paths) >= 2:
                query_paths.extend(paths[:2])
                gallery_paths.extend(paths[2:])
            else:
                gallery_paths.extend(paths)
        gallery_paths.extend(query_paths)   # standard ReID: query images also in gallery
    else:   # dataset_b
        query_df   = df[df["split"] == "query"]
        gallery_df = df[df["split"] == "gallery"]
        query_paths   = query_df["image_path"].tolist()
        gallery_paths = gallery_df["image_path"].tolist()

    return query_paths, gallery_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ReID prediction CSV from a trained CLIP-ReID model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset_root",  type=str, required=True,
                        help="Root of the dataset (contains test.parquet and images/)")
    parser.add_argument("--dataset_name",  type=str, required=True,
                        choices=["dataset_a", "dataset_b"])
    parser.add_argument("--output",        type=str, required=True,
                        help="Output CSV path, e.g. predictions/dataset_a.csv")

    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Path to Stage 2 checkpoint (.pt). "
                             "Omit to run zero-shot pretrained CLIP.")
    parser.add_argument("--proj_dim",      type=int, default=None,
                        help="Projection head output dim, if used during training. "
                             "Must match --proj_dim passed to train_stage2.py.")

    parser.add_argument("--top_k",         type=int, default=50,
                        help="Number of gallery items to rank per query (default 50).")
    parser.add_argument("--batch_size",    type=int, default=128)
    parser.add_argument("--image_size",    type=int, default=224)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--device",        type=str, default="cuda")
    args = parser.parse_args()

    # Respect --device mps on Apple Silicon; only fall back to CPU when the
    # requested backend is actually unavailable.
    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = args.device
    if device != args.device:
        print(f"[INFO] {args.device} not available — falling back to CPU")

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt_label = args.checkpoint or "none (zero-shot pretrained CLIP)"
    print(f"Loading StudentModel  checkpoint={ckpt_label}")
    model = StudentModel(
        device=device,
        checkpoint=args.checkpoint,
        proj_dim=args.proj_dim,
    )
    print(f"Embedding dim: {model.embedding_dim}")

    # ── Load paths ──────────────────────────────────────────────────────────
    query_paths, gallery_paths = load_query_gallery(args.dataset_root, args.dataset_name)
    print(f"Queries : {len(query_paths)}")
    print(f"Gallery : {len(gallery_paths)}")

    # ── Encode ──────────────────────────────────────────────────────────────
    img_size = args.image_size

    print("Encoding query images...")
    query_emb = encode_images(
        model,
        InferenceDataset(args.dataset_root, query_paths, img_size),
        args.batch_size,
        args.num_workers,
    )

    print("Encoding gallery images...")
    gallery_emb = encode_images(
        model,
        InferenceDataset(args.dataset_root, gallery_paths, img_size),
        args.batch_size,
        args.num_workers,
    )

    # ── Rank ────────────────────────────────────────────────────────────────
    print("Computing rankings...")
    # query_emb and gallery_emb are already L2-normalized → dot product = cosine similarity
    similarity = query_emb @ gallery_emb.T                      # (n_query, n_gallery)
    top_k = min(args.top_k, similarity.shape[1])
    rankings = np.argsort(-similarity, axis=1)[:, :top_k]       # descending

    # ── Save ────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query_index": q_idx,
            "ranked_gallery_indices": ",".join(str(x) for x in rankings[q_idx]),
        }
        for q_idx in range(len(query_paths))
    ]
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predictions saved → {args.output}  ({len(rows)} queries, top-{top_k} gallery)")


if __name__ == "__main__":
    main()
