# datasets/loader.py
#
# Data loading utilities for the CLIP-ReID pipeline.
#
# Public API:
#   get_train_loader(dataset_root, split="train", batch_size=64, pk_sampler=False,
#                   P=16, K=4, image_size=224, num_workers=4,
#                   debug=False, debug_samples=200, parquet_path=None)
#       -> (DataLoader, num_identities)
#
# The DataLoader yields batches of (images, labels, camids) where:
#   images  : (B, 3, 224, 224) normalized tensor
#   labels  : (B,) contiguous integer identity indices (0-based)
#   camids  : (B,) camera ID integers (may be 0 for dataset_a which has no real cameras)

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmentations for training: crop, flip, color jitter, random erasing."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])


def get_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Minimal transforms for inference: resize and normalize only."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class ReIDDataset(Dataset):
    """
    Loads ReID images described by a list of record dicts.

    Each record must have:
        image_path : str   — path relative to dataset_root
        label      : int   — contiguous 0-based identity index
        camid      : int   — camera ID (may be 0 if unavailable)
    """

    def __init__(self, dataset_root: str, records: List[dict], transform):
        self.root = dataset_root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        rec = self.records[idx]
        path = Path(self.root) / rec["image_path"]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, rec["label"], rec["camid"]


class PKBatchSampler(Sampler):
    """
    Yields batches of size P*K by sampling K images from each of P random identities.

    Guarantees every batch contains positive pairs, which triplet loss requires.
    Only identities with >= 2 images are eligible (need at least one positive pair).

    Args:
        labels : list/array of integer identity labels (0-based, contiguous)
        P      : number of identities per batch
        K      : number of images per identity per batch
    """

    def __init__(self, labels: List[int], P: int, K: int):
        self.P = P
        self.K = K

        self.pid_to_indices: dict = defaultdict(list)
        for idx, pid in enumerate(labels):
            self.pid_to_indices[pid].append(idx)

        # Only keep identities that have at least 2 images so positives exist
        self.valid_pids = [
            pid for pid, idxs in self.pid_to_indices.items() if len(idxs) >= 2
        ]

        if len(self.valid_pids) < P:
            raise ValueError(
                f"PKBatchSampler needs at least P={P} identities with >=2 images, "
                f"but only {len(self.valid_pids)} qualify. "
                f"Lower P or use more training data."
            )

        # Number of complete batches per epoch
        self._n_batches = len(self.valid_pids) // P

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        pids = self.valid_pids.copy()
        random.shuffle(pids)

        for start in range(0, self._n_batches * self.P, self.P):
            batch_pids = pids[start: start + self.P]
            batch_indices = []
            for pid in batch_pids:
                pool = self.pid_to_indices[pid]
                if len(pool) >= self.K:
                    chosen = random.sample(pool, self.K)
                else:
                    # Oversample with replacement when identity has < K images
                    chosen = random.choices(pool, k=self.K)
                batch_indices.extend(chosen)
            yield batch_indices


def get_train_loader(
    dataset_root: str,
    split: str = "train",
    batch_size: int = 64,
    pk_sampler: bool = False,
    P: int = 16,
    K: int = 4,
    image_size: int = 224,
    num_workers: int = 4,
    debug: bool = False,
    debug_samples: int = 200,
    parquet_path: Optional[str] = None,
) -> Tuple[DataLoader, int]:
    """
    Build a training DataLoader from a Parquet metadata file.

    Args:
        dataset_root  : Root directory; image paths in parquet are relative to this.
        split         : Which split to use ("train"). Filters on the 'split' column
                        when present.
        batch_size    : Batch size for standard (non-PK) loader.
        pk_sampler    : If True, use the PK batch sampler (required for Stage 2).
        P             : Identities per batch (PK sampler only).
        K             : Images per identity per batch (PK sampler only).
        image_size    : Square crop/resize target.
        num_workers   : DataLoader worker processes.
        debug         : Subsample to debug_samples for fast iteration.
        debug_samples : Max samples in debug mode.
        parquet_path  : Override path to parquet; defaults to dataset_root/train.parquet.

    Returns:
        (loader, num_identities)
        num_identities: count of unique identities in the loaded split (0-based labels).
    """
    if parquet_path is None:
        parquet_path = str(Path(dataset_root) / "train.parquet")

    df = pd.read_parquet(parquet_path)

    # Filter to the requested split if the column exists
    if "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)

    if len(df) == 0:
        available = df["split"].unique().tolist() if "split" in df.columns else ["(no split column)"]
        raise ValueError(
            f"No rows found for split='{split}' in {parquet_path}. "
            f"Available splits: {available}"
        )

    # Debug mode: simple random sample (avoids pandas 2.x groupby column-drop bug)
    if debug and len(df) > debug_samples:
        df = df.sample(n=min(debug_samples, len(df)), random_state=42).reset_index(drop=True)

    # Remap identities to contiguous 0-based integer labels
    unique_ids = sorted(df["identity"].unique())
    id_to_label = {pid: i for i, pid in enumerate(unique_ids)}
    num_identities = len(unique_ids)

    # Camera IDs — dataset_a uses dummy 0s; fill NaN so int() never fails
    if "camera_id" in df.columns:
        df["camera_id"] = pd.to_numeric(df["camera_id"], errors="coerce").fillna(0).astype(int)
    else:
        df["camera_id"] = 0

    records = [
        {
            "image_path": row["image_path"],
            "label": id_to_label[row["identity"]],
            "camid": int(row["camera_id"]),
        }
        for _, row in df.iterrows()
    ]

    transform = get_train_transforms(image_size)
    dataset = ReIDDataset(dataset_root, records, transform)
    labels = [r["label"] for r in records]

    if pk_sampler:
        sampler = PKBatchSampler(labels, P=P, K=K)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    return loader, num_identities
