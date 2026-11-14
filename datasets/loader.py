"""Training loader with contiguous identity labels and balanced P x K batches."""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from reid_data import read_metadata, validate_images


class TrainingDataset(Dataset):
    def __init__(self, root, frame):
        self.root = Path(root)
        self.frame = frame.reset_index(drop=True)
        self.identities = sorted(frame.identity.unique().tolist())
        mapping = {identity: i for i, identity in enumerate(self.identities)}
        self.labels = [mapping[x] for x in self.frame.identity]
        validate_images(root, self.frame.image_path)
        # Preserve the original project's ImageNet normalization for checkpoint compatibility.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = self.frame.iloc[index].image_path
        with Image.open(self.root / path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index], path


class PKBatchSampler(Sampler):
    """Sample P distinct identities, K images each; repeat short identities."""
    def __init__(self, labels, P, K):
        self.groups = {}
        for index, label in enumerate(labels):
            self.groups.setdefault(label, []).append(index)
        if P < 2 or K < 2 or P > len(self.groups):
            raise ValueError(f"PK sampling requires 2 <= P <= {len(self.groups)} identities and K >= 2")
        self.P, self.K = P, K
        self.batches = max(1, (len(labels) + P * K - 1) // (P * K))

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            batch = []
            for identity in random.sample(list(self.groups), self.P):
                indices = self.groups[identity]
                batch.extend(random.sample(indices, self.K) if len(indices) >= self.K else random.choices(indices, k=self.K))
            yield batch


def get_train_loader(dataset_root, split="train", batch_size=64, pk_sampler=False,
                     P=16, K=4, debug=False, debug_samples=200, parquet_path=None,
                     num_workers=0):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    metadata = Path(parquet_path) if parquet_path else Path(dataset_root) / f"{split}.parquet"
    frame = read_metadata(metadata.parent, metadata.name)
    if debug:
        # Keep the label vocabulary stable across stages, even in debug mode.
        if debug_samples < frame.identity.nunique():
            raise ValueError("debug_samples must cover every training identity")
        first = frame.groupby("identity", sort=True).head(1)
        rest = frame.drop(first.index).head(max(0, debug_samples - len(first)))
        import pandas as pd
        frame = pd.concat([first, rest]).sort_index()
    dataset = TrainingDataset(dataset_root, frame)
    if pk_sampler:
        loader = DataLoader(dataset, batch_sampler=PKBatchSampler(dataset.labels, P, K), num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader, len(dataset.identities)
