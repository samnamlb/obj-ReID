#!/usr/bin/env python3
"""
Measure the efficiency metrics the project is graded on (30% of grade):
  - throughput (images encoded per second)
  - peak memory (whatever the backend can report)
  - embedding dim
  - total evaluation time (encode + rank)
  - model parameter count

Covers both the ResNet50 baseline and the CLIP-ReID StudentModel so we can
compare apples-to-apples. Writes a JSON to results/ so the report can cite
real numbers.

Usage:
    # ResNet50 baseline
    python scripts/measure_efficiency.py \
        --model resnet \
        --dataset_root datasets/dataset_a \
        --dataset_name dataset_a \
        --device mps

    # CLIP-ReID zero-shot (no checkpoint)
    python scripts/measure_efficiency.py \
        --model clip --device mps

    # CLIP-ReID with a trained Stage 2 checkpoint
    !python scripts/measure_efficiency.py \
    --model clip \
    --dataset_root datasets/test \
    --dataset_name dataset_a \
    --checkpoint checkpoints/stage2_last.pt \
    --device cuda
"""
import argparse
import gc
import json
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `models` and other top-level packages importable when this script
# is run from its subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data loading (mirrors resnet_baseline.py/predict.py so we measure the same
# workload those scripts produce)
# ---------------------------------------------------------------------------

class _InferenceDataset(Dataset):
    def __init__(self, root, paths, image_size=224):
        self.root = root
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.paths[idx])).convert("RGB")
        return self.transform(img), idx


def _load_paths(dataset_root: str, dataset_name: str):
    """Same query/gallery split logic as predict.py / resnet_baseline.py."""
    df = pd.read_parquet(os.path.join(dataset_root, "test.parquet"))
    if dataset_name == "dataset_a":
        q, g = [], []
        for _, group in df.groupby("identity"):
            paths = group["image_path"].tolist()
            if len(paths) >= 2:
                q.extend(paths[:2])
                g.extend(paths[2:])
            else:
                g.extend(paths)
        g.extend(q)
    else:
        q = df[df["split"] == "query"]["image_path"].tolist()
        g = df[df["split"] == "gallery"]["image_path"].tolist()
    return q, g


# ---------------------------------------------------------------------------
# Model adapters — a thin uniform interface so we can time both models fairly
# ---------------------------------------------------------------------------

class _ResNetAdapter(nn.Module):
    name = "resnet50_zero_shot"
    embedding_dim = 2048  # resnet50 with fc=Identity outputs 2048-d

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.to(device).eval()

    @torch.inference_mode()
    def encode(self, images):
        return F.normalize(self.backbone(images.to(self.device)), p=2, dim=1)


class _CLIPAdapter(nn.Module):
    def __init__(self, device, checkpoint=None, proj_dim=None):
        super().__init__()
        # Import here so CPU-only environments that don't need CLIP don't pay.
        from models.model import StudentModel
        self._student = StudentModel(
            device=str(device), checkpoint=checkpoint, proj_dim=proj_dim
        )
        self.name = "clip_reid" + ("_finetuned" if checkpoint else "_zeroshot")
        self.embedding_dim = self._student.embedding_dim

    @torch.inference_mode()
    def encode(self, images):
        # StudentModel.encode() handles device transfer and returns CPU tensors.
        # To keep ranking on-device, move it back to the requested device.
        return self._student.encode(images).to(self._student.device)


def _count_parameters(module) -> int:
    """Count parameters, handling both nn.Module and the StudentModel wrapper.

    StudentModel is a plain Python class (not nn.Module), so its internal
    encoder/projection aren't registered as submodules. We reach into them
    directly in that case.
    """
    if isinstance(module, _CLIPAdapter):
        s = module._student
        n = sum(p.numel() for p in s._image_encoder.parameters())
        if s._proj is not None:
            n += sum(p.numel() for p in s._proj.parameters())
        return n
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Memory helpers — backend-aware peak tracking
# ---------------------------------------------------------------------------

def _reset_memory_counters(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # MPS has no peak counter; we approximate with process RSS.
        torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None


def _peak_memory_mb(device: torch.device) -> float:
    """Best-effort peak memory in MB for this backend."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    # For MPS / CPU, fall back to peak process RSS via getrusage.
    # On macOS ru_maxrss is in bytes; on Linux it's in KB. We detect.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 ** 2)   # bytes -> MB
    else:
        return usage / 1024.0         # KB -> MB


# ---------------------------------------------------------------------------
# Timing core
# ---------------------------------------------------------------------------

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_encode(model, dataset, batch_size, num_workers, device, warmup_batches=3):
    """Return (encoded_tensor, elapsed_sec). Warms up before timing."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,   # MPS doesn't support pinned memory
    )

    # Warmup
    warmup_iter = iter(loader)
    for i in range(warmup_batches):
        try:
            images, _ = next(warmup_iter)
        except StopIteration:
            break
        _ = model.encode(images)
    _sync(device)

    # Timed pass (re-iterate to go through everything)
    _reset_memory_counters(device)
    start = time.perf_counter()

    emb_list, idx_list = [], []
    for images, indices in loader:
        emb = model.encode(images)
        emb_list.append(emb.cpu())
        idx_list.append(indices)
    _sync(device)

    elapsed = time.perf_counter() - start

    embeddings = torch.cat(emb_list, dim=0)
    order = torch.cat(idx_list, dim=0)
    embeddings = embeddings[order.argsort()]
    return embeddings.numpy(), elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet", "clip"], required=True)
    p.add_argument("--dataset_root", type=str, default="datasets/dataset_a")
    p.add_argument("--dataset_name", type=str, default="dataset_a",
                   choices=["dataset_a", "dataset_b"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Stage 2 checkpoint (clip model only)")
    p.add_argument("--proj_dim", type=int, default=None)
    p.add_argument("--device", type=str, default="mps",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="results")
    args = p.parse_args()

    # Resolve device, same policy as StudentModel.
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    # Build model
    if args.model == "resnet":
        model = _ResNetAdapter(device)
    else:
        model = _CLIPAdapter(device, checkpoint=args.checkpoint, proj_dim=args.proj_dim)

    param_count = _count_parameters(model)

    # Build datasets
    q_paths, g_paths = _load_paths(args.dataset_root, args.dataset_name)

    print(f"Model       : {model.name}")
    print(f"Device      : {device}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Queries     : {len(q_paths)}")
    print(f"Gallery     : {len(g_paths)}")
    print(f"Embedding D : {model.embedding_dim}")
    print(f"Parameters  : {param_count:,}")
    print()

    # Encode query + gallery (timed + memory tracked)
    print("Timing query encoding...")
    q_emb, q_time = _time_encode(
        model, _InferenceDataset(args.dataset_root, q_paths),
        args.batch_size, args.num_workers, device,
    )
    peak_after_query = _peak_memory_mb(device)

    print("Timing gallery encoding...")
    g_emb, g_time = _time_encode(
        model, _InferenceDataset(args.dataset_root, g_paths),
        args.batch_size, args.num_workers, device,
    )
    peak_after_gallery = _peak_memory_mb(device)

    # Ranking (matmul + argsort). Time it separately — this is part of
    # "total evaluation time" per the proposal.
    print("Timing ranking...")
    rank_start = time.perf_counter()
    sim = q_emb @ g_emb.T
    rankings = np.argsort(-sim, axis=1)[:, :50]
    rank_time = time.perf_counter() - rank_start

    total_images = len(q_paths) + len(g_paths)
    total_encode_time = q_time + g_time
    total_eval_time = total_encode_time + rank_time
    throughput = total_images / total_encode_time

    results = {
        "model_name": model.name,
        "device": str(device),
        "dataset": args.dataset_name,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "num_queries": len(q_paths),
        "num_gallery": len(g_paths),
        "total_images": total_images,
        "embedding_dim": int(model.embedding_dim),
        "parameters": int(param_count),
        "query_encode_time_sec": float(q_time),
        "gallery_encode_time_sec": float(g_time),
        "rank_time_sec": float(rank_time),
        "total_eval_time_sec": float(total_eval_time),
        "throughput_images_per_sec": float(throughput),
        "peak_memory_mb_after_query": float(peak_after_query),
        "peak_memory_mb_after_gallery": float(peak_after_gallery),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # Print summary
    print()
    print("=" * 50)
    print("EFFICIENCY RESULTS")
    print("=" * 50)
    print(f"Throughput       : {throughput:,.1f} images/sec")
    print(f"Total encode time: {total_encode_time:6.2f} sec  "
          f"(query {q_time:.2f} + gallery {g_time:.2f})")
    print(f"Rank time        : {rank_time:6.3f} sec")
    print(f"Total eval time  : {total_eval_time:6.2f} sec")
    print(f"Peak memory      : {peak_after_gallery:,.1f} MB  (RSS-based on MPS/CPU)")
    print(f"Embedding dim    : {model.embedding_dim}")
    print(f"Parameters       : {param_count:,}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"efficiency_{model.name}_{device.type}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
