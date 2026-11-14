#!/usr/bin/env python3
"""Measure image encoding throughput, full-gallery ranking time and peak memory.

Real-data comparisons use the same shared split and transforms as prediction.
Use --synthetic for an offline random-weight pipeline check, never accuracy claims.
See README.md for timing scopes, memory definitions and comparison commands.
"""
import argparse
import gc
import json
import os
import platform
import statistics
try:
    import resource
except ImportError:
    resource = None
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


from reid_data import load_paths as _load_paths, validate_images


# ---------------------------------------------------------------------------
# Model adapters — a thin uniform interface so we can time both models fairly
# ---------------------------------------------------------------------------

class _ResNetAdapter(nn.Module):
    name = "resnet50_zero_shot"
    embedding_dim = 2048  # resnet50 with fc=Identity outputs 2048-d

    def __init__(self, device, random_weights=False):
        super().__init__()
        self.device = device
        if random_weights:
            self.name = "resnet50_random_init"
        self.backbone = models.resnet50(weights=None if random_weights else models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.to(device).eval()

    @torch.inference_mode()
    def encode(self, images):
        return F.normalize(self.backbone(images.to(self.device)), p=2, dim=1)


class _CLIPAdapter(nn.Module):
    def __init__(self, device, checkpoint=None, proj_dim=None, random_weights=False):
        super().__init__()
        # Import here so CPU-only environments that don't need CLIP don't pay.
        from models.model import StudentModel
        self._student = StudentModel(
            device=str(device), checkpoint=checkpoint, proj_dim=proj_dim,
            pretrained=None if random_weights else "openai"
        )
        self.name = "clip_reid_random_init" if random_weights else "clip_reid" + ("_finetuned" if checkpoint else "_zeroshot")
        self.embedding_dim = self._student.embedding_dim

    @torch.inference_mode()
    def encode(self, images):
        # Keep features on-device until the shared timing loop copies once to CPU.
        return self._student.encode(images, return_cpu=False)


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
    if resource is None:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / (1024 ** 2)
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
        torch.cuda.synchronize(device)
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
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--synthetic", action="store_true", help="Random tensors and random weights; pipeline check only")
    p.add_argument("--synthetic_queries", type=int, default=4)
    p.add_argument("--synthetic_gallery", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.batch_size, args.repeats, args.synthetic_queries, args.synthetic_gallery, args.cpu_threads) < 1 or args.num_workers < 0:
        p.error("batch size, repeats and synthetic sizes must be positive; workers nonnegative")
    if args.model == "resnet" and (args.checkpoint or args.proj_dim):
        p.error("checkpoint and proj_dim are supported only for clip")
    if args.synthetic and (args.checkpoint or args.proj_dim):
        p.error("synthetic uses random weights without checkpoint or projection")
    from runtime import seed_everything
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)

    # Resolve device, same policy as StudentModel.
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    if not args.synthetic:
        q_paths, g_paths = _load_paths(args.dataset_root, args.dataset_name)
        validate_images(args.dataset_root, q_paths + g_paths)

    # Build model
    if args.model == "resnet":
        model = _ResNetAdapter(device, random_weights=args.synthetic)
    else:
        model = _CLIPAdapter(device, checkpoint=args.checkpoint, proj_dim=args.proj_dim, random_weights=args.synthetic)

    param_count = _count_parameters(model)

    # Build datasets
    if args.synthetic:
        from torch.utils.data import TensorDataset
        q_paths, g_paths = list(range(args.synthetic_queries)), list(range(args.synthetic_gallery))
        q_dataset = TensorDataset(torch.randn(len(q_paths), 3, 224, 224), torch.arange(len(q_paths)))
        g_dataset = TensorDataset(torch.randn(len(g_paths), 3, 224, 224), torch.arange(len(g_paths)))
    else:
        q_dataset = _InferenceDataset(args.dataset_root, q_paths)
        g_dataset = _InferenceDataset(args.dataset_root, g_paths)

    print(f"Model       : {model.name}")
    print(f"Device      : {device}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Queries     : {len(q_paths)}")
    print(f"Gallery     : {len(g_paths)}")
    print(f"Embedding D : {model.embedding_dim}")
    print(f"Parameters  : {param_count:,}")
    print()

    query_times, gallery_times, peaks = [], [], []
    for repeat in range(args.repeats):
        print(f"Timing repeat {repeat + 1}/{args.repeats}...")
        q_emb, q_time = _time_encode(model, q_dataset, args.batch_size, args.num_workers, device)
        peaks.append(_peak_memory_mb(device))
        g_emb, g_time = _time_encode(model, g_dataset, args.batch_size, args.num_workers, device)
        peaks.append(_peak_memory_mb(device))
        query_times.append(q_time)
        gallery_times.append(g_time)
    q_time, g_time = statistics.median(query_times), statistics.median(gallery_times)
    peak_after_query, peak_after_gallery = max(peaks[::2]), max(peaks[1::2])

    # Ranking (matmul + argsort). Time it separately — this is part of
    # "total evaluation time" per the proposal.
    print("Timing ranking...")
    rank_start = time.perf_counter()
    sim = q_emb @ g_emb.T
    rankings = np.argsort(-sim, axis=1)
    rank_time = time.perf_counter() - rank_start

    total_images = len(q_paths) + len(g_paths)
    total_encode_time = q_time + g_time
    total_eval_time = total_encode_time + rank_time
    throughput = total_images / total_encode_time

    results = {
        "model_name": model.name,
        "synthetic": args.synthetic,
        "weights": "random" if args.synthetic else "checkpoint" if args.checkpoint else "pretrained",
        "seed": args.seed,
        "repeats": args.repeats,
        "query_times_sec": query_times,
        "gallery_times_sec": gallery_times,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "memory_metric": "cuda_peak_allocated_MiB" if device.type == "cuda" else "process_lifetime_peak_RSS_MiB",
        "timing_scope": "median encoding passes including image loading, preprocessing and CPU embedding copy; CPU full-gallery ranking; excludes model load and warmup",
        "num_workers": args.num_workers,
        "cpu_threads": torch.get_num_threads(),
        "amortized_encode_ms_per_image": 1000 * total_encode_time / total_images,
        "embedding_bytes_per_image": int(q_emb.shape[1] * q_emb.dtype.itemsize),
        "similarity_matrix_bytes": int(sim.nbytes),
        "device": str(device),
        "dataset": "synthetic" if args.synthetic else args.dataset_name,
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
    print(f"Peak memory      : {max(peak_after_query, peak_after_gallery):,.1f} MiB  (process lifetime RSS on MPS/CPU)")
    print(f"Embedding dim    : {model.embedding_dim}")
    print(f"Parameters       : {param_count:,}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"efficiency_{model.name}_{device.type}{'_synthetic' if args.synthetic else ''}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
