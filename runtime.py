"""Small shared runtime helpers for portable, repeatable experiments."""
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available() or requested == "mps" and not torch.backends.mps.is_available():
        print(f"{requested} unavailable; using CPU")
        requested = "cpu"
    return torch.device(requested)
