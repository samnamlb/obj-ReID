# train_stage1.py
#
# Stage 1: Learn per-identity text prompt tokens.
#
# Only self.text_encoder.prompt_tokens has requires_grad=True.
# Both image_encoder and text_encoder backbone are frozen.
#
# Output:
#   checkpoints/stage1_prompts.pt
#
# Usage:
#   python train_stage1.py \
#       --dataset_root ./datasets \
#       --epochs 60 \
#       --category animal
#
#Alt Usage:
#  python .\train_stage1.py
#   --dataset_root .\datasets\dataset_a
#  --parquet_path .\datasets\dataset_a\train.parquet
#  --epochs 60 --category animal

import argparse
import os
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.clip_reid_model import CLIPReIDModel
from losses import ContrastiveLoss
from datasets.loader import get_train_loader


def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader, num_identities = get_train_loader(
        dataset_root=args.dataset_root,
        split="train",
        batch_size=args.batch_size,
        pk_sampler=False,
        debug=args.debug,
        debug_samples=args.debug_samples,
        parquet_path=args.parquet_path,
    )
    print(f"Loaded dataset: {num_identities} identities")

    if args.num_identities is not None and args.num_identities != num_identities:
        print(
            f"[WARNING] --num_identities={args.num_identities} overrides "
            f"the detected count of {num_identities}. "
            f"This will cause index errors if raw identity values exceed the override."
        )
        num_identities = args.num_identities

    model = CLIPReIDModel(
        num_identities=num_identities,
        num_tokens=args.num_tokens,
        category=args.category,
    ).to(device)

    model.set_stage1()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"Stage 1 trainable parameters: {n_params:,}  "
          f"({num_identities} identities x {args.num_tokens} tokens)")

    criterion = ContrastiveLoss(temperature=0.07, learnable_temp=True).to(device)

    optimizer = optim.Adam(
        list(trainable) + list(criterion.parameters()),
        lr=args.lr,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\nStarting Stage 1 for {args.epochs} epoch(s)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        criterion.train()

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, identity_indices, _ in pbar:
            images = images.to(device)
            identity_indices = identity_indices.to(device)

            image_features, text_features = model(images, identity_indices, stage=1)
            loss = criterion(image_features, text_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        temp = criterion.log_temp.exp().item()
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | temp={temp:.4f}")

        if epoch == 5 and avg_loss > 5.0:
            print("[WARNING] Loss still high after 5 epochs. "
                  "Verify image_encoder and text_encoder are frozen.")

    save_path = os.path.join(args.checkpoint_dir, "stage1_prompts.pt")
    torch.save(
        {
            "prompt_tokens": model.text_encoder.prompt_tokens.detach().cpu(),
            "num_tokens": args.num_tokens,
            "num_identities": num_identities,
            "category": args.category,
            "temperature": criterion.log_temp.exp().item(),
        },
        save_path,
    )
    print(f"\nStage 1 complete. Prompts saved to: {save_path}")
    print(f"num_identities={num_identities}  (use this value for --num_identities in Stage 2)")


def parse_args():
    p = argparse.ArgumentParser(description="CLIP-ReID Stage 1: Learn text prompts")

    p.add_argument("--dataset_root", type=str, required=True,
                   help="Directory containing the images/ folder and train.parquet")
    p.add_argument("--parquet_path", type=str, default=None,
                   help="Override path to train.parquet (default: dataset_root/train.parquet)")
    p.add_argument("--num_identities", type=int, default=None,
                   help="Override identity count (default: auto-detected from parquet)")
    p.add_argument("--category", type=str, default="animal",
                   choices=["animal", "vehicle", "person"])

    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3.5e-4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_samples", type=int, default=200)

    args = p.parse_args()
    if args.debug:
        args.epochs = 2
        print(f"[DEBUG] Limiting to {args.debug_samples} samples, 2 epochs.")
    return args


if __name__ == "__main__":
    args = parse_args()
    train_stage1(args)