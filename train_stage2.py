# train_stage2.py
#
# Stage 2: Fine-tune the image encoder using three combined losses.
#
# Requires Stage 1 checkpoint (checkpoints/stage1_prompts.pt).
# Text encoder and prompt tokens are frozen. Text features are pre-computed once.
#
# Outputs:
#   checkpoints/stage2_best.pt
#   checkpoints/stage2_last.pt
#
# Usage:
#   python train_stage2.py \
#       --dataset_root ./datasets \
#       --stage1_checkpoint checkpoints/stage1_prompts.pt \
#       --category animal \
#       --epochs 60

import argparse
import os
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from models.clip_reid_model import CLIPReIDModel
from losses import StageTwoLoss
from datasets.loader import get_train_loader


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, path, metadata=None):
    state = {
        "encoder": model.image_encoder.state_dict(),
        "classifier": model.classifier.state_dict(),
    }
    if model.proj is not None:
        state["proj"] = model.proj.state_dict()
    if metadata:
        state["metadata"] = metadata
    torch.save(state, path)


def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoader — PK sampler required for triplet loss
    loader, num_identities = get_train_loader(
        dataset_root=args.dataset_root,
        split="train",
        pk_sampler=True,
        P=args.P,
        K=args.K,
        debug=args.debug,
        debug_samples=args.debug_samples,
        parquet_path=args.parquet_path,
    )
    print(f"Loaded dataset: {num_identities} identities, batch={args.P * args.K}")

    if args.num_identities is not None and args.num_identities != num_identities:
        print(f"[WARNING] --num_identities override: {args.num_identities} (detected: {num_identities})")
        num_identities = args.num_identities

    model = CLIPReIDModel(
        num_identities=num_identities,
        num_tokens=args.num_tokens,
        category=args.category,
        proj_dim=args.proj_dim,
    ).to(device)

    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location=device)
    ckpt_num_ids = stage1_ckpt.get("num_identities", None)
    if ckpt_num_ids is not None and ckpt_num_ids != num_identities:
        raise ValueError(
            f"Stage 1 checkpoint has {ckpt_num_ids} identities but current dataset "
            f"has {num_identities}. They must match."
        )
    model.text_encoder.prompt_tokens.data.copy_(
        stage1_ckpt["prompt_tokens"].to(device)
    )
    print(f"Loaded Stage 1 prompts from: {args.stage1_checkpoint}")

    model.set_stage2()

    print("Pre-computing text features for all identities...")
    model.eval()
    with torch.no_grad():
        all_text_features = model.precompute_text_features(
            num_identities=num_identities,
            device=device,
        )
    all_text_features = all_text_features.detach()
    print(f"Text features shape: {all_text_features.shape}")

    criterion = StageTwoLoss(
        lambda_id=args.lambda_id,
        lambda_tri=args.lambda_tri,
        lambda_i2t=args.lambda_i2t,
        label_smoothing=0.1,
        triplet_margin=args.triplet_margin,
        i2t_temperature=0.07,
    ).to(device)

    encoder_params = list(model.image_encoder.parameters())
    head_params = list(model.classifier.parameters())
    if model.proj is not None:
        head_params += list(model.proj.parameters())

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr_encoder},
            {"params": head_params,    "lr": args.lr_head},
        ],
        weight_decay=1e-4,
    )

    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\nStarting Stage 2 for {args.epochs} epoch(s)...")
    print(f"  Encoder LR: {args.lr_encoder}  |  Head LR: {args.lr_head}")
    print(f"  PK: P={args.P}, K={args.K}  ->  batch size {args.P * args.K}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()

        totals = {"id": 0.0, "tri": 0.0, "i2t": 0.0, "total": 0.0}
        n_batches = 0

        # Collect a sample batch for the health check before iterating
        # (avoids creating a second iterator mid-loop)
        sample_batch = None

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch_idx, (images, identity_indices, _) in enumerate(pbar):
            if batch_idx == 0:
                sample_batch = images[:16].clone()

            images = images.to(device)
            identity_indices = identity_indices.to(device)

            image_features, logits = model(images, identity_indices, stage=2)

            loss, components = criterion(
                image_features=image_features,
                logits=logits,
                all_text_features=all_text_features,
                labels=identity_indices,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in totals:
                totals[k] += components[k]
            n_batches += 1
            pbar.set_postfix({"loss": f"{components['total']:.3f}"})

        scheduler.step()

        avgs = {k: v / max(n_batches, 1) for k, v in totals.items()}
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"total={avgs['total']:.4f} | "
            f"id={avgs['id']:.4f} | "
            f"tri={avgs['tri']:.4f} | "
            f"i2t={avgs['i2t']:.4f}"
        )

        # Embedding health check every 10 epochs
        if epoch % 10 == 0 and sample_batch is not None:
            with torch.no_grad():
                model.eval()
                sample_embs = model.encode_image(sample_batch.to(device))
                std = sample_embs.std(dim=0).mean().item()
                print(f"  [health] embedding std={std:.4f}  (healthy: 0.1-0.5; near 0 = collapse)")
                model.train()

        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"stage2_epoch{epoch}.pt")
            save_checkpoint(model, ckpt_path, metadata={"epoch": epoch})
            print(f"  Saved: {ckpt_path}")

    last_path = os.path.join(args.checkpoint_dir, "stage2_last.pt")
    save_checkpoint(model, last_path, metadata={"epoch": args.epochs})
    print(f"\nStage 2 complete. Final checkpoint: {last_path}")
    print(
        "To load at inference:\n"
        f"  StudentModel(device='cuda', checkpoint='{last_path}', proj_dim={args.proj_dim})"
    )


def parse_args():
    p = argparse.ArgumentParser(description="CLIP-ReID Stage 2: Fine-tune image encoder")

    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--parquet_path", type=str, default=None)
    p.add_argument("--num_identities", type=int, default=None)
    p.add_argument("--stage1_checkpoint", type=str, default="checkpoints/stage1_prompts.pt")
    p.add_argument("--category", type=str, default="animal",
                   choices=["animal", "vehicle", "person"])

    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--proj_dim", type=int, default=None)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=10)

    p.add_argument("--P", type=int, default=16)
    p.add_argument("--K", type=int, default=4)

    p.add_argument("--lr_encoder", type=float, default=5e-6)
    p.add_argument("--lr_head", type=float, default=3.5e-4)

    p.add_argument("--lambda_id",  type=float, default=1.0)
    p.add_argument("--lambda_tri", type=float, default=1.0)
    p.add_argument("--lambda_i2t", type=float, default=0.5)
    p.add_argument("--triplet_margin", type=float, default=0.3)

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
    train_stage2(args)