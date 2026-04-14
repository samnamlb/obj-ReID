#!/usr/bin/env python3
"""
Build proper train/val splits from the course-provided test.parquet.

Motivation
----------
The original `train.parquet` in this repo was reverse-engineered from
`test.parquet` and has a 100% image-path overlap with test, i.e. training
on it is training on the eval set. This script replaces it with an
identity-level 80/20 split so that no identity appears in both files.

Output
------
    datasets/dataset_a/train.parquet  — ~80% of identities
    datasets/dataset_a/val.parquet    — ~20% of identities, held out

Usage
-----
    python scripts/make_splits.py \
        --source datasets/dataset_a/test.parquet \
        --out_dir datasets/dataset_a \
        --val_frac 0.2 \
        --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Path to the full labeled parquet to split.")
    p.add_argument("--out_dir", required=True, help="Output directory for train.parquet / val.parquet.")
    p.add_argument("--val_frac", type=float, default=0.2, help="Fraction of identities held out for val.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_parquet(args.source)
    required = {"image_path", "identity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source parquet missing columns: {missing}")

    # Only identities with >= 2 images can go into val (need query + gallery).
    # .copy() because pandas 3.x returns a read-only view from .to_numpy(),
    # and np.random.Generator.shuffle requires a writeable array.
    counts = df.groupby("identity").size()
    val_eligible = counts[counts >= 2].index.to_numpy().copy()
    val_ineligible = counts[counts < 2].index.to_numpy()

    rng = np.random.default_rng(args.seed)
    rng.shuffle(val_eligible)
    n_val = int(round(len(val_eligible) * args.val_frac))
    val_ids = set(val_eligible[:n_val].tolist())
    # Singletons always go to train — they're useless for ReID eval anyway.
    train_ids = set(val_eligible[n_val:].tolist()) | set(val_ineligible.tolist())

    assert val_ids.isdisjoint(train_ids), "Identity split is not disjoint"

    train_df = df[df["identity"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["identity"].isin(val_ids)].reset_index(drop=True)

    # Set split labels so downstream loaders (which filter on 'split') work.
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="test")  # evaluate.py expects 'test' for dataset_a

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train.parquet"
    val_out = out_dir / "val.parquet"
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    # Sanity: identity sets must be disjoint, path sets must be disjoint.
    assert set(train_df["identity"]).isdisjoint(set(val_df["identity"]))
    assert set(train_df["image_path"]).isdisjoint(set(val_df["image_path"]))

    print(f"Source: {args.source}")
    print(f"  rows: {len(df):,}  identities: {df['identity'].nunique():,}")
    print()
    print(f"Train -> {train_out}")
    print(f"  rows: {len(train_df):,}  identities: {train_df['identity'].nunique():,}")
    print(f"Val   -> {val_out}")
    print(f"  rows: {len(val_df):,}  identities: {val_df['identity'].nunique():,}")
    print()
    print(f"Identity overlap: {len(set(train_df['identity']) & set(val_df['identity']))}  (should be 0)")
    print(f"Path overlap:     {len(set(train_df['image_path']) & set(val_df['image_path']))}  (should be 0)")


if __name__ == "__main__":
    main()
