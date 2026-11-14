"""Shared metadata validation and deterministic evaluation splits."""
from pathlib import Path

import pandas as pd


def read_metadata(root, filename="test.parquet", dataset_name="dataset_a"):
    path = Path(root) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}. See README.md for the dataset layout.")
    df = pd.read_parquet(path)
    required = {"image_path", "identity"}
    if filename == "test.parquet":
        required.add("camera_id")
        if dataset_name == "dataset_b":
            required.add("split")
    if not required.issubset(df.columns):
        raise ValueError(f"{path} requires columns {sorted(required)}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError(f"{path} must contain non-null metadata rows")
    if df.image_path.duplicated().any():
        raise ValueError("image_path must be unique in each metadata file")
    for value in df.image_path:
        relative = Path(str(value))
        if relative.is_absolute() or not (Path(root).resolve() / relative).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError(f"image_path must stay inside dataset_root: {value}")
    return df


def split_metadata(root, dataset_name):
    df = read_metadata(root, dataset_name=dataset_name)
    if dataset_name == "dataset_a":
        queries, gallery = [], []
        for _, group in df.groupby("identity", sort=True):
            if len(group) >= 2:
                queries.append(group.iloc[:2])
                gallery.append(group.iloc[2:])
            else:
                gallery.append(group)
        if not queries:
            raise ValueError("Dataset A requires at least one identity with two images")
        query = pd.concat(queries, ignore_index=True)
        gallery = pd.concat([*gallery, query], ignore_index=True)
    elif dataset_name == "dataset_b":
        if not df["split"].isin(["query", "gallery"]).all():
            raise ValueError("Dataset B test split must be query or gallery")
        query = df[df.split == "query"].reset_index(drop=True)
        gallery = df[df.split == "gallery"].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if query.empty or gallery.empty:
        raise ValueError("Query and gallery must both contain images")
    return query, gallery


def load_paths(root, dataset_name):
    query, gallery = split_metadata(root, dataset_name)
    return query.image_path.tolist(), gallery.image_path.tolist()


def validate_images(root, paths):
    missing = [str(p) for p in paths if not (Path(root) / p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images; first: {missing[0]}. Fix paths before running; indices must not change.")
