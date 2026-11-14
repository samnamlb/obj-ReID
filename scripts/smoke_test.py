"""Run real CLIP inference and retrieval on temporary generated images, offline."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from evaluate import evaluate_dataset
from models.model import StudentModel
from predict import InferenceDataset, encode_images, load_query_gallery
from datasets.loader import get_train_loader
from runtime import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/smoke_test.json")
    args = parser.parse_args()
    seed_everything(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for identity in range(2):
            for view in range(3):
                name = f"id{identity}_{view}.png"
                pixels = np.random.default_rng(identity * 10 + view).integers(0, 256, (48, 48, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(root / name)
                rows.append(dict(image_path=name, identity=f"object_{identity}", camera_id=view))
        frame = pd.DataFrame(rows)
        frame.to_parquet(root / "train.parquet", index=False)
        frame.to_parquet(root / "test.parquet", index=False)
        loader, count = get_train_loader(root, pk_sampler=True, P=2, K=2)
        images, labels, _ = next(iter(loader))
        assert images.shape == (4, 3, 224, 224) and count == 2
        print("Training loader / PK sampler: passed")
        # No pretrained download, no dataset, no assertion of retrieval quality.
        model = StudentModel(device="cpu", pretrained=None)
        query, gallery = load_query_gallery(root, "dataset_a")
        q_emb = encode_images(model, InferenceDataset(root, query), 2, 0)
        g_emb = encode_images(model, InferenceDataset(root, gallery), 2, 0)
        assert q_emb.shape == (4, 512) and np.isfinite(q_emb).all()
        np.testing.assert_allclose(np.linalg.norm(q_emb, axis=1), 1, atol=1e-5)
        ranking = np.argsort(-(q_emb @ g_emb.T), axis=1)
        prediction = root / "prediction.csv"
        pd.DataFrame({"query_index": range(len(query)), "ranked_gallery_indices": [",".join(map(str, row)) for row in ranking]}).to_csv(prediction, index=False)
        metrics = evaluate_dataset(prediction, root, "dataset_a")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "passed", "synthetic": True, "weights": "random", "device": "cpu", "torch_version": str(torch.__version__), "checks": ["PK loader", "CLIP ViT-B/16 inference", "normalized 512-D embeddings", "full gallery ranking", "self-match-excluded evaluation"], "synthetic_metrics_not_project_accuracy": metrics}, indent=2))
        print(f"Offline smoke test passed. Report: {output}")


if __name__ == "__main__":
    main()
