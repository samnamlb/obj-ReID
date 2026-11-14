"""Recompute saved retrieval scores and validate an experiment's evidence files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from evaluate import evaluate_dataset
from reid_data import split_metadata, validate_images


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiments/dataset_a_run")
    parser.add_argument("--image_root", help="Optionally check the original extracted image inventory")
    parser.add_argument("--output", help="Optional verification JSON path")
    args = parser.parse_args()
    root = Path(args.experiment)
    config = json.loads((root / "experiment.json").read_text())
    metadata = root / "split_metadata"
    frames = {name: pd.read_parquet(metadata / f"{name}.parquet") for name in ("train", "val", "test")}
    for name, frame in frames.items():
        require(frame.identity.nunique() == config[f"{name}_identities"], f"{name}: identity count differs from config")
        require(not frame.image_path.duplicated().any(), f"{name}: duplicate image paths")
        require(not frame[["image_path", "identity", "camera_id"]].isna().any().any(), f"{name}: null metadata")
        if args.image_root:
            validate_images(args.image_root, frame.image_path)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        require(not set(frames[left].identity) & set(frames[right].identity), "Overlapping partition identities")
        require(not set(frames[left].image_path) & set(frames[right].image_path), "Overlapping partition images")
    query, gallery = split_metadata(metadata, "dataset_a")
    summary = pd.read_csv(root / "comparison.csv").set_index("model")
    models = {"resnet": "resnet50_zero_shot", "clip_pretrained": "clip_reid_zeroshot", "clip_finetuned": "clip_reid_finetuned"}
    results = {}
    for label, model in models.items():
        actual = evaluate_dataset(root / "predictions" / f"{label}.csv", metadata, "dataset_a")
        reports = sorted((root / "results").glob(f"{label}_*.json"))
        require(len(reports) == 1, f"Expected one quality report for {label}")
        saved = json.loads(reports[0].read_text())["datasets"]["dataset_a"]
        for metric, value in actual["performance"].items():
            require(math.isclose(value, saved["performance"][metric], abs_tol=1e-8), f"{label}: {metric} differs")
        for metric in ("Rank-1", "mAP", "mINP"):
            require(math.isclose(actual["performance"][metric], summary.loc[label, metric], abs_tol=1e-8), f"{label}: comparison CSV differs")
        efficiency = json.loads((root / "results" / f"efficiency_{model}_cuda.json").read_text())
        require(efficiency["num_queries"] == len(query) and efficiency["num_gallery"] == len(gallery), "Efficiency split count mismatch")
        q, g = statistics.median(efficiency["query_times_sec"]), statistics.median(efficiency["gallery_times_sec"])
        require(math.isclose(q, efficiency["query_encode_time_sec"]) and math.isclose(g, efficiency["gallery_encode_time_sec"]), "Encoding median mismatch")
        require(math.isclose((len(query) + len(gallery)) / (q + g), efficiency["throughput_images_per_sec"]), "Throughput mismatch")
        require(math.isclose(q + g + efficiency["rank_time_sec"], efficiency["total_eval_time_sec"]), "Total timing mismatch")
        require(math.isclose(summary.loc[label, "images/sec"], efficiency["throughput_images_per_sec"]), "Summary throughput mismatch")
        results[label] = actual["performance"]
        print(f"{label}: all saved retrieval metrics and timing calculations match")
    hashes = {}
    for folder in (metadata, root / "new_checkpoints"):
        for path in sorted(folder.glob("*")):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    result = {"status": "passed", "scope": "Saved predictions independently reevaluated locally; GPU timings checked arithmetically, not rerun", "image_existence_checked": bool(args.image_root), "partitions": {name: {"images": len(frame), "identities": int(frame.identity.nunique())} for name, frame in frames.items()}, "queries": len(query), "gallery": len(gallery), "metrics": results, "sha256": hashes}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Experiment verification passed")


if __name__ == "__main__":
    main()
