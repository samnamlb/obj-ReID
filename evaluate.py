#!/usr/bin/env python3

# !python evaluate.py \
#   --student_id 730751385 \
#   --prediction predictions/dataset_a.csv \
#   --datasets_root datasets \
#   --datasets dataset_a

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from reid_data import split_metadata


def compute_cmc_map(predictions, query_pids, gallery_pids, query_camids,
                    gallery_camids, max_rank=50, exclude_same_camera=True,
                    query_paths=None, gallery_paths=None):
    """Exact full-gallery metrics; eligible queries with no retrieved hit score zero."""
    curves, aps, inps = [], [], []
    for q, order in enumerate(predictions):
        eligible = np.ones(len(gallery_pids), dtype=bool)
        if query_paths is not None and gallery_paths is not None:
            eligible &= np.asarray(gallery_paths) != query_paths[q]
        if exclude_same_camera:
            eligible &= ~((gallery_pids == query_pids[q]) & (gallery_camids == query_camids[q]))
        total_relevant = np.sum((gallery_pids == query_pids[q]) & eligible)
        if total_relevant == 0:
            continue
        order = np.asarray(order, dtype=np.int64)
        matches = (gallery_pids[order[eligible[order]]] == query_pids[q]).astype(int)
        cumulative = matches.cumsum()
        curve = np.minimum(cumulative, 1)
        padded = np.zeros(max_rank)
        length = min(len(curve), max_rank)
        padded[:length] = curve[:length]
        if length and length < max_rank:
            padded[length:] = curve[length - 1]
        curves.append(padded)
        aps.append(float(((cumulative / np.arange(1, len(matches) + 1)) * matches).sum() / total_relevant))
        hits = np.flatnonzero(matches)
        inps.append(float(total_relevant / (hits[-1] + 1)) if len(hits) == total_relevant else 0.0)
    if not curves:
        raise ValueError("No evaluable queries: each needs a non-self eligible gallery match")
    return np.mean(curves, axis=0), float(np.mean(aps)), float(np.mean(inps))


def _ground_truth(root, name):
    query, gallery = split_metadata(root, name)
    return (query.identity.to_numpy(), query.camera_id.to_numpy(),
            gallery.identity.to_numpy(), gallery.camera_id.to_numpy(), len(query), len(gallery))


def load_dataset_a_gt(root):
    return _ground_truth(root, "dataset_a")


def load_dataset_b_gt(root):
    return _ground_truth(root, "dataset_b")


def evaluate_dataset(
    prediction_path: str,
    dataset_root: str,
    dataset_name: str,
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict:
    """Evaluate predictions against ground truth."""

    # Load ground truth
    if dataset_name == "dataset_a":
        query_pids, query_camids, gallery_pids, gallery_camids, n_q, n_g = load_dataset_a_gt(dataset_root)
        exclude_same_camera = False
    else:
        query_pids, query_camids, gallery_pids, gallery_camids, n_q, n_g = load_dataset_b_gt(dataset_root)
        exclude_same_camera = True

    # Load predictions
    pred_df = pd.read_csv(prediction_path)
    required_cols = {"query_index", "ranked_gallery_indices"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(f"Prediction CSV must have columns: {required_cols}. Got: {set(pred_df.columns)}")

    # Full permutations prevent duplicate, missing, or truncated entries inflating metrics.
    if n_q == 0 or n_g == 0:
        raise ValueError("Empty query/gallery split")
    expected = set(range(n_q))
    if pred_df.query_index.duplicated().any() or set(pred_df.query_index) != expected:
        raise ValueError("Predictions must contain each query_index exactly once")
    predictions = []
    for _, row in pred_df.sort_values("query_index").iterrows():
        indices = [int(x.strip()) for x in str(row["ranked_gallery_indices"]).split(",") if x.strip()]
        if len(indices) != n_g or set(indices) != set(range(n_g)):
            raise ValueError("Each ranking must be a full gallery permutation; regenerate with --top_k 0")
        predictions.append(indices)
    predicted_queries, missing = expected, 0
    query, gallery = split_metadata(dataset_root, dataset_name)
    max_rank = max(k_values)
    cmc, mAP, mINP = compute_cmc_map(
        np.asarray(predictions), query_pids, gallery_pids, query_camids, gallery_camids,
        max_rank=max_rank, exclude_same_camera=exclude_same_camera,
        query_paths=query.image_path.to_numpy(), gallery_paths=gallery.image_path.to_numpy(),
    )

    # Cast to Python float so json.dump can serialize (numpy float32 is not JSON-serializable)
    rank_metrics = {}
    for k in k_values:
        if k <= len(cmc):
            rank_metrics[f"Rank-{k}"] = float(cmc[k - 1] * 100)
        else:
            rank_metrics[f"Rank-{k}"] = float(cmc[-1] * 100) if len(cmc) > 0 else 0.0

    results = {
        "performance": {
            **rank_metrics,
            "mAP": float(mAP * 100),
            "mINP": float(mINP * 100),
            "combined": float((mAP + cmc[0]) / 2 * 100),
        },
        "submission_info": {
            "num_queries": n_q,
            "num_gallery": n_g,
            "num_predicted_queries": len(predicted_queries.intersection(range(n_q))),
            "num_missing_queries": missing,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="COMP560 Object Re-Identification Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--student_id", type=str, required=True, help="Your student ID")
    parser.add_argument(
        "--prediction", type=str, required=True,
        help="Path to prediction CSV (or directory with dataset_a.csv, dataset_b.csv)",
    )
    parser.add_argument("--datasets_root", type=str, default="./datasets", help="Root directory containing datasets")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["dataset_a", "dataset_b"],
        choices=["dataset_a", "dataset_b"],
        help="Datasets to evaluate on",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("COMP560 Object Re-Identification Evaluation")
    print("=" * 60)
    print(f"Student ID: {args.student_id}")
    print(f"Prediction: {args.prediction}")
    print(f"Datasets: {args.datasets}")
    print("=" * 60)

    all_results = {
        "student_id": args.student_id,
        "timestamp": timestamp,
        "prediction_path": args.prediction,
        "datasets": {},
    }

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name}")
        print("=" * 60)

        dataset_root = os.path.join(args.datasets_root, dataset_name)
        if not os.path.exists(os.path.join(dataset_root, "test.parquet")):
            print(f"  ERROR: Ground truth not found at {dataset_root}")
            all_results["datasets"][dataset_name] = {"error": "ground truth not found"}
            continue

        if os.path.isdir(args.prediction):
            pred_path = os.path.join(args.prediction, f"{dataset_name}.csv")
        else:
            pred_path = args.prediction

        if not os.path.exists(pred_path):
            print(f"  ERROR: Prediction file not found at {pred_path}")
            all_results["datasets"][dataset_name] = {"error": "prediction file not found"}
            continue

        try:
            results = evaluate_dataset(pred_path, dataset_root, dataset_name)
            all_results["datasets"][dataset_name] = results

            print(f"\nPerformance Metrics:")
            for metric, value in results["performance"].items():
                print(f"  {metric}: {value:.2f}%")

            print(f"\nSubmission Info:")
            for key, value in results["submission_info"].items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["datasets"][dataset_name] = {"error": str(e)}

    output_file = output_dir / f"{args.student_id}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    summary_file = output_dir / f"{args.student_id}_{timestamp}_summary.csv"
    with open(summary_file, "w") as f:
        f.write("dataset,Rank-1,Rank-5,Rank-10,Rank-20,mAP,mINP,combined\n")
        for dataset_name, results in all_results["datasets"].items():
            if "error" not in results:
                perf = results["performance"]
                f.write(f"{dataset_name},{perf.get('Rank-1', 0):.2f},{perf.get('Rank-5', 0):.2f},"
                       f"{perf.get('Rank-10', 0):.2f},{perf.get('Rank-20', 0):.2f},"
                       f"{perf.get('mAP', 0):.2f},{perf.get('mINP', 0):.2f},{perf.get('combined', 0):.2f}\n")
    print(f"Summary saved to: {summary_file}")

    if any("error" in result for result in all_results["datasets"].values()):
        raise SystemExit(1)
    return all_results


if __name__ == "__main__":
    main()
