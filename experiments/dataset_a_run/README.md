# Completed Dataset A subset experiment

This run compares ImageNet ResNet50, pretrained CLIP ViT-B/16, and a newly fine-tuned CLIP-ReID-style encoder on the same held-out wildlife subset. The new model's mAP is 69.79%, versus 54.65% for pretrained CLIP, with essentially unchanged measured inference throughput and GPU memory.

## Experiment design

| Partition | Images | Identities |
|---|---:|---:|
| Training | 4,376 | 800 |
| Validation | 514 | 100 |
| Test | 529 | 100 |

Seed 42 selects training and validation identities from the supplied training inventory. Test identities are drawn from the supplied test inventory after excluding all identities present in that training inventory. Eligible identities have at least three images, with at most eight selected per identity. Images and identities are disjoint across all three partitions. The manifests are saved in `split_metadata/`; all referenced image paths were checked locally.

This is a newly defined development experiment, not a reconstruction of the lost original course split. The original large test inventory contains the original training images; it must not be used wholesale as a held-out test. The new checkpoint records the exact training identities, matching the new training manifest. The older recovered checkpoint is a different artifact and is not used in this accuracy comparison.

Dataset A evaluation uses the first two images per identity as queries and includes all images in the gallery, excluding the exact query image when scoring. The saved manifests produce 200 queries and 529 gallery images. Predictions are full-gallery permutations; missing queries, duplicate indices and truncated rankings are rejected.

## Training

- Backbone: pretrained OpenAI CLIP ViT-B/16 with QuickGELU; four prompt tokens per identity, category `animal`.
- Stage 1: 10 epochs, batch size 64, two data-loader workers, Adam learning rate 0.00035, cosine schedule, seed 42.
- Stage 2: five epochs, P=4 and K=4 (batch size 16), zero workers, AdamW encoder/head learning rates 0.000005 / 0.00035, one warmup epoch followed by cosine scheduling, seed 42.
- Loss weights: identity, triplet and image-to-text alignment = 1 / 1 / 0.5; label smoothing 0.1, triplet margin 0.3, alignment temperature 0.07.
- Images: 224-by-224 resize with ImageNet normalization; training adds random horizontal flip. This respects the course input contract. All compared encoders receive the same evaluation preprocessing.
- No projection head: inference uses the 512-D image encoder only. Text and classifier branches are not used in retrieval.
- The final Stage 2 checkpoint is evaluated; the reserved validation split was not used for checkpoint selection. No test-driven tuning or repeated-seed study is evidenced by this run.

The executed notebook records the Stage 1 command and Stage 2 completion. Stage 1 output was not captured in that cell; its completed prompt checkpoint records the training identity mapping. Stage 2 output and checkpoint metadata confirm epoch 5 and the saved training IDs. `experiment.json` records the updated Stage 1 batch/worker settings.

## Results

| Model | Rank-1 | Rank-5 | mAP | mINP | Images/s | Peak allocated CUDA memory | Dimensions |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 63.00% | 84.00% | 58.07% | 43.65% | 37.35 | 260.60 MiB | 2,048 |
| Pretrained CLIP | 57.00% | 81.50% | 54.65% | 40.92% | 34.08 | 494.42 MiB | 512 |
| Fine-tuned CLIP | 73.50% | 89.50% | 69.79% | 57.75% | 34.10 | 494.42 MiB | 512 |

Fine-tuning gains 15.14 percentage points of mAP and 16.50 points of Rank-1 over pretrained CLIP. This run supports improved representation quality without a material change in measured inference cost. It does not show that fine-tuning accelerates the backbone. ResNet remains faster and smaller in GPU memory, while its 2,048-D embeddings require four times the per-image float32 storage.

All GPU runs used an A100-SXM4-40GB, batch size 16, zero workers, four CPU threads, and three warmed-up encoding repeats. Throughput includes loading, preprocessing, transfers and CPU embedding copies. CPU full-gallery ranking is measured separately; fine-tuned CLIP's total encoding plus ranking is 21.3849 seconds. GPU memory is peak allocated CUDA tensor memory during encoding, not the GPU's full process allocation. The console's parenthetical CPU/MPS RSS note is generic; the CUDA JSON explicitly records `cuda_peak_allocated_MiB`.

Environment records show Python 3.13.16, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA 13.0, pandas 2.2.3 and NumPy 2.4.6. Input loading appears substantial in this end-to-end workload; these speeds should not be described as pure encoder compute benchmarks. Timing differences near 0.1% between the two CLIP models are not evidence of a speed improvement.

## Verification and reproduction

The project owner ran GPU training and benchmarking in Colab. Locally, every saved prediction CSV was reevaluated against the saved manifests and matched every reported Rank-K, mAP, mINP and combined score. The comparison CSV and timing calculations were checked. Both new checkpoints' identity mappings match the training manifest; the Stage 2 checkpoint loaded strictly and produced finite, unit-normalized 512-D embeddings from held-out images. These checks are recorded in `verification.json`; local checkpoint inspection additionally verified epoch 5 and seed 42.

From the repository root, after installation:

```sh
python -m uv run --locked python scripts/verify_experiment.py
```

This rechecks saved accuracy without datasets or weight downloads. With the original images available, add `--image_root datasets/dataset_a` to check their paths. To restore training and test data from the saved manifests:

```sh
python -m uv run --locked python scripts/restore_experiment_data.py --image_root datasets/dataset_a --splits train val test
python -m uv run --locked python train_stage1.py --dataset_root runs/reproduction/dataset_a --epochs 10 --batch_size 64 --num_workers 2 --device cuda --seed 42 --checkpoint_dir runs/reproduction/checkpoints
python -m uv run --locked python train_stage2.py --dataset_root runs/reproduction/dataset_a --stage1_checkpoint runs/reproduction/checkpoints/stage1_prompts.pt --epochs 5 --warmup_epochs 1 --P 4 --K 4 --num_workers 0 --device cuda --seed 42 --checkpoint_dir runs/reproduction/checkpoints
```

GPU installations may require the Colab notebook or a compatible CUDA PyTorch environment instead of the local Windows CPU lock. Recorded seeds do not guarantee bitwise-identical training across environments.

## Artifact map

| Path | Role |
|---|---|
| `comparison.csv` | Three-model result table |
| `results/` | Per-model accuracy and efficiency JSON/CSV |
| `predictions/` | All three full-gallery prediction CSVs |
| `split_metadata/` | Exact train/validation/test manifests |
| `experiment.json` | Run settings |
| `environment.txt`, `hardware.json` | Recorded Colab environment; not a portable requirements file |
| `Run_Obj_ReID_Colab.ipynb` | Original executed notebook retained as evidence |
| `verification.json` | Local checks and metadata/checkpoint SHA-256 values |
| `new_checkpoints/` | New prompt/encoder weights, retained locally and ignored by Git |
| `source_and_original_artifacts.zip` | Original uploaded code and recovered weights, retained locally and ignored by Git |

The executed notebook includes an interactively added configuration-update cell before the configuration's definition. Preserve it as run evidence; use the cleaned notebook in `output/colab/` for a new top-to-bottom run. Keep Drive and local backups. The source archive and weight binaries are excluded from Git. The repository supplies no public checkpoint download; a new user can verify saved predictions, retrain from the manifests with access to the images, or obtain the weights separately from the project owner.

## Limits

This is one seed on a small wildlife subset, with no confidence intervals, Dataset B evaluation, projection-head ablation or external deployment benchmark. The accuracy comparison is within this recorded experiment; it is not the historical 88.88% mAP result or a claim of state-of-the-art performance. Benchmarking used three encoding repeats but only one ranking measurement per model. Additional independent seeds and a larger test set would be needed to establish robustness.
