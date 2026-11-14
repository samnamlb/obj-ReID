# Original Dataset A and checkpoint recovery

The original artifacts described here have since been supplemented by a separately trained, documented experiment. See [experiments/dataset_a_run/README.md](experiments/dataset_a_run/README.md) for the new checkpoint and held-out comparison. The missing provenance below applies to the old checkpoint, not the new model.

Inspected November 13, 2026. The original metadata and checkpoints were preserved.

## Recovered files

- `checkpoints/stage1_prompts.pt`: 800 identities, four prompt tokens per identity, 512-dimensional prompt vectors, category `animal`.
- `checkpoints/stage2_last.pt`: CLIP ViT-B/16 image encoder, 800-class training classifier, metadata records epoch 5. No separate retrieval projection head; inference embeddings are 512-dimensional.
- `datasets/dataset_a/train.parquet`: 24,532 images, 6,712 identities.
- `datasets/dataset_a/test.parquet`: 30,561 images, 8,391 identities.
- Both metadata files have `image_path`, `split`, `identity`, and `camera_id`, no null entries, and no duplicate image paths within each file.

## Functional checkpoint check

The Stage 2 encoder loaded strictly into StudentModel without downloading pretrained weights. Four recovered real images produced finite 512-dimensional embeddings with unit L2 norms. The local details are in `results/checkpoint_recovery.json`. This establishes that the saved image encoder is usable; it does not establish retrieval quality or controlled throughput.

## Split provenance requires care

Every supplied training image is also present in the supplied test metadata. All 6,712 supplied training identities likewise occur in that test file. Consequently, the entire test metadata file should not be presented as an independent held-out test set.

The test file contains 6,029 additional images from 1,679 identities absent from the supplied training metadata. That is a candidate identity-disjoint evaluation subset relative to this training metadata. It is not automatically verified as disjoint from the checkpoint's actual training data: the checkpoint was trained with 800 identities, while this metadata lists 6,712, and the old checkpoint saves no identity mapping or training-image manifest.

Do not overwrite the original metadata or retrain merely to make identity counts match. The training classifier is not needed for image-only inference. For a defensible held-out evaluation, recover the original training subset/notebook if possible. Otherwise, explicitly disclose the uncertainty around the old checkpoint, or retrain with a documented split for a new reproducible experiment.

## Evidence recovered from the original notebook

`Obj_ReID_Training.ipynb` explains the different identity count. Its Stage 1 command explicitly uses `train_5gb.parquet`; its saved output reports 800 identities and ten training epochs. Stage 2 output likewise reports 800 identities and five epochs. These counts are consistent with the supplied checkpoints. The Stage 2 command does not explicitly select a Parquet file, so the original loader's default behavior still needs verification.

The notebook's archive listing includes:

- `dataset_a/train_5gb.parquet`
- `dataset_a/val.parquet`
- `test_debug_1gb/test_debug.parquet`
- the original `loader.py`

These are different files from the supplied large `train.parquet` and `test.parquet`. Their contents and original loader are needed to verify the actual split. The notebook provides no code that reconstructs their selection from the larger metadata.

Saved output identifies an NVIDIA A100-SXM4-40GB with Python 3.12, torch 2.10.0+cu128, torchvision 0.25.0+cu128, and open-clip-torch 3.3.0. The notebook clones `aleem-dev`, rather than `aleem-final`, without pinning a commit. Its efficiency output records 130 queries and 296 gallery images, confirming a much smaller historical evaluation workload than the currently recovered dataset.

The prediction cell's saved output is a path error, while subsequent cells evaluate an existing prediction CSV. Thus, the saved accuracy output does not by itself establish that this notebook regenerated those predictions from the supplied checkpoint. All code cells have null execution counts, and directory-changing/setup steps are incomplete or inconsistent, so do not execute the notebook unchanged as a reproducible workflow.

Recovery targets named by the notebook are the Google Drive folder `COMP560_ReID_Backup/obj-ReID/datasets/` and the original roughly 900 MB `ReID_dataset.tar.gz`. The downloaded `reid_project_final.zip` explicitly excluded `obj-ReID/datasets/*`, so it is unlikely to contain the missing split metadata. The later Drive backup copied the whole repository without that exclusion. Preserve the original notebook as historical evidence.

## Archive recovery

`scripts/prepare_dataset.py` extracts only image files referenced by these metadata files into `datasets/dataset_a/`, preserving the expected `images/...` paths. It rejects unsafe paths and duplicate referenced archive entries, skips matching-size existing files, and does not change the metadata. It records missing paths and overlap counts in `results/dataset_a_recovery.json`.

Recovery completed: all 30,561 unique referenced images were found and extracted, with zero missing archive paths. Their total uncompressed size is 5,704,370,029 bytes.

```sh
python -m uv run --locked python scripts/prepare_dataset.py --archive images.tar.gz --dataset_root datasets/dataset_a
```

Images, archive, checkpoints, and local recovery reports are excluded from Git. No Dataset B vehicle data has been established by this recovery.
