# Historical and recovered-checkpoint reports

The completed new training/accuracy comparison is in [experiments/dataset_a_run](../experiments/dataset_a_run/README.md). The files described below concern the older recovered checkpoint and historical artifacts.

## Current recovered-checkpoint efficiency

`recovered_checkpoint_run/efficiency_clip_reid_finetuned_cuda.json` is the user-supplied output of the cleaned Colab benchmark. It uses real images and the recovered Stage 2 checkpoint, not synthetic tensors or random weights.

| Setting or metric | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| Batch size / encoding repeats | 16 / 3 |
| Queries / gallery / encoded occurrences | 200 / 529 / 729 |
| Throughput | 34.1686 images/s |
| Peak allocated CUDA tensor memory | 494.4238 MiB |
| Total encoding + ranking | 21.3497 s |
| Query / gallery encoding medians | 5.9871 s / 15.3482 s |
| CPU ranking | 0.01437 s |
| Embeddings | 512-D float32, 2,048 bytes/image |
| Encoder parameters | 86,192,640 |
| Python / PyTorch / CUDA | 3.13.16 / 2.14.1+cu130 / 13.0 |
| Workers / CPU threads | 0 / 4 |

The report's timestamp is `2026-11-14T00:53:04`; it has no timezone offset, so do not infer a local time from it. The runtime path identifies run `20261114_004748`. Full metric scopes are described in the main README.

The JSON calculations were checked locally against its recorded repeat timings and counts. The benchmark was run by the project owner in Colab. This report alone establishes no retrieval accuracy, baseline speedup or training/test independence for the original checkpoint.

## Saved run provenance

The supplied run folder now includes `experiment.json`, `hardware.json`, `environment.txt`, and `split_metadata/train.parquet`, `val.parquet`, and `test.parquet`.

Local validation confirmed:

| Partition | Images | Identities |
|---|---:|---:|
| New training subset | 4,376 | 800 |
| New validation subset | 514 | 100 |
| New test subset | 529 | 100 |

All partitions have distinct image paths and disjoint identities, and every referenced image exists locally. Training/validation images are drawn from the supplied training metadata; all new test identities are absent from that metadata. The shared Dataset A splitter produces exactly 200 queries and 529 gallery images, matching the GPU report.

This validates the **new experiment's** partitioning. It does not establish that the old checkpoint was trained on those 800 selected identities: the original training subset is still missing. The optional benchmark uses the old checkpoint, not a new model trained on these manifests. The epochs in `experiment.json` describe planned training settings and are not evidence that either new training stage was run.

The saved environment identifies NumPy 2.4.6, pandas 2.2.3, pyarrow 23.0.1, PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130. `environment.txt` is an environment record, not a portable installation command: it includes a Colab-specific editable path. Use the project setup instructions for installation.

Save the executed notebook with output alongside the Drive backup as well. Exact split file SHA-256 values:

```text
train.parquet  7ad2534275e75f602ea25b19eb7c40f72f122db6c7353a078fc70bdc6aca7fbe
val.parquet    d9c4e8f72178a0def814cd420d87ba0de50eb4ab5d630604496ce19d392db888
test.parquet   fe3f1d49a7a23ad767e98d734ea05d6b56939e581c2086273c9ccba889534628
```

Keep the source bundle, original image archive and checkpoints in durable backups outside Git. Do not upload the large binaries to the Git repository. The local source bundle and checkpoint currently have these SHA-256 identifiers:

```text
stage2_last.pt
07ff68d32189bdd32c385b453d3c81a7544164399f06bdfccd08b111750ce3d5

obj_reid_colab.zip
b605a23cd80b25b2109e56c093cc3b7952c7dfddac12db92085fd6205bdb4e43
```

These hashes identify the local artifacts; the supplied GPU report does not record hashes, so they are not independent proof of the bytes used remotely. A rerun requires the matching code, checkpoint, metadata, images and environment. Copying old subset counts or changing only the reported dataset name cannot recreate it.

## Historical artifacts

The April 17 JSON/CSV reports and the top-level `efficiency_clip_reid_finetuned_cuda.json` are preserved historical artifacts. The older speed of 111.43 images/s used batch size 64, 130 queries and 296 gallery images, with different measurement code/software. It is not directly comparable with the current run. Its old accuracy results have not been verified under the corrected evaluation protocol.

Synthetic CPU reports and local recovery reports are diagnostic and ignored by Git. They are not trained-model accuracy or controlled comparative performance results.
