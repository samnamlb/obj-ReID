# Local validation record

Validated November 13, 2026, from `aleem-final` commit `00c2eec`, with changes kept on the local `aleem-final-review` branch. No changes have been pushed.

## Environment

- Windows, Python 3.13.5, CPU execution.
- Locked installation: PyTorch 2.11.0, torchvision 0.26.0, open-clip-torch 3.3.0, pandas 3.0.2, pyarrow 23.0.1, pytest 9.0.3.
- `python -m uv sync --locked` installed the dependencies and the editable project successfully.

## Completed checks

| Check | Outcome |
|---|---|
| Regression suite | 9 tests passed |
| Real CLIP ViT-B/16 offline inference | Passed with generated images and random weights |
| Training data loader | Passed with generated Parquet metadata, sparse identity IDs and P-by-K batches |
| Both training stages | Passed one epoch each using a small differentiable CLIP substitute |
| Projected checkpoint round trip | Passed; saved 16-D projection detected automatically, conflicting dimension rejected |
| Retrieval metrics | Known AP/mINP values, self-match exclusion, camera exclusions, short CMC curves and invalid rankings tested |
| CLIP synthetic efficiency command | Passed on CPU; produced a JSON report |
| ResNet50 synthetic efficiency command | Passed on CPU; produced a JSON report |
| Python compilation | Passed for application modules |
| Command entry points | All eight scripts returned help successfully |
| Dependency lock consistency | Offline lock check passed |
| Patch formatting | Git diff whitespace check passed |

Generated local reports:

- `results/smoke_test.json`
- `results/efficiency_clip_reid_random_init_cpu_synthetic.json`
- `results/efficiency_resnet50_random_init_cpu_synthetic.json`

These reports are ignored by Git. Synthetic metrics and timing numbers are diagnostic only. The benchmark checks were allowed to overlap and are not a controlled performance comparison; use the sequential comparison procedure in the README for reportable results.

The initial restricted test runs encountered Windows temporary-directory permissions; reruns with temporary-file access succeeded. The application itself required no dataset or pretrained weight downloads for these offline checks.

## Corrections affecting results

- Restored the missing training loader and added identity mapping checks.
- Centralized evaluation splits across CLIP prediction, ResNet prediction, efficiency measurement and ground truth.
- Replaced silent missing-image filtering with an error so gallery indices stay aligned.
- Excluded exact self-images, required full gallery permutations and padded CMC curves after filtering.
- Fixed reduced-dimension Stage 2 alignment, keeping text alignment in native 512-D space.
- Treated same-identity Stage 1 samples as positive pairs and kept frozen backbones in evaluation mode.
- Added portable CPU memory measurement and removed the redundant CLIP GPU-to-CPU-to-GPU transfer in benchmarking.
- Added warmup/repeat information, software/hardware context and explicit metric scope to efficiency reports.

## Recovered artifacts and user-run CUDA validation

After the initial offline checks, both checkpoints and the Dataset A image inventory were recovered. All 30,561 referenced images were extracted with no missing paths, and the Stage 2 model passed inference on four real images. See [the historical archive](archive/README.md) for original training-split provenance limitations.

The project owner subsequently ran the cleaned efficiency script in Colab and supplied the recovered-checkpoint JSON now preserved in `archive/project_history.zip`. It reports a real-data, checkpoint-backed A100-SXM4-40GB run with three repeats and batch size 16: 34.1686 images/s, 494.4238 MiB peak allocated CUDA memory, 21.3497 s encoding plus ranking, 200 queries and 529 gallery images. Embeddings are 512-D; the encoder has 86,192,640 parameters.

Local arithmetic checks passed for the repeat medians, throughput, total time, image counts, embedding storage and similarity-matrix storage. This is a reviewed user-run CUDA report, not an independently repeated local GPU experiment. The exact split manifests, experiment configuration, hardware record and full dependency environment have now been supplied with the report. Local checks confirmed 4,376 training images / 800 identities, 514 validation images / 100 identities, and 529 test images / 100 identities. All partitions are image- and identity-disjoint, all image paths exist, and test identities are absent from the supplied original training metadata. The shared splitter yields the report's exact 200-query / 529-gallery workload. These are newly defined experiment splits, not recovered evidence of the old checkpoint's training membership. Those recovered-checkpoint-only files do not establish fresh training; the separate completed training evidence is described below.

## Remaining verification

Full training convergence, historical A100 throughput and MPS execution have not been established. The new checkpoint's held-out accuracy is now independently re-scored below; user-run CUDA efficiency is recorded for both experiments. Dataset images and checkpoints are present locally but excluded from Git. Linux/macOS and Python 3.11/3.12 have not been tested locally; a GitHub Actions workflow is provided for Windows/Linux/macOS checks but has not been run remotely.

The historical accuracy reports were generated under the earlier evaluation behavior and must be regenerated. The new README preserves them as historical artifacts and explains the limitations. It makes no claim that the new code reproduces their scores.

## Completed new training experiment

The full new Colab run is now saved under `experiments/dataset_a_run/`. Both new checkpoints record the exact 800 training identities, matching the saved manifest. The Stage 2 checkpoint records epoch 5 and seed 42, differs from the recovered old checkpoint, loads strictly, and passes finite unit-norm 512-D inference on two held-out real images.

All three saved prediction CSVs were reevaluated locally against their exact test metadata. Every reported accuracy metric matched: ResNet50 mAP 58.0678%, pretrained CLIP mAP 54.6533%, and fine-tuned CLIP mAP 69.7899%. All query/gallery counts and encoding-timing calculations matched the saved reports and comparison CSV. The experiment verifier records hashes in `experiments/dataset_a_run/verification.json`.

The new data restoration command successfully reconstructed the 529-image test dataset without changing the original data. The run remains a single-seed custom subset experiment, not a full-course or multi-seed benchmark. Original-checkpoint provenance limitations and historical accuracy concerns above do not negate this separate, documented new comparison.
