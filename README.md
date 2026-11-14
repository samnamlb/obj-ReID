# Object Re-Identification with CLIP

A COMP560 project on recognizing the same animal across different images. We fine-tune CLIP to produce image embeddings, then use cosine similarity to retrieve matching individuals from a gallery. The goal is to improve retrieval accuracy while keeping inference practical.

## Results

Evaluated on a wildlife subset with **100 held-out identities, 200 queries, and 529 gallery images**. Training, validation, and test identities are separate.

| Model | Rank-1 | mAP | Images/sec | GPU memory | Embedding size |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 63.00% | 58.07% | 37.35 | 260.60 MiB | 2,048 |
| Pretrained CLIP | 57.00% | 54.65% | 34.08 | 494.42 MiB | 512 |
| **Fine-tuned CLIP** | **73.50%** | **69.79%** | **34.10** | **494.42 MiB** | **512** |

Fine-tuning improved mAP by **15.14 percentage points** over pretrained CLIP, with essentially unchanged inference throughput and memory. ResNet50 was faster and used less GPU memory, but had lower retrieval accuracy.

Measured on an **NVIDIA A100-SXM4-40GB**, batch size 16, with three encoding repeats. Throughput includes image loading, preprocessing, and transfers; memory is peak allocated GPU tensor memory. These results are from one seed on a custom Dataset A subset.

[Experiment details](experiments/dataset_a_run/README.md) · [Results CSV](experiments/dataset_a_run/comparison.csv) · [Executed notebook](experiments/dataset_a_run/Run_Obj_ReID_Colab.ipynb)

## Approach

The model uses CLIP ViT-B/16 and two training stages:

1. Learn identity-specific text prompts while keeping both CLIP encoders frozen.
2. Fine-tune the image encoder using classification, triplet, and image-to-text losses.

Only the image encoder is used at inference, producing normalized 512-dimensional embeddings. This experiment used 800 training identities, 10 Stage 1 epochs, and five Stage 2 epochs.

## Setup

Requires **Python 3.11-3.13**. From the repository folder:

```sh
python -m pip install uv
python -m uv sync --locked
```

For GPU training, use the [Colab notebook](output/colab/Run_Obj_ReID_Colab.ipynb) or an environment with CUDA-enabled PyTorch. The local Windows lock uses CPU wheels. Datasets and trained weights are not included in Git.

## Try it without a dataset

Verify the saved comparison or run inference on generated images:

```sh
python -m uv run --locked python scripts/verify_experiment.py
python -m uv run --locked python scripts/smoke_test.py
```

The first command recomputes the saved accuracy scores. The second checks the pipeline with random weights and generated images.

## Use your own data

Place your images and metadata in this layout:

```text
datasets/my_data/
  images/
  train.parquet
  test.parquet
```

Metadata needs `image_path` and `identity`; test metadata also needs `camera_id`. Image paths are relative to the dataset folder. Keep training and test identities separate.

With a trained checkpoint at `checkpoints/model.pt`, generate rankings and measure efficiency:

```sh
python -m uv run --locked python predict.py --dataset_root datasets/my_data --dataset_name dataset_a --checkpoint checkpoints/model.pt --output predictions/my_data.csv --device cpu
python -m uv run --locked python scripts/measure_efficiency.py --model clip --dataset_root datasets/my_data --checkpoint checkpoints/model.pt --device cpu
```

Use `--device cuda` on a supported GPU. For training and evaluation settings, see the [experiment notes](experiments/dataset_a_run/README.md).

## Team and references

Aleem Azimov, Leslie Boney, Kevin Parris, Tanveer Thethi, and Cole Vazzana.

Built on [CLIP](https://arxiv.org/abs/2103.00020), [CLIP-ReID](https://arxiv.org/abs/2211.13977), and [OpenCLIP](https://github.com/mlfoundations/open_clip).
