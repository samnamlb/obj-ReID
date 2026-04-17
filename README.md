# COMP560 Course Project — Object Re-Identification

The goal of the project is to implement a **ReID model** that maps images of objects (animals, vehicles, people, etc.) into discriminative embeddings, enabling retrieval of the same individual across different viewpoints, cameras, and conditions.

## Overview
CLIP-based object re-identification using a two-stage training pipeline.

## Setup
pip install -r requirements.txt

## Training
python train_stage1.py
python train_stage2.py

## Evaluation
python evaluate.py ...

## Efficiency
python scripts/measure_efficiency.py ...

## Results
Dataset A:
- mAP: 88.88%
- mINP: 77.84%
- Combined: 94.44%
- Throughput (CUDA): 111.4 img/s
- Peak GPU Memory ~1GB
- Embedding Dimension: 512
- GPU used to train this: NVIDIA A100-SXM4-40GB

