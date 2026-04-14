# QA Notes — Cole (2026-04-13)

QA review of the repo as of `dcdcff9` ("Kevin worked on the model, should train now, needs better gpu").

**tl;dr:** I found 9 issues. 7 fixed in code, 2 flagged for team discussion. Training
was not actually working due to an inverted temperature in `ContrastiveLoss`, and Stage 2
was silently saving NaN-weighted checkpoints. Both are fixed and covered by regression
tests. I also found a bug in `evaluate.py` that underreports `mINP` by a factor of
`num_positives` per query (~20pp lower than correct). Corrected baseline numbers below.

## Findings

| # | Severity | Finding | Fix |
|---|----------|---------|-----|
| 1 | **Critical** | `datasets/dataset_a/train.parquet` was 100% overlap with `test.parquet` — training on it = training on the eval set. | Replaced with identity-disjoint 80/20 split via `scripts/make_splits.py`. New `train.parquet` has 7,658 identities / 25,493 images. New `val.parquet` has 733 identities / 5,068 images. |
| 2 | High | `evaluate.py` crashed with `TypeError: Object of type float32 is not JSON serializable` at `json.dump`. | Cast `mAP`/`mINP`/`Rank-K` to Python `float` before serializing. |
| 3 | Medium | **Protocol concern.** `evaluate.py`'s dataset_a protocol appends queries to the gallery (`gallery_rows.extend(query_rows)`) with no self-match exclusion. The query image itself ends up at rank 0 for every query, pushing Rank-K metrics trivially near 100%. | Not fixed — this is the course-provided protocol. **Flag to staff on Piazza.** mAP/mINP are still meaningful; Rank-K is not. Report should lead with mAP. |
| 4 | Medium | `open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")` silently substitutes standard GELU for the QuickGELU activation OpenAI originally trained with. Degrades embedding quality ~0.5–1 % mAP. | Added `force_quick_gelu=True` in `models/clip_reid_model.py:167` and `models/model.py:23`. |
| 5 | Low | `train_stage1.py` / `train_stage2.py`: passing `--debug` silently **overrides** `--epochs`, pinning to 2. Users lose their epoch count with no warning. | Not fixed — usability bug. Either drop the hardcode or print a warning. |
| 6 | **Critical** | `ContrastiveLoss` in `losses/__init__.py` computed `logits = sim * temp` instead of `logits = sim / temp`. With `tau = 0.07`, effective scale was 0.07× instead of ~14.3× — gradients were ~200× too small. **This is why Stage 1 loss was stuck at the `log(batch_size) ≈ 4.16` random-guess floor across multiple runs.** Kevin's "the model seems to be working" was misleading — it ran without crashing, but wasn't actually learning. | Changed to `/ temp` in `losses/__init__.py:37-39`. Verified: with the fix, 2-epoch debug run drops loss 4.17 → 3.88 (vs 4.17 → 4.16 before). |
| 7 | **Critical** | `TripletLossHardMining._pairwise_euclidean` used `clamp(min=0.0).sqrt()`. When the PK sampler oversamples with replacement (identity has < K images), the same image appears twice in a batch → pairwise distance = 0 → `d/dx √x at x=0 = ∞` → backward produces NaN → all weights become NaN → `train_stage2.py` silently saves a checkpoint with NaN weights. Caught during the first real Stage 2 smoke run. | Changed floor to `1e-12` in `losses/__init__.py`. Added `test_backward_through_duplicate_pairs_is_finite` regression test. |
| 8 | Medium | `StudentModel.__init__` and `predict.py` silently ignore `--device mps`, pinning Apple Silicon users to CPU (~10× slower encoding). | Fixed in `models/model.py:20-26` and `predict.py:160-167`. |
| 9 | **Critical** | `evaluate.py`'s `mINP` formula uses the **capped** `cmc` array as the numerator: `inp = cmc[max_pos_idx] / (max_pos_idx + 1)`. Since `cmc` is capped at 1, this always gives `1 / rank_of_last_positive` — not the standard mINP of `num_positives / rank_of_last_positive` (Ye et al. 2021). Our reported mINP is **underestimated by a factor of `num_positives` for every multi-positive query**. | Changed to use uncapped `tmp_cmc` in `evaluate.py:89-95`. Covered by 12 new unit tests in `tests/test_evaluate.py`. |

## Corrected baseline numbers

Re-ran `evaluate.py` after all fixes:

| Metric | ResNet50 zero-shot | CLIP ViT-B/16 zero-shot† |
|---|---|---|
| Rank-1  | 99.92 % | 99.92 % |
| Rank-5  | 100.00 % | 100.00 % |
| mAP     | **86.01 %** | 85.61 % |
| mINP    | **71.57 %** (was 48.99 % pre-fix) | 71.32 % (was 51.18 % pre-fix) |

† "CLIP-ReID zero-shot" here is the barely-trained 2-epoch debug checkpoint — functionally equivalent to raw CLIP backbone. Real CLIP-ReID fine-tuning is still pending (item 5 in the plan below).

## Efficiency numbers

Both models on MPS (Apple M-series), batch 64, full dataset_a test set:

| Metric | ResNet50 | CLIP ViT-B/16 |
|---|---|---|
| Throughput      | 120.4 img/s | 83.4 img/s |
| Peak memory     | 848 MB | 1,526 MB |
| Embedding dim   | 2048 | **512** (4× smaller) |
| Parameters      | 23.5 M | 86.2 M |
| Total eval time | 327.5 s | 467.2 s |

CLIP gives us the smaller embeddings we promised in the proposal (storage/retrieval efficiency), at the cost of encode throughput and memory. **Trade-off is defensible for the report.** See `results/efficiency_*.json`.

## What's new in the repo

- `scripts/make_splits.py` — identity-level train/val split generator
- `scripts/measure_efficiency.py` — throughput/memory/param-count measurement
- `tests/` — 40 unit tests for losses, model invariants, evaluate.compute_cmc_map, make_splits
- `datasets/dataset_a/{train,val,test}.parquet` — clean splits (old leaky `train.parquet` deleted; recoverable via git history on `main`)
- `results/` — baseline metrics and efficiency measurements
- `QA_NOTES.md` — this file
- pytest added as dev dependency in `pyproject.toml` / `uv.lock`

## How to run the test suite

```bash
uv sync          # one-time
uv run pytest    # 40 tests, ~8 sec on Apple Silicon
```

## Outstanding work

1. **Longer real training run** on full `train.parquet`, evaluated against `val.parquet`. With the ContrastiveLoss fix, Stage 1 should actually learn; the current Stage 2 checkpoint is from 22 gradient steps and not meaningful. ~1–2 hr on MPS.
2. **Flag finding #3 to the TAs on Piazza** — Rank-K saturation under the provided dataset_a protocol affects every team's reported numbers. Either we're all misreading it, or the protocol is unintentionally trivialized. Worth asking.
3. Dataset B: not included in the course's local-dev bundle. Any dataset_b claims in the report will need to wait for course-graded eval.
