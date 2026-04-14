"""
Unit tests for evaluate.compute_cmc_map.

This is the function whose output IS our grade. If its math is wrong, every
number we report to the course is wrong. These tests pin down the expected
behavior with inputs small enough that expected mAP/Rank-K/mINP can be
computed by hand.

Conventions mirroring the real pipeline:
- `predictions[i]` is a permutation of gallery indices sorted by descending similarity.
- `matches[k] = 1` iff gallery[predictions[i, k]] has the same identity as query i.
- CMC = cumulative "has at least one positive been seen by rank k", capped at 1.
- AP (per query) = sum over positive ranks k of precision@k, divided by num positives.
- mINP (per query) = num_positives_retrieved_by_last_positive / (rank_of_last_positive + 1).
"""
import numpy as np
import pytest

from evaluate import compute_cmc_map


# ---------------------------------------------------------------------------
# Tiny helpers for building inputs
# ---------------------------------------------------------------------------

def _scenario(predictions, query_pids, gallery_pids,
              query_camids=None, gallery_camids=None,
              max_rank=10, exclude_same_camera=False):
    predictions = np.asarray(predictions, dtype=np.int64)
    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    if query_camids is None:
        query_camids = np.zeros(len(query_pids), dtype=np.int64)
    if gallery_camids is None:
        gallery_camids = np.zeros(len(gallery_pids), dtype=np.int64)
    return compute_cmc_map(
        predictions, query_pids, gallery_pids,
        np.asarray(query_camids), np.asarray(gallery_camids),
        max_rank=max_rank, exclude_same_camera=exclude_same_camera,
    )


# ---------------------------------------------------------------------------
# Perfect-ranking sanity checks
# ---------------------------------------------------------------------------

class TestPerfectRanking:
    def test_single_query_single_positive(self):
        """One query, one positive at rank 0 → Rank-1=1.0, mAP=1.0, mINP=1.0."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1, 2, 3]],
            query_pids=[1],
            gallery_pids=[1, 2, 3, 4],
        )
        assert cmc[0] == pytest.approx(1.0)
        assert mAP == pytest.approx(1.0)
        assert mINP == pytest.approx(1.0)

    def test_perfect_all_positives_first(self):
        """3 positives all ranked first. mAP = 1.0, mINP = 3/3 = 1.0."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1, 2, 3]],
            query_pids=[1],
            gallery_pids=[1, 1, 1, 2],
        )
        assert cmc[0] == pytest.approx(1.0)
        assert mAP == pytest.approx(1.0)
        assert mINP == pytest.approx(1.0), (
            "3 positives at ranks 0,1,2 → last positive at rank 2 → "
            "mINP = 3/3 = 1.0. If this is 1/3 = 0.333 the numerator is "
            "being capped at 1 (standard mINP bug)."
        )


# ---------------------------------------------------------------------------
# Non-trivial rankings — hand-calculated AP and CMC
# ---------------------------------------------------------------------------

class TestHandCalculated:
    def test_single_positive_at_rank_3(self):
        """One positive at rank 3 (index 3). AP = 1/4, mINP = 1/4, Rank-1..3 = 0, Rank-4 = 1."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1, 2, 3]],
            query_pids=[1],
            gallery_pids=[2, 3, 4, 1],
        )
        assert cmc[0] == pytest.approx(0.0)
        assert cmc[1] == pytest.approx(0.0)
        assert cmc[2] == pytest.approx(0.0)
        assert cmc[3] == pytest.approx(1.0)
        assert mAP == pytest.approx(0.25)
        assert mINP == pytest.approx(0.25)

    def test_two_positives_spaced(self):
        """Positives at ranks 0 and 4 of 5-item gallery.
        AP = (1/1 + 2/5) / 2 = 0.7
        mINP = 2 / 5 = 0.4  (two positives, last at index 4)
        """
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1, 2, 3, 4]],
            query_pids=[1],
            gallery_pids=[1, 2, 3, 2, 1],
        )
        expected_ap = (1.0 / 1 + 2.0 / 5) / 2
        assert mAP == pytest.approx(expected_ap), f"expected AP={expected_ap:.4f}, got {mAP:.4f}"
        assert mINP == pytest.approx(2.0 / 5.0), (
            "mINP with 2 positives last at rank 4 should be 2/5 = 0.4. "
            "If this is 1/5 = 0.2 the numerator is being capped at 1."
        )
        assert cmc[0] == pytest.approx(1.0)

    def test_two_queries_averaged(self):
        """q0 perfect, q1 rank-2 positive → Rank-1 = (1+0)/2 = 0.5, mAP = (1.0 + 0.5)/2 = 0.75."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1], [0, 1]],
            query_pids=[1, 2],
            gallery_pids=[1, 2],
        )
        # q0: matches=[1,0] — AP = 1/1 = 1, cmc = [1, 1]
        # q1: matches=[0,1] — AP = 1/2 = 0.5, cmc = [0, 1]
        assert cmc[0] == pytest.approx(0.5)
        assert cmc[1] == pytest.approx(1.0)
        assert mAP == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Camera-exclusion behavior (dataset_b protocol)
# ---------------------------------------------------------------------------

class TestSameCameraExclusion:
    def test_excludes_same_pid_same_cam(self):
        """Query pid=1 cam=0; gallery has two same-cam hits (should be filtered) and one diff-cam hit."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1, 2]],
            query_pids=[1],
            gallery_pids=[1, 1, 1],
            query_camids=[0],
            gallery_camids=[0, 0, 1],
            exclude_same_camera=True,
        )
        # After same-camera filter, only index 2 remains — it's a positive → Rank-1=1.
        assert cmc[0] == pytest.approx(1.0)
        assert mAP == pytest.approx(1.0)

    def test_not_excluded_when_different_pid(self):
        """A gallery item with the same camera but a DIFFERENT identity must NOT be filtered."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1]],
            query_pids=[1],
            gallery_pids=[2, 1],     # index 0 is a distractor (same cam), index 1 is the positive
            query_camids=[0],
            gallery_camids=[0, 1],
            exclude_same_camera=True,
        )
        # Nothing gets filtered (distractor has different pid). Positive is at rank 1.
        assert cmc[0] == pytest.approx(0.0)
        assert cmc[1] == pytest.approx(1.0)
        assert mAP == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_query_with_no_valid_positive_is_skipped(self):
        """When a query has no positives in the gallery, it should be skipped
        entirely — not counted in num_valid_query, not averaged in."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1], [0, 1]],
            query_pids=[1, 99],        # identity 99 has no gallery matches
            gallery_pids=[1, 2],
        )
        # Only q0 should contribute (perfect).
        assert mAP == pytest.approx(1.0)
        assert cmc[0] == pytest.approx(1.0)

    def test_all_queries_unmatched_returns_zeros(self):
        """If every query has zero positives after filtering, metrics should
        be zero — not NaN or crash."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1]],
            query_pids=[99],
            gallery_pids=[1, 2],
            max_rank=2,
        )
        assert mAP == pytest.approx(0.0)
        assert mINP == pytest.approx(0.0)
        assert np.all(cmc == 0.0)

    def test_cmc_is_monotone_non_decreasing(self):
        """Cumulative match curve must never decrease as rank grows."""
        cmc, _, _ = _scenario(
            predictions=[[0, 1, 2, 3, 4]],
            query_pids=[1],
            gallery_pids=[2, 1, 3, 1, 4],
        )
        assert np.all(np.diff(cmc) >= 0), f"CMC is not monotone: {cmc}"

    def test_max_rank_truncates_cmc(self):
        """CMC length must equal max_rank."""
        cmc, _, _ = _scenario(
            predictions=[[0, 1, 2, 3, 4]],
            query_pids=[1],
            gallery_pids=[1, 2, 3, 4, 5],
            max_rank=3,
        )
        assert cmc.shape == (3,)

    def test_output_types_are_json_serializable_after_cast(self):
        """Return types should be numpy types; evaluate.py casts them before json.dump."""
        cmc, mAP, mINP = _scenario(
            predictions=[[0, 1]],
            query_pids=[1],
            gallery_pids=[1, 2],
        )
        assert isinstance(cmc, np.ndarray)
        # mAP / mINP come out as numpy scalars — evaluate.py casts them before dumping.
        assert float(mAP) == pytest.approx(1.0)
        assert float(mINP) == pytest.approx(1.0)
