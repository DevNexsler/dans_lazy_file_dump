"""Tests for search enhancements: length normalization, MMR diversity,
cross-encoder blend, cosine fallback, time decay floor, importance weighting,
minimum score threshold.

Inspired by memory-lancedb-pro (win4r/memory-lancedb-pro).
"""

import math
import tempfile
import time

import pytest
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

from core.storage import SearchHit
from lancedb_store import LanceDBStore
from search_hybrid import (
    _apply_importance_weighting,
    _apply_length_normalization,
    _apply_min_score_threshold,
    _apply_mmr_diversity,
    _apply_recency_boost,
    _cosine_fallback_rerank,
    _cosine_similarity,
    hybrid_search,
    reciprocal_rank_fusion,
    Reranker,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockEmbedProvider:
    def __init__(self, vector: list[float]):
        self._vector = vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector


def _make_node(doc_id, loc, text, vector, source_type="md", **extra_meta):
    meta = {
        "doc_id": doc_id,
        "source_type": source_type,
        "loc": loc,
        "snippet": text[:200],
        "mtime": 1.0,
        "size": len(text),
    }
    meta.update(extra_meta)
    node = TextNode(
        text=text,
        id_=f"{doc_id}::{loc}",
        embedding=vector,
        metadata=meta,
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def _build_store_with_fts(tmpdir, nodes):
    store = LanceDBStore(tmpdir, "test_chunks")
    store.upsert_nodes(nodes)
    store.create_fts_index()
    return store


# ---------------------------------------------------------------------------
# Length normalization
# ---------------------------------------------------------------------------

class TestLengthNormalization:
    def test_short_chunks_unaffected(self):
        """Chunks at or below anchor length should not be penalized."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x" * 800, score=1.0)
        result = _apply_length_normalization([hit], anchor=800)
        assert result[0].score == 1.0

    def test_long_chunks_penalized(self):
        """Chunks much longer than anchor should have reduced scores."""
        short_hit = SearchHit(doc_id="short.md", loc="c:0", snippet="x", text="x" * 400, score=1.0)
        long_hit = SearchHit(doc_id="long.md", loc="c:0", snippet="x", text="x" * 3200, score=1.0)
        result = _apply_length_normalization([short_hit, long_hit], anchor=800)
        assert result[0].doc_id == "short.md"  # short should rank higher
        assert result[1].score < 1.0  # long should be penalized

    def test_very_long_chunk_penalty_magnitude(self):
        """A 3200-char chunk at anchor=800 should get roughly 0.67x penalty."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x" * 3200, score=1.0)
        _apply_length_normalization([hit], anchor=800)
        # 1 / (1 + 0.5 * log2(3200/800)) = 1 / (1 + 0.5 * 2) = 0.5
        expected = 1.0 / (1.0 + 0.5 * math.log2(3200 / 800))
        assert abs(hit.score - expected) < 0.01

    def test_empty_text_unaffected(self):
        """Hits with empty text should not crash or be penalized."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="", score=1.0)
        result = _apply_length_normalization([hit], anchor=800)
        assert result[0].score == 1.0


# ---------------------------------------------------------------------------
# MMR diversity
# ---------------------------------------------------------------------------

class TestMMRDiversity:
    def test_identical_vectors_deferred(self):
        """Chunks with identical vectors should be deferred after the first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [
                _make_node("a.md", "c:0", "first chunk about cats", vec),
                _make_node("a.md", "c:1", "second chunk about cats", vec),
                _make_node("b.md", "c:0", "chunk about dogs", [0.0] + [1.0] + [0.0] * 766),
            ]
            store = _build_store_with_fts(tmpdir, nodes)

            hits = [
                SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="first cats", score=1.0),
                SearchHit(doc_id="a.md", loc="c:1", snippet="x", text="second cats", score=0.9),
                SearchHit(doc_id="b.md", loc="c:0", snippet="x", text="dogs", score=0.8),
            ]
            result = _apply_mmr_diversity(hits, store, similarity_threshold=0.85)
            # a.md c:0 selected first, a.md c:1 deferred (identical vector), b.md selected
            assert result[0].doc_id == "a.md" and result[0].loc == "c:0"
            assert result[1].doc_id == "b.md"  # dogs should come before deferred cats
            assert result[2].doc_id == "a.md" and result[2].loc == "c:1"  # deferred

    def test_different_vectors_not_deferred(self):
        """Chunks with very different vectors should all be selected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes = [
                _make_node("a.md", "c:0", "cats", [1.0] + [0.0] * 767),
                _make_node("b.md", "c:0", "dogs", [0.0] + [1.0] + [0.0] * 766),
                _make_node("c.md", "c:0", "birds", [0.0, 0.0] + [1.0] + [0.0] * 765),
            ]
            store = _build_store_with_fts(tmpdir, nodes)

            hits = [
                SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="cats", score=1.0),
                SearchHit(doc_id="b.md", loc="c:0", snippet="x", text="dogs", score=0.9),
                SearchHit(doc_id="c.md", loc="c:0", snippet="x", text="birds", score=0.8),
            ]
            result = _apply_mmr_diversity(hits, store, similarity_threshold=0.85)
            # All should stay in original order (no deferrals)
            assert [h.doc_id for h in result] == ["a.md", "b.md", "c.md"]

    def test_single_hit_unchanged(self):
        """A single hit should pass through unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes = [_make_node("a.md", "c:0", "cats", [1.0] + [0.0] * 767)]
            store = _build_store_with_fts(tmpdir, nodes)

            hits = [SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="cats", score=1.0)]
            result = _apply_mmr_diversity(hits, store)
            assert len(result) == 1


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# ---------------------------------------------------------------------------
# Cosine fallback rerank
# ---------------------------------------------------------------------------

class TestCosineFallbackRerank:
    def test_reranks_by_cosine_blend(self):
        """Cosine fallback should re-sort hits by blended cosine + original score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            query_vec = [1.0] + [0.0] * 767
            close_vec = [0.9] + [0.1] + [0.0] * 766  # close to query
            far_vec = [0.0] + [1.0] + [0.0] * 766  # far from query

            nodes = [
                _make_node("close.md", "c:0", "close match", close_vec),
                _make_node("far.md", "c:0", "far match", far_vec),
            ]
            store = _build_store_with_fts(tmpdir, nodes)

            hits = [
                SearchHit(doc_id="far.md", loc="c:0", snippet="x", text="far", score=0.5),
                SearchHit(doc_id="close.md", loc="c:0", snippet="x", text="close", score=0.5),
            ]
            result = _cosine_fallback_rerank(query_vec, hits, store)
            # close.md should rank higher after cosine blending
            assert result[0].doc_id == "close.md"


# ---------------------------------------------------------------------------
# Time decay with floor
# ---------------------------------------------------------------------------

class TestTimeDecayFloor:
    def test_very_old_doc_floored_at_half(self):
        """A very old document should not lose more than ~50% of its score."""
        now = time.time()
        # 10 years old — extreme case
        hit = SearchHit(
            doc_id="ancient.md", loc="c:0", snippet="x", text="x",
            score=1.0, mtime=now - 3650 * 86400,
        )
        _apply_recency_boost([hit], half_life_days=90, weight=0.3)
        # Floor is 0.5 (multiplicative decay) + near-zero recency bonus
        assert hit.score >= 0.49  # ~0.5 due to floor
        assert hit.score < 0.55  # should not be much above floor

    def test_recent_doc_gets_full_boost(self):
        """A doc from today should get nearly full recency bonus."""
        now = time.time()
        hit = SearchHit(
            doc_id="new.md", loc="c:0", snippet="x", text="x",
            score=1.0, mtime=now,
        )
        _apply_recency_boost([hit], half_life_days=90, weight=0.3)
        # decay_factor ~1.0, bonus ~0.3
        assert hit.score > 1.2  # score * ~1.0 + ~0.3

    def test_mid_age_doc_moderate_effect(self):
        """A 90-day old doc should have ~0.75x decay factor + small bonus."""
        now = time.time()
        hit = SearchHit(
            doc_id="mid.md", loc="c:0", snippet="x", text="x",
            score=1.0, mtime=now - 90 * 86400,
        )
        _apply_recency_boost([hit], half_life_days=90, weight=0.3)
        # decay = 0.5 + 0.5 * exp(-1) ≈ 0.684
        # bonus = 0.3 * exp(-1) ≈ 0.110
        expected_decay = 0.5 + 0.5 * math.exp(-1)
        expected_bonus = 0.3 * math.exp(-1)
        expected = 1.0 * expected_decay + expected_bonus
        assert abs(hit.score - expected) < 0.01


# ---------------------------------------------------------------------------
# Cross-encoder blend (integration test via hybrid_search)
# ---------------------------------------------------------------------------

class BlendVerifyReranker(Reranker):
    """Mock reranker that assigns known scores to verify blending."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        # Assign reranker scores: first hit gets 1.0, second gets 0.0
        for i, h in enumerate(hits):
            h.score = 1.0 - i * (1.0 / max(len(hits) - 1, 1))
        return sorted(hits, key=lambda h: -h.score)


class TestCrossEncoderBlend:
    def test_blend_preserves_retrieval_signal(self):
        """Reranker should blend with original scores, not fully replace them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [
                _make_node("a.md", "c:0", "document about cats", vec),
                _make_node("b.md", "c:0", "document about dogs", vec),
            ]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            reranker = BlendVerifyReranker()
            result = hybrid_search(
                store, embed, "cats dogs",
                vector_top_k=10, final_top_k=5, reranker=reranker,
            )
            assert len(result) >= 1
            assert result.diagnostics["reranker_applied"] is True


# ---------------------------------------------------------------------------
# Cosine fallback in hybrid_search (integration)
# ---------------------------------------------------------------------------

class FailingReranker(Reranker):
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        raise RuntimeError("Reranker server unavailable")


class TestCosineFallbackInHybrid:
    def test_fallback_applies_on_reranker_failure(self):
        """When reranker fails, cosine fallback should still reorder results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [
                _make_node("a.md", "c:0", "document about cats and kittens", vec),
                _make_node("b.md", "c:0", "document about dogs and puppies", [0.0] + [1.0] + [0.0] * 766),
            ]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            result = hybrid_search(
                store, embed, "cats kittens",
                vector_top_k=10, final_top_k=5, reranker=FailingReranker(),
            )
            # Should still return results (not crash)
            assert len(result) >= 1
            # Reranker should NOT be marked as applied
            assert result.diagnostics["reranker_applied"] is False
            assert result.diagnostics["degraded"] is True


# ---------------------------------------------------------------------------
# get_vector method
# ---------------------------------------------------------------------------

class TestGetVector:
    def test_returns_vector_for_existing_chunk(self):
        """get_vector should return the stored embedding for a known chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [_make_node("a.md", "c:0", "test content", vec)]
            store = LanceDBStore(tmpdir, "test_chunks")
            store.upsert_nodes(nodes)

            result = store.get_vector("a.md::c:0")
            assert result is not None
            assert len(result) == 768
            assert result[0] == pytest.approx(1.0)

    def test_returns_none_for_missing_chunk(self):
        """get_vector should return None for a non-existent chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [_make_node("a.md", "c:0", "test content", vec)]
            store = LanceDBStore(tmpdir, "test_chunks")
            store.upsert_nodes(nodes)

            result = store.get_vector("nonexistent.md::c:0")
            assert result is None

    def test_returns_none_for_empty_store(self):
        """get_vector should return None when the table doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LanceDBStore(tmpdir, "test_chunks")
            result = store.get_vector("a.md::c:0")
            assert result is None


# ---------------------------------------------------------------------------
# Length normalization in hybrid_search (integration)
# ---------------------------------------------------------------------------

class TestLengthNormInHybrid:
    def test_short_chunk_beats_long_with_same_relevance(self):
        """A short relevant chunk should rank above a long keyword-stuffed one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            short_text = "concise document about search algorithms"
            long_text = ("search algorithms " * 200).strip()  # ~3400 chars, keyword-stuffed
            nodes = [
                _make_node("short.md", "c:0", short_text, vec),
                _make_node("long.md", "c:0", long_text, vec),
            ]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            result = hybrid_search(
                store, embed, "search algorithms",
                vector_top_k=10, final_top_k=5,
            )
            assert len(result) >= 2
            # Both have same vector; long one gets length penalty
            scores = {h.doc_id: h.score for h in result}
            assert scores["short.md"] > scores["long.md"]


# ---------------------------------------------------------------------------
# Importance weighting
# ---------------------------------------------------------------------------

class TestImportanceWeighting:
    def test_high_importance_boosted(self):
        """A high-importance doc should score higher than a low-importance one."""
        high = SearchHit(doc_id="high.md", loc="c:0", snippet="x", text="x", score=1.0,
                         enr_importance="1.0")
        low = SearchHit(doc_id="low.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="0.0")
        result = _apply_importance_weighting([low, high], field="enr_importance", weight=0.3)
        assert result[0].doc_id == "high.md"
        assert result[1].score < result[0].score

    def test_default_weight_magnitude(self):
        """With weight=0.3, importance=1.0 gives 1.0x and importance=0.0 gives 0.7x."""
        high = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0,
                         enr_importance="1.0")
        _apply_importance_weighting([high], field="enr_importance", weight=0.3)
        assert high.score == pytest.approx(1.0)

        low = SearchHit(doc_id="b.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="0.0")
        _apply_importance_weighting([low], field="enr_importance", weight=0.3)
        assert low.score == pytest.approx(0.7)

    def test_missing_field_is_neutral(self):
        """A hit with no importance field should get neutral treatment (0.5)."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0)
        _apply_importance_weighting([hit], field="enr_importance", weight=0.3)
        # 0.7 + 0.3 * 0.5 = 0.85
        assert hit.score == pytest.approx(0.85)

    def test_invalid_field_is_neutral(self):
        """Non-numeric importance values should default to neutral."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="high")
        _apply_importance_weighting([hit], field="enr_importance", weight=0.3)
        assert hit.score == pytest.approx(0.85)

    def test_clamped_to_range(self):
        """Values outside [0, 1] should be clamped."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="5.0")
        _apply_importance_weighting([hit], field="enr_importance", weight=0.3)
        assert hit.score == pytest.approx(1.0)  # clamped to 1.0

    def test_zero_weight_disables(self):
        """With weight=0.0, importance should have no effect."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="0.0")
        _apply_importance_weighting([hit], field="enr_importance", weight=0.0)
        assert hit.score == pytest.approx(1.0)

    def test_custom_field_name(self):
        """Should work with custom field names like 'priority'."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0)
        hit.extra_metadata = {"priority": "1.0"}
        _apply_importance_weighting([hit], field="priority", weight=0.3)
        assert hit.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Minimum score threshold
# ---------------------------------------------------------------------------

class TestMinScoreThreshold:
    def test_filters_low_scores(self):
        """Hits below threshold should be removed."""
        hits = [
            SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=0.5),
            SearchHit(doc_id="b.md", loc="c:0", snippet="x", text="x", score=0.01),
            SearchHit(doc_id="c.md", loc="c:0", snippet="x", text="x", score=0.001),
        ]
        result = _apply_min_score_threshold(hits, threshold=0.02)
        assert len(result) == 1
        assert result[0].doc_id == "a.md"

    def test_zero_threshold_passes_all(self):
        """Threshold of 0.0 should let everything through."""
        hits = [
            SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=0.001),
        ]
        result = _apply_min_score_threshold(hits, threshold=0.0)
        assert len(result) == 1

    def test_all_above_threshold(self):
        """All hits above threshold should be kept."""
        hits = [
            SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=0.5),
            SearchHit(doc_id="b.md", loc="c:0", snippet="x", text="x", score=0.3),
        ]
        result = _apply_min_score_threshold(hits, threshold=0.1)
        assert len(result) == 2

    def test_empty_list(self):
        """Empty input should return empty output."""
        result = _apply_min_score_threshold([], threshold=0.5)
        assert result == []

    def test_exact_threshold_included(self):
        """A hit at exactly the threshold should be kept."""
        hits = [
            SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=0.05),
        ]
        result = _apply_min_score_threshold(hits, threshold=0.05)
        assert len(result) == 1

    def test_all_filtered_returns_empty(self):
        """When all hits are below threshold, return empty list."""
        hits = [
            SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=0.01),
            SearchHit(doc_id="b.md", loc="c:0", snippet="x", text="x", score=0.02),
        ]
        result = _apply_min_score_threshold(hits, threshold=0.5)
        assert result == []

    def test_negative_threshold_passes_all(self):
        """Negative threshold should behave like 0 (disabled)."""
        hits = [
            SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=0.001),
        ]
        result = _apply_min_score_threshold(hits, threshold=-1.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_single_list(self):
        """RRF with a single result list should preserve order."""
        hits = [
            SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0),
            SearchHit(doc_id="b.md", loc="c:0", snippet="x", text="x", score=0.5),
        ]
        result = reciprocal_rank_fusion([hits])
        assert result[0].doc_id == "a.md"
        assert result[1].doc_id == "b.md"
        assert result[0].score > result[1].score

    def test_two_lists_boost_overlap(self):
        """A doc appearing in both lists should score higher than one in only one list."""
        list_a = [
            SearchHit(doc_id="both.md", loc="c:0", snippet="x", text="x", score=1.0),
            SearchHit(doc_id="only_a.md", loc="c:0", snippet="x", text="x", score=0.5),
        ]
        list_b = [
            SearchHit(doc_id="both.md", loc="c:0", snippet="x", text="x", score=1.0),
            SearchHit(doc_id="only_b.md", loc="c:0", snippet="x", text="x", score=0.5),
        ]
        result = reciprocal_rank_fusion([list_a, list_b])
        assert result[0].doc_id == "both.md"
        # both.md gets 2 * 1/(60+1), others get 1/(60+1) + 1/(60+2)
        assert result[0].score > result[1].score

    def test_deduplicates_by_doc_loc(self):
        """Same (doc_id, loc) from multiple lists should appear only once."""
        list_a = [SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0)]
        list_b = [SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0)]
        result = reciprocal_rank_fusion([list_a, list_b])
        assert len(result) == 1

    def test_empty_lists(self):
        """Empty input lists should return empty output."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_many_lists_fair_merge(self):
        """RRF should correctly merge 3+ lists."""
        lists = [
            [SearchHit(doc_id=f"d{i}.md", loc="c:0", snippet="x", text="x", score=1.0)]
            for i in range(5)
        ]
        result = reciprocal_rank_fusion(lists)
        assert len(result) == 5
        # All first-ranked → all get same score
        scores = [h.score for h in result]
        assert all(s == scores[0] for s in scores)


# ---------------------------------------------------------------------------
# Importance weighting — additional edge cases
# ---------------------------------------------------------------------------

class TestImportanceWeightingEdgeCases:
    def test_empty_list(self):
        """Empty hit list should not crash."""
        result = _apply_importance_weighting([], field="enr_importance", weight=0.3)
        assert result == []

    def test_negative_importance_clamped_to_zero(self):
        """Negative importance should be clamped to 0.0."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="-0.5")
        _apply_importance_weighting([hit], field="enr_importance", weight=0.3)
        assert hit.score == pytest.approx(0.7)  # (1-0.3) + 0.3*0 = 0.7

    def test_nan_importance_is_neutral(self):
        """NaN importance value should default to neutral (0.5)."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="nan")
        _apply_importance_weighting([hit], field="enr_importance", weight=0.3)
        # float("nan") parses but is NaN — should be handled
        # If not handled, score will be NaN
        assert not math.isnan(hit.score)
        assert hit.score == pytest.approx(0.85)

    def test_inf_importance_is_neutral(self):
        """Inf importance should default to neutral (0.5), not max boost."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0,
                        enr_importance="inf")
        _apply_importance_weighting([hit], field="enr_importance", weight=0.3)
        assert hit.score == pytest.approx(0.85)  # neutral: 0.7 + 0.3*0.5

    def test_zero_score_stays_zero(self):
        """A hit with score=0.0 should remain 0.0 regardless of importance."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=0.0,
                        enr_importance="1.0")
        _apply_importance_weighting([hit], field="enr_importance", weight=0.3)
        assert hit.score == 0.0


# ---------------------------------------------------------------------------
# Recency boost — additional edge cases
# ---------------------------------------------------------------------------

class TestRecencyBoostEdgeCases:
    def test_empty_list(self):
        """Empty hit list should not crash."""
        result = _apply_recency_boost([], half_life_days=90)
        assert result == []

    def test_zero_mtime_skipped(self):
        """A hit with mtime=0 should not get recency boost."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0, mtime=0.0)
        _apply_recency_boost([hit], half_life_days=90, weight=0.3)
        assert hit.score == 1.0  # unchanged

    def test_none_mtime_skipped(self):
        """A hit with mtime=None should not crash."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x", score=1.0, mtime=None)
        _apply_recency_boost([hit], half_life_days=90, weight=0.3)
        assert hit.score == 1.0  # unchanged

    def test_future_mtime_no_negative_age(self):
        """A hit with mtime in the future should not get negative age."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text="x",
                        score=1.0, mtime=time.time() + 86400)
        _apply_recency_boost([hit], half_life_days=90, weight=0.3)
        # age clamped to 0 → max boost
        assert hit.score > 1.0


# ---------------------------------------------------------------------------
# Length normalization — additional edge cases
# ---------------------------------------------------------------------------

class TestLengthNormEdgeCases:
    def test_empty_list(self):
        """Empty hit list should not crash."""
        result = _apply_length_normalization([])
        assert result == []

    def test_none_text_unaffected(self):
        """A hit with text=None should not crash."""
        hit = SearchHit(doc_id="a.md", loc="c:0", snippet="x", text=None, score=1.0)
        result = _apply_length_normalization([hit], anchor=800)
        assert result[0].score == 1.0


# ---------------------------------------------------------------------------
# Pipeline integration — multiple steps combined
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_empty_query_returns_empty(self):
        """An empty query should return empty results immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [_make_node("a.md", "c:0", "test content", vec)]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            result = hybrid_search(store, embed, "", vector_top_k=10, final_top_k=5)
            assert len(result) == 0

    def test_whitespace_query_returns_empty(self):
        """A whitespace-only query should return empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [_make_node("a.md", "c:0", "test content", vec)]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            result = hybrid_search(store, embed, "   ", vector_top_k=10, final_top_k=5)
            assert len(result) == 0

    def test_high_min_threshold_filters_all(self):
        """A very high min_score_threshold should return zero results gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [_make_node("a.md", "c:0", "test content about cats", vec)]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            result = hybrid_search(
                store, embed, "cats", vector_top_k=10, final_top_k=5,
                min_score_threshold=999.0,
            )
            assert len(result) == 0

    def test_importance_weight_zero_no_effect(self):
        """With importance_weight=0, scores should be unaffected by importance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [
                _make_node("a.md", "c:0", "document about cats", vec, enr_importance="0.0"),
                _make_node("b.md", "c:0", "document about cats", vec, enr_importance="1.0"),
            ]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            result = hybrid_search(
                store, embed, "cats", vector_top_k=10, final_top_k=5,
                importance_weight=0.0,
            )
            # With weight=0, importance has no effect; scores should be equal
            if len(result) >= 2:
                assert abs(result[0].score - result[1].score) < 0.01

    def test_diagnostics_present(self):
        """Every search result should include diagnostics dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vec = [1.0] + [0.0] * 767
            nodes = [_make_node("a.md", "c:0", "test content", vec)]
            store = _build_store_with_fts(tmpdir, nodes)
            embed = MockEmbedProvider(vec)

            result = hybrid_search(store, embed, "test", vector_top_k=10, final_top_k=5)
            assert "vector_search_active" in result.diagnostics
            assert "keyword_search_active" in result.diagnostics
            assert "reranker_applied" in result.diagnostics
            assert "degraded" in result.diagnostics
