"""Hybrid search: parallel vector + keyword (BM25/FTS), fuse with RRF, optional re-rank.

Architecture:
  1. Embed query → vector search (semantic similarity)
  2. Keyword search (BM25/FTS via tantivy) — runs in parallel with step 1
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. Length normalization (prevents long chunks from dominating)
  5. Importance weighting (boosts high-priority documents)
  6. Optional recency boost + time decay with floor
  7. Optional cross-encoder re-rank with cosine fallback on failure
  8. MMR diversity (removes near-duplicate chunks)
  9. Minimum score threshold (discards low-relevance noise)
  10. Final top_k

RRF reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).

Enhancements inspired by memory-lancedb-pro (win4r/memory-lancedb-pro):
  - Length normalization: log-based penalty for overly long chunks
  - Importance weighting: score *= (1 - w + w * importance) based on metadata field
  - MMR diversity: cosine similarity deduplication (threshold 0.85)
  - Cross-encoder blend: 60/40 reranker/original score preservation
  - Cosine fallback: lightweight rerank when cross-encoder fails
  - Time decay floor: old docs never lose more than 50% relevance
  - Minimum score threshold: filters noise below configurable cutoff
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from core.storage import SearchHit

if TYPE_CHECKING:
    from lancedb_store import LanceDBStore
    from providers.embed.base import EmbedProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search result with diagnostics
# ---------------------------------------------------------------------------

class SearchResult:
    """Hybrid search result — behaves like a list of SearchHit, plus diagnostics.

    diagnostics dict:
        keyword_search_active: bool — True if BM25/FTS returned results successfully.
        reranker_applied: bool — True if the cross-encoder reranker ran successfully.
        degraded: bool — True if any retrieval stage failed silently.
    """

    __slots__ = ("hits", "diagnostics")

    def __init__(
        self,
        hits: list[SearchHit],
        diagnostics: dict | None = None,
    ) -> None:
        self.hits = hits
        self.diagnostics = diagnostics or {
            "vector_search_active": True,
            "keyword_search_active": True,
            "reranker_applied": False,
            "degraded": False,
        }

    # List-compatible interface so existing callers (for h in result, result[0], etc.) still work.
    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, idx):
        return self.hits[idx]

    def __iter__(self):
        return iter(self.hits)

    def __bool__(self) -> bool:
        return bool(self.hits)

    def __eq__(self, other):
        if isinstance(other, list):
            return self.hits == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"SearchResult(hits={len(self.hits)}, diagnostics={self.diagnostics})"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]],
    k: int = 60,
) -> list[SearchHit]:
    """Fuse multiple ranked result lists using RRF.

    score(doc) = Σ  1 / (k + rank_i)   for each list i where doc appears.
    k=60 is near-optimal per Cormack et al.

    Deduplicates by (doc_id, loc) — keeps the first occurrence's metadata.
    """
    scores: dict[tuple[str, str], float] = {}
    hits_map: dict[tuple[str, str], SearchHit] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            key = (hit.doc_id, hit.loc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in hits_map:
                hits_map[key] = hit

    sorted_keys = sorted(scores.keys(), key=lambda key: -scores[key])

    fused: list[SearchHit] = []
    for key in sorted_keys:
        hit = hits_map[key]
        hit.score = scores[key]
        fused.append(hit)

    return fused


# ---------------------------------------------------------------------------
# Length normalization
# ---------------------------------------------------------------------------

def _apply_length_normalization(
    hits: list[SearchHit],
    anchor: int = 800,
) -> list[SearchHit]:
    """Penalize long chunks that score high due to keyword density.

    Formula: score *= 1 / (1 + 0.5 * log2(len / anchor))
    Chunks at or below `anchor` chars are unaffected (factor >= 1.0 capped to 1.0).
    Very long chunks (e.g., 3200 chars at anchor=800) get ~0.67x.

    Inspired by memory-lancedb-pro's length normalization approach.
    """
    for hit in hits:
        text_len = len(hit.text) if hit.text else 0
        if text_len > anchor:
            factor = 1.0 / (1.0 + 0.5 * math.log2(text_len / anchor))
            hit.score *= factor
    hits.sort(key=lambda h: -h.score)
    return hits


# ---------------------------------------------------------------------------
# Importance weighting
# ---------------------------------------------------------------------------

def _apply_importance_weighting(
    hits: list[SearchHit],
    field: str = "enr_importance",
    weight: float = 0.3,
) -> list[SearchHit]:
    """Boost documents with higher importance/priority metadata.

    Reads a numeric metadata field (0.0–1.0) from each hit and scales the score:
        score *= (1 - weight + weight * importance)

    With default weight=0.3:
        importance=1.0 → score *= 1.0 (no change)
        importance=0.5 → score *= 0.85
        importance=0.0 → score *= 0.7

    The field is looked up on the hit object first (e.g., frontmatter-promoted
    columns like ``priority``), then in ``extra_metadata``. If the field is
    missing or non-numeric, importance defaults to 0.5 (neutral).

    Inspired by memory-lancedb-pro's importance weighting.
    """
    for hit in hits:
        # Try direct attribute, then extra_metadata
        raw = getattr(hit, field, None)
        if not raw:  # None or empty string
            raw = (hit.extra_metadata or {}).get(field)

        # Parse to float, default to 0.5 (neutral) if missing/invalid
        try:
            importance = float(raw) if raw else 0.5
        except (TypeError, ValueError):
            importance = 0.5

        # Clamp to [0, 1]
        importance = max(0.0, min(1.0, importance))

        factor = (1.0 - weight) + weight * importance
        hit.score *= factor

    hits.sort(key=lambda h: -h.score)
    return hits


# ---------------------------------------------------------------------------
# Minimum score threshold
# ---------------------------------------------------------------------------

def _apply_min_score_threshold(
    hits: list[SearchHit],
    threshold: float = 0.0,
) -> list[SearchHit]:
    """Discard results below a minimum score threshold.

    Set threshold=0.0 (default) to disable — all results pass through.
    Typical production values: 0.01–0.05 for RRF scores.

    Inspired by memory-lancedb-pro's noise filtering.
    """
    if threshold <= 0.0:
        return hits
    return [h for h in hits if h.score >= threshold]


# ---------------------------------------------------------------------------
# Reranker protocol (optional)
# ---------------------------------------------------------------------------

class Reranker:
    """Base class for re-rankers. Subclass and override `rerank`."""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Re-score and re-sort hits. Return sorted list (best first)."""
        raise NotImplementedError


class DeepInfraReranker(Reranker):
    """Re-rank using Qwen3-Reranker-8B via DeepInfra inference API.

    Always-on (no cold start), batch scoring (one API call), direct relevance scores.

    Uses 60/40 blending: final_score = 0.6 * reranker_score + 0.4 * original_score.
    This preserves retrieval-stage signal and prevents the cross-encoder from
    completely overriding the fusion stage.
    """

    BLEND_RERANKER = 0.6
    BLEND_ORIGINAL = 0.4

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-8B",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        self.timeout = timeout
        self.base_url = f"https://api.deepinfra.com/v1/inference/{model}"

        if not self.api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY not set. Set it in .env or pass api_key."
            )

        logger.info("DeepInfraReranker initialized: model=%s", model)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Score all candidates in one batch call, blend with original scores, return sorted."""
        if not hits:
            return hits

        import httpx

        # Preserve original scores for blending
        original_scores = {(h.doc_id, h.loc): h.score for h in hits}

        documents = [h.text[:4000] for h in hits]

        try:
            resp = httpx.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "queries": [query],
                    "documents": documents,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"DeepInfra reranker failed: {e}") from e

        # Normalize reranker scores to [0, 1] for blending
        raw_scores = data.get("scores", [])
        if raw_scores:
            max_score = max(raw_scores)
            min_score = min(raw_scores)
            score_range = max_score - min_score
            if score_range > 0:
                norm_scores = [(s - min_score) / score_range for s in raw_scores]
            else:
                norm_scores = [1.0] * len(raw_scores)
        else:
            norm_scores = []

        # Normalize original scores to [0, 1] for blending
        orig_vals = [original_scores.get((h.doc_id, h.loc), 0.0) for h in hits]
        if orig_vals:
            max_orig = max(orig_vals)
            min_orig = min(orig_vals)
            orig_range = max_orig - min_orig
            if orig_range > 0:
                norm_orig = [(v - min_orig) / orig_range for v in orig_vals]
            else:
                norm_orig = [1.0] * len(orig_vals)
        else:
            norm_orig = []

        # Blend: 60% reranker + 40% original
        for i in range(len(hits)):
            reranker_s = norm_scores[i] if i < len(norm_scores) else 0.0
            original_s = norm_orig[i] if i < len(norm_orig) else 0.0
            hits[i].score = self.BLEND_RERANKER * reranker_s + self.BLEND_ORIGINAL * original_s

        ranked = sorted(hits, key=lambda h: -h.score)
        logger.info(
            "DeepInfraReranker: scored %d candidates (60/40 blend), top_score=%.4f",
            len(ranked),
            ranked[0].score if ranked else 0.0,
        )
        return ranked


def build_reranker(config: dict) -> Reranker | None:
    """Factory: build a Reranker from the search.reranker config section.

    Config example:
        search:
          reranker:
            enabled: true
            provider: "deepinfra"
            model: "Qwen/Qwen3-Reranker-8B"
            timeout: 30
    """
    reranker_cfg = config.get("search", {}).get("reranker", {})
    if not reranker_cfg.get("enabled", False):
        return None

    provider = reranker_cfg.get("provider", "deepinfra")

    if provider == "deepinfra":
        return DeepInfraReranker(
            model=reranker_cfg.get("model", "Qwen/Qwen3-Reranker-8B"),
            api_key=reranker_cfg.get("api_key"),
            timeout=reranker_cfg.get("timeout", 30.0),
        )
    else:
        logger.warning("Unknown reranker provider: %s", provider)
        return None


# ---------------------------------------------------------------------------
# Cosine similarity fallback reranker
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_fallback_rerank(
    query_vector: list[float],
    hits: list[SearchHit],
    store: "LanceDBStore",
    blend_cosine: float = 0.3,
    blend_original: float = 0.7,
) -> list[SearchHit]:
    """Lightweight rerank using cosine similarity when cross-encoder fails.

    Retrieves stored embeddings and computes cosine similarity with the query
    vector. Blends 70% original score + 30% cosine similarity.

    Falls back silently to original order if embeddings can't be retrieved.
    """
    try:
        for hit in hits:
            chunk_uid = f"{hit.doc_id}::{hit.loc}"
            stored_vec = store.get_vector(chunk_uid)
            if stored_vec is not None:
                cos_sim = _cosine_similarity(query_vector, stored_vec)
                hit.score = blend_original * hit.score + blend_cosine * cos_sim
        hits.sort(key=lambda h: -h.score)
        logger.info("Cosine fallback rerank applied to %d hits", len(hits))
    except Exception as e:
        logger.warning("Cosine fallback rerank failed: %s", e)
    return hits


# ---------------------------------------------------------------------------
# Recency boost + time decay
# ---------------------------------------------------------------------------

def _apply_recency_boost(
    hits: list[SearchHit],
    half_life_days: float = 90.0,
    weight: float = 0.3,
) -> list[SearchHit]:
    """Apply recency boost (additive) and time decay (multiplicative with floor).

    Two mechanisms (inspired by memory-lancedb-pro):

    1. Recency boost (additive): small bonus for recent content.
       boost = weight * exp(-age_days / half_life)
       Default: up to +0.3 for very recent docs, decaying to ~0 over months.

    2. Time decay (multiplicative, floor at 0.5): old content gradually loses
       relevance but never more than 50%.
       factor = 0.5 + 0.5 * exp(-age_days / decay_half_life)
       decay_half_life = half_life_days (same parameter, different mechanism).

    Combined: score = score * time_decay_factor + recency_boost
    """
    now = time.time()
    for hit in hits:
        if hit.mtime and hit.mtime > 0:
            age_days = max((now - hit.mtime) / 86400.0, 0.0)

            # Time decay (multiplicative, floor at 0.5)
            decay_factor = 0.5 + 0.5 * math.exp(-age_days / half_life_days)
            hit.score *= decay_factor

            # Recency boost (additive)
            recency_bonus = weight * math.exp(-age_days / half_life_days)
            hit.score += recency_bonus

    hits.sort(key=lambda h: -h.score)
    return hits


# ---------------------------------------------------------------------------
# MMR diversity
# ---------------------------------------------------------------------------

def _apply_mmr_diversity(
    hits: list[SearchHit],
    store: "LanceDBStore",
    similarity_threshold: float = 0.85,
) -> list[SearchHit]:
    """Remove near-duplicate chunks using Maximal Marginal Relevance.

    Greedy selection: for each candidate (in score order), compute cosine
    similarity against all already-selected results. If similarity > threshold
    to any selected result, defer it (append at end). This prevents
    near-duplicate chunks from consuming all top-K slots.

    Falls back silently to original order if vectors can't be retrieved.

    Inspired by memory-lancedb-pro's MMR diversity approach.
    """
    if len(hits) <= 1:
        return hits

    try:
        # Retrieve vectors for all hits
        vectors: dict[str, list[float]] = {}
        for hit in hits:
            chunk_uid = f"{hit.doc_id}::{hit.loc}"
            vec = store.get_vector(chunk_uid)
            if vec is not None:
                vectors[chunk_uid] = vec

        if not vectors:
            return hits

        selected: list[SearchHit] = []
        deferred: list[SearchHit] = []

        for hit in hits:
            chunk_uid = f"{hit.doc_id}::{hit.loc}"
            hit_vec = vectors.get(chunk_uid)

            if hit_vec is None:
                selected.append(hit)
                continue

            too_similar = False
            for sel_hit in selected:
                sel_uid = f"{sel_hit.doc_id}::{sel_hit.loc}"
                sel_vec = vectors.get(sel_uid)
                if sel_vec is not None:
                    sim = _cosine_similarity(hit_vec, sel_vec)
                    if sim > similarity_threshold:
                        too_similar = True
                        break

            if too_similar:
                deferred.append(hit)
            else:
                selected.append(hit)

        result = selected + deferred
        if deferred:
            logger.info(
                "MMR diversity: %d selected, %d deferred (threshold=%.2f)",
                len(selected), len(deferred), similarity_threshold,
            )
        return result

    except Exception as e:
        logger.warning("MMR diversity failed (using original order): %s", e)
        return hits


# ---------------------------------------------------------------------------
# Main hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    store: "LanceDBStore",
    embed_provider: "EmbedProvider",
    query: str,
    vector_top_k: int = 50,
    keyword_top_k: int = 50,
    final_top_k: int = 10,
    rrf_k: int = 60,
    doc_id_prefix: str | None = None,
    source_type: str | None = None,
    tags: str | None = None,
    status: str | None = None,
    folder: str | None = None,
    reranker: Reranker | None = None,
    prefer_recent: bool = False,
    recency_half_life_days: float = 90.0,
    recency_weight: float = 0.3,
    metadata_filters: dict[str, str] | None = None,
    enr_doc_type: str | None = None,
    enr_topics: str | None = None,
    importance_field: str = "enr_importance",
    importance_weight: float = 0.3,
    min_score_threshold: float = 0.0,
) -> SearchResult:
    """Run vector + keyword search in parallel, fuse with RRF, optionally re-rank.

    Pipeline:
      1. Pre-filter (SQL WHERE on LanceDB)
      2. Parallel vector + keyword retrieval
      3. RRF fusion
      4. Length normalization
      5. Importance weighting (boosts docs with high importance/priority metadata)
      6. Optional recency boost + time decay (with floor)
      7. Optional cross-encoder re-rank (60/40 blend, cosine fallback)
      8. MMR diversity (removes near-duplicate chunks)
      9. Minimum score threshold (discards noise)
      10. Final top_k

    Returns a SearchResult (list-compatible) with a diagnostics dict.
    """
    if not query.strip():
        return SearchResult(hits=[])

    diagnostics = {
        "vector_search_active": True,
        "keyword_search_active": True,
        "reranker_applied": False,
        "degraded": False,
    }

    # 0. Build WHERE clause for pre-filtering (applied inside LanceDB before scoring)
    where = store._build_where_clause(
        doc_id_prefix=doc_id_prefix,
        source_type=source_type,
        status=status,
        folder=folder,
        tags=tags,
        enr_doc_type=enr_doc_type,
        enr_topics=enr_topics,
        metadata_filters=metadata_filters,
    )

    # 1. Embed query
    query_vector = embed_provider.embed_query(query)

    # 2. Parallel retrieval: vector (semantic) + keyword (BM25/FTS)
    with ThreadPoolExecutor(max_workers=2) as executor:
        vec_future = executor.submit(store.vector_search, query_vector, vector_top_k, where)
        kw_future = executor.submit(store.keyword_search, query, keyword_top_k, where)
        try:
            vector_hits = vec_future.result()
        except Exception as e:
            logger.warning("Vector search failed (degraded to keyword-only): %s", e)
            vector_hits = []
            diagnostics["vector_search_active"] = False
        try:
            keyword_hits = kw_future.result()
        except Exception as e:
            logger.warning("Keyword/FTS search failed (degraded to vector-only): %s", e)
            keyword_hits = []
            diagnostics["keyword_search_active"] = False

    logger.info(
        "Retrieval: %d vector hits, %d keyword hits (fts_ok=%s, where=%s)",
        len(vector_hits),
        len(keyword_hits),
        diagnostics["keyword_search_active"],
        where,
    )

    # 3. Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=rrf_k)

    # 4. Length normalization (prevents long keyword-rich chunks from dominating)
    fused = _apply_length_normalization(fused)

    # 5. Importance weighting (boost high-priority docs via metadata field)
    fused = _apply_importance_weighting(fused, field=importance_field, weight=importance_weight)

    # 6. Optional recency boost + time decay (applied after fusion so boost is uniform)
    if prefer_recent:
        fused = _apply_recency_boost(fused, recency_half_life_days, recency_weight)

    # 7. Optional cross-encoder re-rank (on top N candidates for efficiency)
    #    On failure: cosine similarity fallback instead of giving up entirely.
    if reranker is not None:
        rerank_pool = fused[: final_top_k * 6]
        try:
            fused = reranker.rerank(query, rerank_pool)
            diagnostics["reranker_applied"] = True
        except Exception as e:
            logger.warning("Reranker failed, falling back to cosine rerank: %s", e)
            fused = _cosine_fallback_rerank(query_vector, rerank_pool, store)

    # 8. MMR diversity (remove near-duplicate chunks)
    fused = _apply_mmr_diversity(fused, store)

    # 9. Minimum score threshold (discard noise)
    fused = _apply_min_score_threshold(fused, threshold=min_score_threshold)

    # 10. Compute degraded flag
    diagnostics["degraded"] = (
        not diagnostics["vector_search_active"]
        or not diagnostics["keyword_search_active"]
        or (reranker is not None and not diagnostics["reranker_applied"])
    )

    # 11. Final top_k
    return SearchResult(hits=fused[:final_top_k], diagnostics=diagnostics)
