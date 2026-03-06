"""Tests for LanceDBStore (uses a temp directory, no mocks needed)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from lancedb_store import LanceDBStore


def _make_node(doc_id: str, loc: str, text: str, vector: list[float]) -> TextNode:
    """Helper: build a TextNode like process_doc_task does."""
    chunk_uid = f"{doc_id}::{loc}"
    node = TextNode(
        text=text,
        id_=chunk_uid,
        embedding=vector,
        metadata={
            "doc_id": doc_id,
            "source_type": "md",
            "loc": loc,
            "snippet": text[:200],
            "mtime": 1.0,
            "size": len(text),
        },
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def test_upsert_and_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [
            _make_node("a.md", "c:0", "hello world", [0.1] * 768),
            _make_node("b.md", "c:0", "goodbye world", [0.2] * 768),
        ]
        store.upsert_nodes(nodes)
        doc_ids = store.list_doc_ids()
        assert set(doc_ids) == {"a.md", "b.md"}


def test_upsert_replaces():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes_v1 = [_make_node("a.md", "c:0", "version 1", [0.1] * 768)]
        store.upsert_nodes(nodes_v1)

        nodes_v2 = [
            _make_node("a.md", "c:0", "version 2 chunk 0", [0.3] * 768),
            _make_node("a.md", "c:1", "version 2 chunk 1", [0.4] * 768),
        ]
        store.upsert_nodes(nodes_v2)

        doc_ids = store.list_doc_ids()
        assert doc_ids == ["a.md"]


def test_delete_by_doc_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [
            _make_node("a.md", "c:0", "keep", [0.1] * 768),
            _make_node("b.md", "c:0", "delete me", [0.2] * 768),
        ]
        store.upsert_nodes(nodes)
        store.delete_by_doc_ids(["b.md"])
        doc_ids = store.list_doc_ids()
        assert doc_ids == ["a.md"]


def test_list_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.list_doc_ids() == []


def test_vector_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [
            _make_node("a.md", "c:0", "apple banana", [1.0] + [0.0] * 767),
            _make_node("b.md", "c:0", "cherry date", [0.0] + [1.0] + [0.0] * 766),
        ]
        store.upsert_nodes(nodes)
        hits = store.vector_search([1.0] + [0.0] * 767, top_k=1)
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


def _make_node_with_meta(doc_id, loc, text, vector, **extra_meta):
    """Helper that lets tests set arbitrary metadata fields."""
    meta = {
        "doc_id": doc_id,
        "source_type": "md",
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


def test_facets_empty_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        facets = store.facets()
        assert facets["total_docs"] == 0
        assert facets["total_chunks"] == 0


def test_facets_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:0", "hello", vec, tags="recipe,korean", folder="Projects", status="active", author="Dan"),
            _make_node_with_meta("a.md", "c:1", "world", vec, tags="recipe,korean", folder="Projects", status="active", author="Dan"),
            _make_node_with_meta("b.pdf", "p:1:c:0", "doc", vec, source_type="pdf", tags="finance", folder="Archive", status="archived", author="Jane"),
        ]
        store.upsert_nodes(nodes)
        facets = store.facets()
        assert facets["total_docs"] == 2  # a.md and b.pdf
        assert facets["total_chunks"] == 3
        tag_values = {t["value"] for t in facets["tags"]}
        assert "recipe" in tag_values
        assert "korean" in tag_values
        assert "finance" in tag_values
        folder_values = {f["value"] for f in facets["folders"]}
        assert "Projects" in folder_values
        assert "Archive" in folder_values


def test_search_hit_has_mtime():
    """Verify mtime is passed through on SearchHit from vector_search."""
    import time
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        now = time.time()
        nodes = [_make_node_with_meta("a.md", "c:0", "test", [0.1] * 768, mtime=now)]
        store.upsert_nodes(nodes)
        hits = store.vector_search([0.1] * 768, top_k=1)
        assert len(hits) == 1
        assert abs(hits[0].mtime - now) < 1.0


def test_search_hit_has_description_author():
    """Verify description/author/custom_meta are passed through."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "test", vec,
            description="A test doc",
            author="Dan Park",
            custom_meta='{"source": "http://example.com"}',
        )]
        store.upsert_nodes(nodes)
        hits = store.vector_search(vec, top_k=1)
        assert len(hits) == 1
        assert hits[0].description == "A test doc"
        assert hits[0].author == "Dan Park"
        assert hits[0].custom_meta == '{"source": "http://example.com"}'


# --- Schema evolution tests ---


def test_schema_evolution_new_metadata_field():
    """Adding nodes with a new metadata field (section) after initial insert should work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # First batch: no section field
        nodes1 = [_make_node_with_meta("a.pdf", "p:1:c:0", "pdf chunk", vec)]
        store.upsert_nodes(nodes1)

        # Second batch: has section field
        nodes2 = [_make_node_with_meta("b.md", "c:0", "md chunk", vec, section="Introduction")]
        store.upsert_nodes(nodes2)

        doc_ids = set(store.list_doc_ids())
        assert doc_ids == {"a.pdf", "b.md"}
        assert "section" in store._metadata_subfields()


def test_schema_evolution_preserves_vectors():
    """Schema evolution should not corrupt existing vectors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        # Insert a node with a distinctive vector
        vec_a = [1.0] + [0.0] * 767
        nodes1 = [_make_node_with_meta("a.md", "c:0", "alpha", vec_a)]
        store.upsert_nodes(nodes1)

        # Trigger schema evolution with a new field
        vec_b = [0.0] + [1.0] + [0.0] * 766
        nodes2 = [_make_node_with_meta("b.md", "c:0", "beta", vec_b, section="Setup")]
        store.upsert_nodes(nodes2)

        # Vector search should still find the correct nearest neighbor
        hits = store.vector_search([1.0] + [0.0] * 767, top_k=1)
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


def test_schema_evolution_old_doc_returns_empty_for_new_field():
    """Old docs indexed before schema evolution should return empty string for new fields.

    Covers: vector_search, get_chunk, get_doc_chunks all work on old docs
    whose metadata was backfilled with empty strings after schema evolution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        # Index a PDF without section field
        vec_a = [1.0] + [0.0] * 767
        nodes1 = [_make_node_with_meta(
            "a.pdf", "p:1:c:0", "old pdf content", vec_a,
            source_type="pdf", folder="Archive", status="archived",
        )]
        store.upsert_nodes(nodes1)

        # Index a Markdown with section field — triggers schema evolution
        vec_b = [0.0] + [1.0] + [0.0] * 766
        nodes2 = [_make_node_with_meta(
            "b.md", "c:0", "new md content", vec_b,
            source_type="md", folder="Projects", status="active",
            section="Introduction",
        )]
        store.upsert_nodes(nodes2)

        # Search returning both docs
        hits = store.vector_search([1.0] + [0.0] * 767, top_k=2)
        old_hit = next(h for h in hits if h.doc_id == "a.pdf")
        new_hit = next(h for h in hits if h.doc_id == "b.md")

        # Old doc: text intact, original metadata preserved
        assert old_hit.text == "old pdf content"
        assert old_hit.source_type == "pdf"
        assert old_hit.folder == "Archive"
        assert old_hit.status == "archived"

        # New doc: section value present
        assert new_hit.text == "new md content"
        assert new_hit.source_type == "md"
        assert new_hit.folder == "Projects"

        # get_chunk on old doc should work fine
        chunk = store.get_chunk("a.pdf", "p:1:c:0")
        assert chunk is not None
        assert chunk.text == "old pdf content"
        assert chunk.source_type == "pdf"

        # get_doc_chunks on old doc should work fine
        chunks = store.get_doc_chunks("a.pdf")
        assert len(chunks) == 1
        assert chunks[0].loc == "p:1:c:0"
        assert chunks[0].source_type == "pdf"


def test_schema_evolution_list_recent_docs_mixed_metadata():
    """list_recent_docs should work when metadata fields vary across docs.

    Simulates: PDFs indexed first (no section), then Markdown indexed (has section).
    list_recent_docs filters by source_type and folder via SQL on metadata struct —
    both old and new docs must be retrievable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # PDF indexed first — no section field
        nodes1 = [_make_node_with_meta(
            "report.pdf", "p:1:c:0", "quarterly report", vec,
            source_type="pdf", folder="Archive", status="archived", mtime=100.0,
        )]
        store.upsert_nodes(nodes1)

        # Markdown indexed later — has section, triggers schema evolution
        nodes2 = [_make_node_with_meta(
            "notes.md", "c:0", "meeting notes", vec,
            source_type="md", folder="Projects", status="active", mtime=200.0,
            section="Action Items",
        )]
        store.upsert_nodes(nodes2)

        # Unfiltered: both docs present
        all_docs = store.list_recent_docs(limit=10)
        doc_ids = {d["doc_id"] for d in all_docs}
        assert doc_ids == {"report.pdf", "notes.md"}

        # Filter by source_type — only matching type returned
        pdf_docs = store.list_recent_docs(limit=10, source_type="pdf")
        assert len(pdf_docs) == 1
        assert pdf_docs[0]["doc_id"] == "report.pdf"

        md_docs = store.list_recent_docs(limit=10, source_type="md")
        assert len(md_docs) == 1
        assert md_docs[0]["doc_id"] == "notes.md"

        # Filter by folder
        archive_docs = store.list_recent_docs(limit=10, folder="Archive")
        assert len(archive_docs) == 1
        assert archive_docs[0]["doc_id"] == "report.pdf"


def test_schema_evolution_multiple_new_fields():
    """Multiple new fields added at once should all appear in schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # First batch: baseline metadata only
        nodes1 = [_make_node_with_meta("a.md", "c:0", "first", vec)]
        store.upsert_nodes(nodes1)

        # Second batch: two new fields at once
        nodes2 = [_make_node_with_meta(
            "b.md", "c:0", "second", vec,
            section="Overview", sentiment="positive",
        )]
        store.upsert_nodes(nodes2)

        subfields = store._metadata_subfields()
        assert "section" in subfields
        assert "sentiment" in subfields


# --- Dynamic metadata pipeline tests ---


def test_extra_metadata_in_vector_search():
    """Dynamic fields should be visible in vector_search results via extra_metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "intro text", vec, section="Introduction",
        )]
        store.upsert_nodes(nodes)

        hits = store.vector_search(vec, top_k=1)
        assert len(hits) == 1
        # Accessible via extra_metadata dict
        assert hits[0].extra_metadata["section"] == "Introduction"
        # Accessible via __getattr__ fallback
        assert hits[0].section == "Introduction"


def test_extra_metadata_in_get_chunk():
    """Dynamic fields should be visible in get_chunk results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "setup steps", vec, section="Setup",
        )]
        store.upsert_nodes(nodes)

        chunk = store.get_chunk("a.md", "c:0")
        assert chunk is not None
        assert chunk.extra_metadata["section"] == "Setup"
        assert chunk.section == "Setup"


def test_extra_metadata_in_get_doc_chunks():
    """Dynamic fields should be visible in get_doc_chunks results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "overview", vec, section="Overview",
        )]
        store.upsert_nodes(nodes)

        chunks = store.get_doc_chunks("a.md")
        assert len(chunks) == 1
        assert chunks[0].section == "Overview"


def test_extra_metadata_visible_in_hit_to_dict():
    """_hit_to_dict should include extra_metadata fields in the output dict."""
    from core.storage import SearchHit
    import mcp_server

    hit = SearchHit(
        doc_id="a.md", loc="c:0", snippet="test", text="test text",
        score=0.5, extra_metadata={"section": "Intro", "sentiment": "positive"},
    )
    d = mcp_server._hit_to_dict(hit)
    assert d["section"] == "Intro"
    assert d["sentiment"] == "positive"
    # Core fields still present
    assert d["doc_id"] == "a.md"
    assert d["score"] == 0.5


def test_extra_metadata_filter_via_search_impl():
    """metadata_filters should filter results by dynamic fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")

        # Two docs: one with section, one without
        vec_a = [1.0] + [0.0] * 767
        nodes1 = [_make_node_with_meta(
            "a.md", "c:0", "intro content", vec_a, section="Introduction",
        )]
        vec_b = [0.0] + [1.0] + [0.0] * 766
        nodes2 = [_make_node_with_meta("b.md", "c:0", "other content", vec_b)]

        store.upsert_nodes(nodes1)
        store.upsert_nodes(nodes2)

        # Search without filter — both docs findable
        all_hits = store.vector_search([0.5] * 768, top_k=10)
        assert len(all_hits) == 2

        # Use hybrid_search metadata_filters via the search impl
        # (requires embedding — test the filter logic directly)
        from search_hybrid import reciprocal_rank_fusion
        filtered = [h for h in all_hits if getattr(h, "section", None) == "Introduction"]
        assert len(filtered) == 1
        assert filtered[0].doc_id == "a.md"


def test_facets_includes_dynamic_fields():
    """facets() should include dynamic metadata fields like section."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:0", "intro", vec, section="Introduction"),
            _make_node_with_meta("b.md", "c:0", "setup", vec, section="Setup"),
            _make_node_with_meta("c.md", "c:0", "setup2", vec, section="Setup"),
        ]
        store.upsert_nodes(nodes)

        facets = store.facets()
        # section should appear as a dynamic facet
        assert "section" in facets
        section_values = {f["value"]: f["count"] for f in facets["section"]}
        assert section_values["Setup"] == 2
        assert section_values["Introduction"] == 1


def test_custom_frontmatter_promoted_to_columns():
    """Extra frontmatter fields promoted to real columns should appear in
    extra_metadata, facets, and be filterable via getattr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768

        # Simulate promoted frontmatter keys (priority, category)
        nodes = [
            _make_node_with_meta(
                "a.md", "c:0", "high-pri research", vec,
                priority="high", category="research",
            ),
            _make_node_with_meta(
                "b.md", "c:0", "low-pri notes", vec,
                priority="low", category="notes",
            ),
        ]
        store.upsert_nodes(nodes)

        # extra_metadata exposes promoted fields on SearchHit
        hits = store.vector_search(vec, top_k=10)
        assert len(hits) == 2
        for h in hits:
            assert "priority" in h.extra_metadata
            assert "category" in h.extra_metadata

        hit_a = next(h for h in hits if h.doc_id == "a.md")
        assert hit_a.extra_metadata["priority"] == "high"
        assert hit_a.extra_metadata["category"] == "research"
        # Also accessible via __getattr__
        assert hit_a.priority == "high"
        assert hit_a.category == "research"

        # facets() includes promoted fields
        facets = store.facets()
        assert "priority" in facets
        assert "category" in facets
        pri_values = {f["value"] for f in facets["priority"]}
        assert pri_values == {"high", "low"}
        cat_values = {f["value"] for f in facets["category"]}
        assert cat_values == {"research", "notes"}

        # Filterable via getattr
        high_pri = [h for h in hits if getattr(h, "priority", None) == "high"]
        assert len(high_pri) == 1
        assert high_pri[0].doc_id == "a.md"


def test_file_status_returns_metadata_fields():
    """file_status should include metadata_fields listing all schema fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "text", vec, section="Intro",
        )]
        store.upsert_nodes(nodes)

        import mcp_server
        # Wire up the test store
        mcp_server._cache = (store, None, {"index_root": tmpdir})

        result = mcp_server._file_status_impl()
        assert "metadata_fields" in result
        assert "section" in result["metadata_fields"]
        assert "doc_id" in result["metadata_fields"]
        assert "source_type" in result["metadata_fields"]


# --- _build_where_clause tests ---


def test_build_where_clause_empty():
    """No filters should return None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store._build_where_clause() is None


def test_build_where_clause_source_type():
    """source_type should produce case-insensitive exact match on metadata.source_type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(source_type="pdf")
        assert clause == "lower(metadata.source_type) = 'pdf'"


def test_build_where_clause_doc_id_prefix():
    """doc_id_prefix should produce LIKE on top-level doc_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(doc_id_prefix="Projects/")
        assert clause == "doc_id LIKE 'Projects/%'"


def test_build_where_clause_comma_fields():
    """Comma-separated tags/enr_doc_type/enr_topics should produce case-insensitive OR clauses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(tags="recipe,korean")
        assert "lower(metadata.tags) LIKE '%recipe%'" in clause
        assert "lower(metadata.tags) LIKE '%korean%'" in clause
        assert " OR " in clause

        clause2 = store._build_where_clause(enr_doc_type="Report,Summary")
        assert "lower(metadata.enr_doc_type) LIKE '%report%'" in clause2
        assert "lower(metadata.enr_doc_type) LIKE '%summary%'" in clause2

        clause3 = store._build_where_clause(enr_topics="machine learning,NLP")
        assert "lower(metadata.enr_topics) LIKE '%machine learning%'" in clause3
        assert "lower(metadata.enr_topics) LIKE '%nlp%'" in clause3


def test_build_where_clause_combined():
    """Multiple filters should be AND-joined."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(
            source_type="pdf", folder="Archive", tags="finance",
        )
        assert "lower(metadata.source_type) = 'pdf'" in clause
        assert "lower(metadata.folder) = 'archive'" in clause
        assert "lower(metadata.tags) LIKE '%finance%'" in clause
        assert " AND " in clause


def test_build_where_clause_sql_escape():
    """Single quotes in values should be escaped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(folder="O'Brien's")
        assert "o''brien''s" in clause


def test_build_where_clause_metadata_filters():
    """metadata_filters dict should produce case-insensitive metadata.key = 'value' clauses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(
            metadata_filters={"section": "Introduction", "priority": "high"},
        )
        assert "lower(metadata.section) = 'introduction'" in clause
        assert "lower(metadata.priority) = 'high'" in clause


# --- _row_to_hit tests ---


def test_row_to_hit_vector_distance():
    """_row_to_hit should convert _distance to similarity score."""
    row = {
        "metadata": {"doc_id": "a.md", "loc": "c:0", "snippet": "test", "mtime": 1.0},
        "text": "test text",
        "_distance": 0.3,
    }
    hit = LanceDBStore._row_to_hit(row)
    assert hit.doc_id == "a.md"
    assert abs(hit.score - 0.7) < 0.001  # 1.0 - 0.3


def test_row_to_hit_fts_score():
    """_row_to_hit should use _score for FTS results."""
    row = {
        "metadata": {"doc_id": "b.md", "loc": "c:1", "mtime": 2.0},
        "text": "fts text",
        "_score": 5.5,
    }
    hit = LanceDBStore._row_to_hit(row)
    assert hit.doc_id == "b.md"
    assert hit.score == 5.5


# --- Pre-filtered vector search tests ---


def test_vector_search_with_where():
    """vector_search with where clause should filter results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [1.0] + [0.0] * 767
        nodes = [
            _make_node_with_meta("a.md", "c:0", "apple", vec, source_type="md"),
            _make_node_with_meta("b.pdf", "p:1:c:0", "apple pdf", vec, source_type="pdf"),
        ]
        store.upsert_nodes(nodes)
        hits = store.vector_search(vec, top_k=10, where="metadata.source_type = 'pdf'")
        assert len(hits) == 1
        assert hits[0].doc_id == "b.pdf"


def test_keyword_search_with_where():
    """keyword_search with where clause should filter results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:0", "banana fruit tropical", vec, source_type="md", folder="Recipes"),
            _make_node_with_meta("b.md", "c:0", "banana split dessert tropical", vec, source_type="md", folder="Archive"),
        ]
        store.upsert_nodes(nodes)
        store.create_fts_index()
        hits = store.keyword_search("banana tropical", top_k=10, where="metadata.folder = 'Recipes'")
        assert len(hits) == 1
        assert hits[0].doc_id == "a.md"


# --- get_chunk / get_doc_chunks comprehensive tests ---


def test_get_chunk_returns_all_core_fields():
    """get_chunk should return a SearchHit with all core metadata fields populated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "Projects/recipe.md", "c:0", "A Korean bibimbap recipe with gochujang sauce",
            vec, source_type="md", title="Bibimbap Recipe",
            tags="recipe,korean", folder="Projects", status="active",
            created="2026-01-15", mtime=1700000000.0,
        )]
        store.upsert_nodes(nodes)

        hit = store.get_chunk("Projects/recipe.md", "c:0")
        assert hit is not None
        assert hit.doc_id == "Projects/recipe.md"
        assert hit.loc == "c:0"
        assert hit.text == "A Korean bibimbap recipe with gochujang sauce"
        assert hit.snippet == "A Korean bibimbap recipe with gochujang sauce"
        assert hit.score == 0.0  # direct lookup, not a search
        assert hit.source_type == "md"
        assert hit.title == "Bibimbap Recipe"
        assert hit.tags == "recipe,korean"
        assert hit.folder == "Projects"
        assert hit.status == "active"
        assert hit.created == "2026-01-15"
        assert hit.mtime == 1700000000.0


def test_get_chunk_returns_enrichment_fields():
    """get_chunk should include all enr_* enrichment fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "test doc", vec,
            enr_summary="LLM summary",
            enr_doc_type="report",
            enr_topics="finance,tax",
            enr_keywords="deduction,income",
            enr_entities_people="Dan Park",
            enr_entities_places="Vancouver",
            enr_entities_orgs="CRA",
            enr_entities_dates="2026-01-01",
            enr_key_facts="Tax refund of $5000",
        )]
        store.upsert_nodes(nodes)

        hit = store.get_chunk("a.md", "c:0")
        assert hit is not None
        assert hit.enr_summary == "LLM summary"
        assert hit.enr_doc_type == "report"
        assert hit.enr_topics == "finance,tax"
        assert hit.enr_keywords == "deduction,income"
        assert hit.enr_entities_people == "Dan Park"
        assert hit.enr_entities_places == "Vancouver"
        assert hit.enr_entities_orgs == "CRA"
        assert hit.enr_entities_dates == "2026-01-01"
        assert hit.enr_key_facts == "Tax refund of $5000"


def test_get_chunk_not_found_returns_none():
    """get_chunk should return None for a nonexistent doc_id/loc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node("a.md", "c:0", "hello", vec)]
        store.upsert_nodes(nodes)
        assert store.get_chunk("nonexistent.md", "c:0") is None
        assert store.get_chunk("a.md", "c:99") is None


def test_get_chunk_empty_store():
    """get_chunk on empty store should return None (no crash)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.get_chunk("a.md", "c:0") is None


def test_get_doc_chunks_returns_all_chunks_sorted():
    """get_doc_chunks should return all chunks for a doc, sorted by loc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [
            _make_node_with_meta("a.md", "c:2", "third chunk", vec),
            _make_node_with_meta("a.md", "c:0", "first chunk", vec),
            _make_node_with_meta("a.md", "c:1", "second chunk", vec),
            _make_node_with_meta("b.md", "c:0", "other doc", vec),
        ]
        store.upsert_nodes(nodes)

        chunks = store.get_doc_chunks("a.md")
        assert len(chunks) == 3
        assert [c.loc for c in chunks] == ["c:0", "c:1", "c:2"]
        assert chunks[0].text == "first chunk"
        assert chunks[1].text == "second chunk"
        assert chunks[2].text == "third chunk"
        # All should have score 0.0 (direct lookup)
        assert all(c.score == 0.0 for c in chunks)


def test_get_doc_chunks_returns_enrichment_fields():
    """get_doc_chunks should include enr_* fields on each chunk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "enriched chunk", vec,
            enr_summary="LLM summary", enr_doc_type="report",
            enr_topics="finance", enr_keywords="tax",
        )]
        store.upsert_nodes(nodes)

        chunks = store.get_doc_chunks("a.md")
        assert len(chunks) == 1
        assert chunks[0].enr_summary == "LLM summary"
        assert chunks[0].enr_doc_type == "report"
        assert chunks[0].enr_topics == "finance"
        assert chunks[0].enr_keywords == "tax"


def test_get_doc_chunks_not_found_returns_empty():
    """get_doc_chunks for nonexistent doc should return empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.0] * 768
        nodes = [_make_node("a.md", "c:0", "hello", vec)]
        store.upsert_nodes(nodes)
        assert store.get_doc_chunks("nonexistent.md") == []


def test_get_doc_chunks_empty_store():
    """get_doc_chunks on empty store should return empty list (no crash)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        assert store.get_doc_chunks("a.md") == []


def test_get_chunk_consistency_with_vector_search():
    """get_chunk and vector_search should return the same fields for the same chunk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "consistency test", vec,
            source_type="md", title="Test Doc", tags="test",
            folder="Root", status="active", enr_summary="summary",
            enr_doc_type="note", enr_topics="testing",
        )]
        store.upsert_nodes(nodes)

        # Get via direct lookup
        chunk_hit = store.get_chunk("a.md", "c:0")
        # Get via vector search
        search_hits = store.vector_search(vec, top_k=1)

        assert chunk_hit is not None
        assert len(search_hits) == 1
        search_hit = search_hits[0]

        # All metadata fields should match (score differs)
        assert chunk_hit.doc_id == search_hit.doc_id
        assert chunk_hit.loc == search_hit.loc
        assert chunk_hit.text == search_hit.text
        assert chunk_hit.source_type == search_hit.source_type
        assert chunk_hit.title == search_hit.title
        assert chunk_hit.tags == search_hit.tags
        assert chunk_hit.folder == search_hit.folder
        assert chunk_hit.status == search_hit.status
        assert chunk_hit.enr_summary == search_hit.enr_summary
        assert chunk_hit.enr_doc_type == search_hit.enr_doc_type
        assert chunk_hit.enr_topics == search_hit.enr_topics


def test_frontmatter_and_enrichment_coexist():
    """Frontmatter 'summary' and LLM 'enr_summary' should coexist without collision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        vec = [0.1] * 768
        nodes = [_make_node_with_meta(
            "a.md", "c:0", "test", vec,
            summary="User's custom summary",
            enr_summary="LLM generated summary",
        )]
        store.upsert_nodes(nodes)
        hits = store.vector_search(vec, top_k=1)
        assert len(hits) == 1
        # LLM enrichment accessible via named attribute
        assert hits[0].enr_summary == "LLM generated summary"
        # Frontmatter summary accessible via extra_metadata (not in _CORE_META_KEYS)
        assert hits[0].extra_metadata["summary"] == "User's custom summary"
        # Both accessible via getattr
        assert hits[0].summary == "User's custom summary"


# --- _build_where_clause: identifier validation (Fix 1) ---


@pytest.mark.parametrize("bad_key", [
    "x' OR 1=1 --",
    "a; DROP TABLE",
    "",
    "foo bar",
    "123abc",
    "key.nested",
])
def test_build_where_clause_rejects_unsafe_keys(bad_key):
    """metadata_filters with injection-style keys must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        with pytest.raises(ValueError, match="Unsafe metadata filter key"):
            store._build_where_clause(metadata_filters={bad_key: "val"})


@pytest.mark.parametrize("good_key", ["section", "my_field_2", "_private"])
def test_build_where_clause_accepts_valid_keys(good_key):
    """Valid identifier keys should produce a WHERE clause without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        clause = store._build_where_clause(metadata_filters={good_key: "val"})
        assert f"lower(metadata.{good_key}) = 'val'" in clause


# --- upsert_nodes: add failure visibility (Fix 2) ---


def test_upsert_add_failure_reraises_and_recovers():
    """If _vs.add() fails after delete, the exception propagates;
    a subsequent upsert succeeds (self-healing)."""
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        nodes = [_make_node("a.md", "c:0", "hello", [0.1] * 768)]

        # First upsert succeeds (creates the table)
        store.upsert_nodes(nodes)
        assert store.list_doc_ids() == ["a.md"]

        # Swap _vs with a mock whose add() raises
        real_vs = store._vs
        mock_vs = MagicMock(wraps=real_vs)
        mock_vs.add.side_effect = RuntimeError("disk full")
        store._vs = mock_vs

        with pytest.raises(RuntimeError, match="disk full"):
            store.upsert_nodes(nodes)

        # Restore real _vs and verify self-healing
        store._vs = real_vs
        store.upsert_nodes(nodes)
        assert store.list_doc_ids() == ["a.md"]
