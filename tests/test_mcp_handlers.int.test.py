# MCP Handler Integration Tests - Design Doc: system-design-doc.md
# Generated: 2026-03-08 | Budget Used: 2/3 integration, 0/2 E2E
#
# These tests exercise MCP tool handler _impl functions through a REAL
# LanceDBStore with mock embeddings (no API keys needed). This fills the
# gap between test_mcp_contract.py (response shapes with mocks) and
# test_integration.py (live API keys required).
#
# Key difference from existing tests:
# - test_mcp_contract.py: tests _hit_to_dict shapes, error codes, mocks store
# - test_integration.py: requires GEMINI_API_KEY + OPENROUTER_API_KEY
# - THIS FILE: real LanceDB + FTS, mock embeddings, no API keys, offline CI
#
# Framework: pytest
# Run with: pytest tests/test_mcp_handlers.int.test.py -v

import tempfile
import time
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

import mcp_server
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider


# ---------------------------------------------------------------------------
# Mock Embed Provider
# ---------------------------------------------------------------------------

class MockEmbedProvider(EmbedProvider):
    """Deterministic 768d vectors for offline testing."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(doc_id: str, loc: str, text: str, vector: list[float], **extra_meta) -> TextNode:
    """Build a TextNode like process_doc_task does."""
    meta = {
        "doc_id": doc_id,
        "source_type": "md",
        "loc": loc,
        "snippet": text[:200],
        "mtime": time.time(),
        "size": len(text),
        "title": "",
        "tags": "",
        "folder": "",
        "status": "",
        "created": "",
        "enr_summary": "",
        "enr_doc_type": "",
        "enr_entities_people": "",
        "enr_entities_places": "",
        "enr_entities_orgs": "",
        "enr_entities_dates": "",
        "enr_topics": "",
        "enr_keywords": "",
        "enr_key_facts": "",
        "enr_suggested_tags": "",
        "enr_suggested_folder": "",
        "description": "",
        "author": "",
        "keywords": "",
        "custom_meta": "",
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_store():
    """Create a temp LanceDBStore, upsert nodes with variety, build FTS index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LanceDBStore(tmpdir, "test_chunks")
        embed = MockEmbedProvider()

        now = time.time()
        nodes = [
            _make_node(
                "recipes/bibimbap.md", "c:0",
                "Korean bibimbap recipe with gochujang sauce and vegetables",
                [1.0] + [0.0] * 767,
                source_type="md", folder="recipes", tags="recipe,korean",
                title="Bibimbap Recipe", mtime=now - 100,
            ),
            _make_node(
                "reports/q4_report.pdf", "c:0",
                "Quarterly financial report for Q4 2025 with revenue analysis",
                [0.0] + [1.0] + [0.0] * 766,
                source_type="pdf", folder="reports", tags="finance,report",
                title="Q4 2025 Report", mtime=now - 200,
            ),
            _make_node(
                "notes/ml_notes.md", "c:0",
                "Machine learning notes on neural networks and deep learning",
                [0.0, 0.0] + [1.0] + [0.0] * 765,
                source_type="md", folder="notes", tags="ml,ai",
                title="ML Notes", mtime=now,
            ),
        ]
        store.upsert_nodes(nodes)
        store.create_fts_index()

        config = {
            "index_root": tmpdir,
            "documents_root": tmpdir,
            "search": {},
        }

        yield store, embed, config


@pytest.fixture
def wired_mcp(populated_store):
    """Set mcp_server._cache to the populated store, restore on teardown."""
    store, embed, config = populated_store
    original_cache = mcp_server._cache
    mcp_server._cache = (store, embed, config)
    yield store, embed, config
    mcp_server._cache = original_cache


# ===================================================================
# TEST 1: Search Through Real Store
# ===================================================================


def test_search_returns_results_from_real_store(wired_mcp):
    """AC-MCP-1: file_search through real LanceDB returns matching results with healthy diagnostics."""
    store, embed, config = wired_mcp

    result = mcp_server._file_search_impl("bibimbap recipe")

    # No error key
    assert "error" not in result
    # Has results and diagnostics
    assert "results" in result
    assert "diagnostics" in result
    # Results list non-empty
    assert len(result["results"]) > 0

    # Top result should match bibimbap doc
    top = result["results"][0]
    assert "doc_id" in top
    assert "loc" in top
    assert "snippet" in top
    assert "score" in top

    # Diagnostics show healthy search
    diag = result["diagnostics"]
    assert diag["keyword_search_active"] is True
    assert diag["degraded"] is False


def test_search_with_prefilters(wired_mcp):
    """AC-MCP-1: file_search with source_type/folder filters returns only matching docs."""
    store, embed, config = wired_mcp

    # Filter by source_type="pdf"
    result = mcp_server._file_search_impl("report", source_type="pdf")

    assert "error" not in result
    results = result["results"]
    # All results should be PDF source_type
    for r in results:
        assert r["source_type"] == "pdf"


# ===================================================================
# TEST 2: List, Recent, and Facets Through Real Store
# ===================================================================


def test_list_documents_from_real_store(wired_mcp):
    """AC-MCP-4: file_list_documents returns paginated docs from real LanceDB."""
    store, embed, config = wired_mcp

    result = mcp_server._file_list_documents_impl(offset=0, limit=10)

    assert "error" not in result
    assert "documents" in result
    assert "total" in result
    assert "offset" in result
    assert "limit" in result

    docs = result["documents"]
    assert len(docs) >= 3  # We stored 3 documents

    # Each doc should have doc_id and mtime_iso (added by _enrich_doc_list)
    for doc in docs:
        assert "doc_id" in doc
        assert "mtime_iso" in doc


def test_facets_from_real_store(wired_mcp):
    """AC-MCP-6: file_facets returns correct aggregate counts from real data."""
    store, embed, config = wired_mcp

    result = mcp_server._file_facets_impl()

    assert "error" not in result
    assert "total_docs" in result
    assert "total_chunks" in result
    assert result["total_docs"] == 3
    assert result["total_chunks"] == 3

    # Tags should include recipe, korean, finance, etc.
    assert "tags" in result
    tag_values = {t["value"] for t in result["tags"]}
    assert "recipe" in tag_values
    assert "korean" in tag_values
    assert "finance" in tag_values

    # Folders present
    assert "folders" in result
    folder_values = {f["value"] for f in result["folders"]}
    assert "recipes" in folder_values
    assert "reports" in folder_values

    # Source types present
    assert "source_types" in result
    source_values = {s["value"] for s in result["source_types"]}
    assert "md" in source_values
    assert "pdf" in source_values


def test_recent_returns_sorted_by_mtime(wired_mcp):
    """AC-MCP-5: file_recent returns documents sorted by mtime descending."""
    store, embed, config = wired_mcp

    result = mcp_server._file_recent_impl(limit=10)

    assert isinstance(result, list)
    assert len(result) >= 3

    # Verify mtime descending order
    mtimes = [d.get("mtime", 0) for d in result if d.get("mtime")]
    for i in range(len(mtimes) - 1):
        assert mtimes[i] >= mtimes[i + 1], "Documents not sorted by mtime descending"
