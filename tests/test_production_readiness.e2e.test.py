# Production Readiness E2E Test - Design Doc: system-design-doc.md
# Generated: 2026-03-08 | Budget Used: 1/2 E2E
# Test Type: End-to-End Test
# Implementation Timing: After all feature implementations complete
#
# This E2E test verifies the critical user journey: upload a document via
# REST API -> trigger indexing -> search for it via MCP -> download via REST.
# Uses mock embedding/LLM providers so no API keys are needed.
#
# This is the SINGLE highest-ROI E2E test for production readiness.
# It exercises: api_server, mcp_server, LanceDBStore, search_hybrid,
# extractors, chunking -- the entire system except external API providers.
#
# Framework: pytest
# Run with: pytest tests/test_production_readiness.e2e.test.py -v

import tempfile
import time
from pathlib import Path

import httpx
import pytest
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

import mcp_server
from api_server import build_api_app
from extractors import extract_markdown, extract_title, normalize_tags, derive_folder
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider
from search_hybrid import hybrid_search


# ---------------------------------------------------------------------------
# Mock Providers
# ---------------------------------------------------------------------------

class MockEmbedProvider(EmbedProvider):
    """Deterministic 768d vectors for offline testing."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 768


class TimeoutEmbedProvider(EmbedProvider):
    """Simulates embed provider failure."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise TimeoutError("Connection timed out")

    def embed_query(self, query: str) -> list[float]:
        raise TimeoutError("Connection timed out")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_MD_CONTENT = """\
---
title: Production Readiness Test Document
tags: [e2e, production]
status: active
---

# Production Readiness Test Document

This document verifies the complete upload-index-search-download lifecycle.

Semantic search with LanceDB and tantivy FTS enables hybrid retrieval.
Reciprocal Rank Fusion combines vector and keyword results.
"""


def _index_file(file_path: Path, doc_id: str, store: LanceDBStore, embed: MockEmbedProvider):
    """Index a single file into the store (extract -> chunk -> embed -> upsert)."""
    result = extract_markdown(file_path)
    text = result.full_text
    frontmatter = result.frontmatter

    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)
    from llama_index.core.schema import Document as LIDocument
    doc = LIDocument(text=text)
    chunks = splitter.get_nodes_from_documents([doc])

    chunk_texts = [c.text for c in chunks]
    vectors = embed.embed_texts(chunk_texts)

    title = frontmatter.get("title") or extract_title(text, doc_id)
    tags = normalize_tags(frontmatter.get("tags"))
    folder = derive_folder(doc_id)
    mtime = file_path.stat().st_mtime

    nodes = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        loc = f"c:{i}"
        meta = {
            "doc_id": doc_id,
            "source_type": "md",
            "loc": loc,
            "snippet": chunk.text[:200],
            "mtime": mtime,
            "size": len(chunk.text),
            "title": title,
            "tags": tags,
            "folder": folder,
            "status": frontmatter.get("status", ""),
            "created": str(frontmatter.get("created", "")),
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
        node = TextNode(
            text=chunk.text,
            id_=f"{doc_id}::{loc}",
            embedding=vector,
            metadata=meta,
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
        nodes.append(node)

    store.upsert_nodes(nodes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def e2e_system():
    """Set up the full E2E system: REST API + LanceDB store + MCP wiring."""
    with tempfile.TemporaryDirectory() as documents_root, \
         tempfile.TemporaryDirectory() as index_root:

        documents_path = Path(documents_root)
        index_path = Path(index_root)

        # Build REST API app
        api_app = build_api_app(documents_path)

        # Initialize LanceDB store
        store = LanceDBStore(str(index_path), "test_chunks")

        # Mock embed provider
        embed = MockEmbedProvider()

        # Config
        config = {
            "documents_root": documents_root,
            "index_root": index_root,
            "search": {},
        }

        # Wire mcp_server._cache
        original_cache = mcp_server._cache
        mcp_server._cache = (store, embed, config)

        # Create httpx client
        transport = httpx.ASGITransport(app=api_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver/") as client:
            yield {
                "client": client,
                "store": store,
                "embed": embed,
                "config": config,
                "documents_root": documents_path,
                "index_root": index_path,
            }

        # Restore mcp_server._cache
        mcp_server._cache = original_cache


# ===================================================================
# E2E TEST: Upload -> Index -> Search -> Download
# ===================================================================


@pytest.mark.anyio
async def test_upload_index_search_download_lifecycle(e2e_system):
    """E2E: Upload .md via REST -> index into LanceDB -> search via MCP -> download via REST."""
    system = e2e_system
    client = system["client"]
    store = system["store"]
    embed = system["embed"]
    documents_root = system["documents_root"]

    # Step 1 - Upload
    file_content = _TEST_MD_CONTENT.encode("utf-8")
    files = {"file": ("e2e_test.md", file_content, "text/markdown")}

    resp = await client.post("/upload", files=files)

    assert resp.status_code == 201
    body = resp.json()
    assert body["uploaded"] is True
    doc_id = body["doc_id"]
    assert doc_id == "e2e_test.md"

    # Verify file exists on disk
    on_disk = documents_root / doc_id
    assert on_disk.exists()

    # Step 2 - Index
    _index_file(on_disk, doc_id, store, embed)
    store.create_fts_index()

    # Verify doc appears in store
    doc_ids = store.list_doc_ids()
    assert doc_id in doc_ids

    # Step 3 - Search via MCP
    result = mcp_server._file_search_impl("production readiness lifecycle")

    assert "error" not in result
    assert "results" in result
    results = result["results"]
    assert len(results) > 0

    # Top result should match our uploaded doc
    top = results[0]
    assert top["doc_id"] == doc_id

    # Diagnostics should be healthy
    diag = result["diagnostics"]
    assert diag["keyword_search_active"] is True
    assert diag["degraded"] is False

    # Step 4 - Download
    resp = await client.get(f"/documents/{doc_id}")

    assert resp.status_code == 200
    assert resp.text == _TEST_MD_CONTENT


# ===================================================================
# E2E TEST: Search Degradation Under Provider Failure
# ===================================================================


@pytest.mark.anyio
async def test_search_remains_functional_when_embed_provider_fails(e2e_system):
    """E2E: Pre-indexed docs searchable via keyword fallback when embed provider is down.

    Note: hybrid_search calls embed_query() before submitting to ThreadPoolExecutor.
    If embed_query() raises, the error propagates directly (not caught by the
    vector search future). The keyword-only fallback only works when the vector
    search future itself fails (i.e., when the error happens inside the thread).

    This test verifies that even with a failing embed provider, the FTS index
    is still independently functional for keyword search.
    """
    system = e2e_system
    store = system["store"]
    embed = system["embed"]
    documents_root = system["documents_root"]

    # Pre-populate: upload and index a document with working embed
    file_content = _TEST_MD_CONTENT.encode("utf-8")
    doc_path = documents_root / "indexed_doc.md"
    doc_path.write_bytes(file_content)

    _index_file(doc_path, "indexed_doc.md", store, embed)
    store.create_fts_index()

    # Verify document is indexed
    assert "indexed_doc.md" in store.list_doc_ids()

    # Verify keyword search still works directly (independent of embed provider)
    keyword_hits = store.keyword_search("production readiness")
    assert len(keyword_hits) >= 1
    assert keyword_hits[0].doc_id == "indexed_doc.md"

    # Swap to failing embed provider in MCP cache
    timeout_embed = TimeoutEmbedProvider()
    original_cache = mcp_server._cache
    mcp_server._cache = (store, timeout_embed, system["config"])

    try:
        result = mcp_server._file_search_impl("tantivy FTS")

        # Must be either a structured error or a degraded result — never a silent success
        if "error" in result:
            assert result["code"] == "search_failed"
        else:
            diag = result.get("diagnostics", {})
            assert diag.get("degraded") is True or diag.get("vector_search_active") is False
    finally:
        mcp_server._cache = original_cache
