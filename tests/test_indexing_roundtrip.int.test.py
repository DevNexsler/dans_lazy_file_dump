# Indexing Roundtrip Integration Tests - Design Doc: system-design-doc.md
# Generated: 2026-03-08 | Budget Used: 1/3 integration, 0/2 E2E
#
# Tests the full indexing pipeline offline: extract -> chunk -> store -> search.
# Uses REAL extractors and REAL LanceDB but MOCK embedding/LLM providers,
# so no API keys are needed. This fills the gap between:
# - test_extractors.py (extraction only)
# - test_scan.py (scanning + chunking only)
# - test_store.py (store CRUD only)
# - test_integration.py (requires live API keys)
#
# Framework: pytest
# Run with: pytest tests/test_indexing_roundtrip.int.test.py -v

import tempfile
import time
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

from extractors import extract_markdown, extract_title, normalize_tags, derive_folder
from lancedb_store import LanceDBStore
from providers.embed.base import EmbedProvider
from search_hybrid import hybrid_search


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

_MD_CONTENT = """\
---
title: Integration Test Note
tags: [testing, automation]
status: active
created: 2026-01-15
---

# Integration Test Note

This document is about integration testing with LanceDB.

## Section One

Machine learning algorithms are used for document search.
Embedding models convert text to vectors for semantic search.

## Section Two

Keyword search uses BM25 scoring via tantivy FTS index.
Hybrid search fuses vector and keyword results with RRF.
"""


def _run_indexing_pipeline(vault_path: Path, store: LanceDBStore, embed: MockEmbedProvider):
    """Execute the full indexing pipeline for all .md files in vault_path."""
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)

    for md_file in vault_path.glob("*.md"):
        doc_id = md_file.name

        # 1. Extract
        result = extract_markdown(md_file)
        text = result.full_text
        frontmatter = result.frontmatter

        # 2. Chunk
        from llama_index.core.schema import Document as LIDocument
        doc = LIDocument(text=text)
        chunks = splitter.get_nodes_from_documents([doc])

        # 3. Embed
        chunk_texts = [c.text for c in chunks]
        vectors = embed.embed_texts(chunk_texts)

        # 4. Build TextNodes with metadata
        title = frontmatter.get("title") or extract_title(text, doc_id)
        tags = normalize_tags(frontmatter.get("tags"))
        folder = derive_folder(doc_id)
        mtime = md_file.stat().st_mtime

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

        # 5. Upsert
        store.upsert_nodes(nodes)

    # 6. Build FTS index
    store.create_fts_index()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_vault(tmp_path):
    """Create a temp directory with a markdown file containing frontmatter."""
    md_file = tmp_path / "note.md"
    md_file.write_text(_MD_CONTENT)
    return tmp_path


@pytest.fixture
def mock_embed_provider():
    return MockEmbedProvider()


@pytest.fixture
def roundtrip_store(test_vault, mock_embed_provider):
    """Execute the roundtrip pipeline and return (store, embed, vault_path)."""
    with tempfile.TemporaryDirectory() as index_dir:
        store = LanceDBStore(index_dir, "test_chunks")
        _run_indexing_pipeline(test_vault, store, mock_embed_provider)
        yield store, mock_embed_provider, test_vault


# ===================================================================
# TEST 1: Extract -> Chunk -> Store -> Search Roundtrip
# ===================================================================


def test_markdown_roundtrip_extract_to_search(roundtrip_store):
    """AC-INDEX-1..5: Markdown file -> extract -> chunk -> store -> hybrid_search."""
    store, embed, vault_path = roundtrip_store

    result = hybrid_search(store, embed, "machine learning embedding")

    assert len(result) > 0

    # Top result should be from our note.md
    top = result[0]
    assert top.doc_id == "note.md"
    assert top.loc.startswith("c:")
    assert top.text  # non-empty text
    assert top.snippet  # non-empty snippet


def test_roundtrip_all_docs_indexed(roundtrip_store):
    """AC-INDEX-1: All test vault files appear in the store after indexing."""
    store, embed, vault_path = roundtrip_store

    doc_ids = store.list_doc_ids()
    assert "note.md" in doc_ids


def test_roundtrip_fts_index_works(roundtrip_store):
    """AC-INDEX-5: FTS index built after upsert enables keyword search."""
    store, embed, vault_path = roundtrip_store

    hits = store.keyword_search("bibimbap")
    # "bibimbap" is NOT in our test content, so no hits expected
    assert len(hits) == 0

    # "tantivy" IS in our test content (Section Two mentions it)
    hits = store.keyword_search("tantivy")
    assert len(hits) >= 1
    assert hits[0].doc_id == "note.md"


# ===================================================================
# TEST 2: Metadata Preservation Through Pipeline
# ===================================================================


def test_frontmatter_metadata_survives_pipeline(roundtrip_store):
    """AC-INDEX-2: Frontmatter title and tags are preserved through extract->store->retrieve."""
    store, embed, vault_path = roundtrip_store

    chunk = store.get_chunk("note.md", "c:0")
    assert chunk is not None

    # Title from frontmatter
    assert chunk.title == "Integration Test Note"

    # Tags from frontmatter (normalized to comma-separated string)
    assert "testing" in chunk.tags
    assert "automation" in chunk.tags

    # Status from frontmatter
    assert chunk.status == "active"

    # Created from frontmatter
    assert "2026-01-15" in chunk.created
