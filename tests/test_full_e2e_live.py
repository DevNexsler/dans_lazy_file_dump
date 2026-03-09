# Full End-to-End Live Test - Design Doc: system-design-doc.md
# Generated: 2026-03-08 | Budget Used: 4/5 E2E
# Test Type: End-to-End Test (REAL providers, full stack)
# Implementation Timing: After all feature implementations complete
#
# THE GAP THIS FILE FILLS:
# Existing tests cover (a) real providers WITHOUT REST/MCP transport, or
# (b) REST/MCP transport WITH mock providers.  This file is the ONLY test
# that exercises the COMPLETE journey with REAL cloud providers:
#   Upload via REST -> Index with real OpenRouter embeddings + LLM enrichment
#   -> Search via MCP with real DeepInfra reranker -> Download via REST
#   -> Verify taxonomy integration with real embeddings
#
# All tests require OPENROUTER_API_KEY and DEEPINFRA_API_KEY.
# Run with: pytest tests/test_full_e2e_live.py -v -s --timeout=300
#
# Framework: pytest + httpx (async) + anyio
# Providers: OpenRouter (embeddings + LLM), DeepInfra (reranker)

import json
import os
import tempfile
import time
from pathlib import Path

import httpx
import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
_has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (_has_openrouter and _has_deepinfra),
        reason="OPENROUTER_API_KEY and/or DEEPINFRA_API_KEY not set",
    ),
]


# ---------------------------------------------------------------------------
# Test documents -- diverse types to verify cross-document search
# ---------------------------------------------------------------------------

_MD_WITH_FRONTMATTER = """\
---
title: Quarterly Financial Summary Q4 2025
tags: [finance, quarterly-report, budgeting]
status: final
author: Jane Kim
created: 2025-12-20
---

# Quarterly Financial Summary Q4 2025

## Revenue Highlights

Total revenue for Q4 2025 reached $4.2 million, a 15% increase over Q3.
The SaaS division contributed $2.8M, up from $2.3M. Enterprise license
deals closed in November and December drove the majority of growth.

## Operating Expenses

Headcount increased by 12 FTEs, mostly in engineering and customer success.
Cloud infrastructure costs rose 8% due to GPU provisioning for the ML
pipeline. Marketing spend was flat at $320K.

## Key Metrics

- ARR: $16.8M (up from $14.6M in Q3)
- Net Revenue Retention: 118%
- Customer Acquisition Cost: $12,400
- Gross Margin: 72%

## Outlook

Q1 2026 pipeline includes three enterprise deals valued at $1.5M combined.
The board approved a $500K budget increase for AI infrastructure.
"""

_MD_PLAIN = """\
# Home Network Troubleshooting Guide

## Common Issues

### WiFi Drops Intermittently

1. Check router placement -- avoid metal cabinets and microwaves
2. Update firmware on the access point
3. Switch from 2.4 GHz to 5 GHz band for less interference
4. Verify DHCP lease pool is not exhausted

### Slow DNS Resolution

If websites take 5-10 seconds to start loading but then stream fast,
the problem is likely DNS. Try switching to Cloudflare (1.1.1.1) or
Google (8.8.8.8) public DNS servers.

### Port Forwarding Not Working

Ensure the firewall on the target machine allows inbound traffic on
the forwarded port. Double-check the router's NAT rules and make sure
UPnP is disabled if you prefer manual control.

## Hardware Recommendations

- Router: Ubiquiti Dream Machine for prosumer setups
- Access Points: TP-Link EAP670 for Wi-Fi 6
- Switch: Netgear GS308E for managed VLAN support
"""

_MD_THIRD_DOC = """\
# Sourdough Bread Baking Notes

## Starter Maintenance

Feed the starter every 12 hours at room temperature with equal parts
flour and water by weight. Discard half before feeding to maintain
healthy microbial balance. The starter is ready when it doubles in
volume within 4-6 hours after feeding.

## Basic Recipe

- 500g bread flour
- 350g water (70% hydration)
- 100g active starter
- 10g salt

Mix flour and water, autolyse 30 minutes. Add starter and salt,
do 4 sets of stretch-and-folds over 2 hours. Bulk ferment 4-6 hours
at 75F until 50% volume increase. Shape, proof in banneton overnight
in refrigerator. Bake in Dutch oven at 500F for 20 min covered,
then 450F for 25 min uncovered.

## Troubleshooting

Dense crumb often means under-fermentation. Increase bulk ferment time
or use a warmer ambient temperature. Gummy interior usually indicates
insufficient bake time -- the internal temperature should reach 210F.
"""


# ---------------------------------------------------------------------------
# Shared fixture: wire up the FULL system with REAL providers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_system():
    """Build the full system with real providers, index 3 documents, yield
    everything needed for tests.

    This fixture:
    1. Creates temp directories for documents_root and index_root
    2. Writes 3 test .md files to documents_root
    3. Builds real OpenRouter embed + LLM providers from config
    4. Runs enrichment + chunking + embedding + upsert for each doc
    5. Builds FTS index
    6. Wires up REST API app + MCP _cache
    7. Yields system dict with client, store, providers, config
    """
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    from api_server import build_api_app
    from core.config import load_config
    from doc_enrichment import enrich_document
    from extractors import extract_text, extract_title, derive_folder, normalize_tags
    from lancedb_store import LanceDBStore
    from providers.embed import build_embed_provider
    from providers.llm import build_llm_provider
    from search_hybrid import build_reranker
    import mcp_server

    config = load_config()

    with tempfile.TemporaryDirectory() as docs_root, \
         tempfile.TemporaryDirectory() as index_root:

        docs_path = Path(docs_root)
        index_path = Path(index_root)

        # Write 3 test documents to documents_root
        doc_files = {
            "financial_summary.md": _MD_WITH_FRONTMATTER,
            "network_guide.md": _MD_PLAIN,
            "sourdough_notes.md": _MD_THIRD_DOC,
        }
        for fname, content in doc_files.items():
            (docs_path / fname).write_text(content)

        # Build real providers from production config
        embed_provider = build_embed_provider(config)
        llm_generator = build_llm_provider(config)
        enrichment_cfg = config.get("enrichment", {})
        chunking_cfg = config.get("chunking", {})
        splitter = SentenceSplitter(
            chunk_size=chunking_cfg.get("max_chars", 1800),
            chunk_overlap=chunking_cfg.get("overlap", 200),
        )

        # Build LanceDB store in temp index_root
        store = LanceDBStore(index_root, "chunks")

        # Index each document: extract -> enrich -> chunk -> embed -> upsert
        for doc_id, content in doc_files.items():
            ext = "md"
            source_type = "md"

            # Extract text using the real extractor
            result = extract_text(
                file_path=str(docs_path / doc_id),
                ext=ext,
            )
            full_text = result.full_text
            fm = result.frontmatter

            title = fm.get("title") or extract_title(full_text, doc_id)
            tags = normalize_tags(fm.get("tags"))
            folder = derive_folder(doc_id)
            status = fm.get("status", "active")
            created = str(fm["created"]) if "created" in fm else ""
            author = str(fm.get("author", "")).strip()

            doc_meta = {
                "doc_id": doc_id,
                "source_type": source_type,
                "mtime": time.time(),
                "size": len(content),
                "title": title,
                "tags": tags,
                "folder": folder,
                "status": status,
                "created": created,
                "description": "",
                "author": author,
                "keywords": "",
                "custom_meta": "",
                "section": "",
            }

            # Enrich via real LLM
            enrichment = {}
            if llm_generator:
                enrichment = enrich_document(
                    text=full_text,
                    title=title,
                    source_type=source_type,
                    generator=llm_generator,
                    max_input_chars=enrichment_cfg.get("max_input_chars", 20000),
                    max_output_tokens=enrichment_cfg.get("max_output_tokens", 5000),
                )
                enrichment.pop("_enrichment_failed", None)
            doc_meta.update(enrichment)

            # Chunk and embed
            chunks_text = splitter.split_text(full_text)
            vectors = embed_provider.embed_texts(chunks_text)

            # Build TextNodes
            nodes = []
            for i, (chunk_text, vector) in enumerate(zip(chunks_text, vectors)):
                loc = f"c:{i}"
                chunk_uid = f"{doc_id}::{loc}"
                snippet = (chunk_text[:200] + "...") if len(chunk_text) > 200 else chunk_text
                meta = {**doc_meta, "loc": loc, "snippet": snippet}
                node = TextNode(
                    text=chunk_text,
                    id_=chunk_uid,
                    embedding=vector,
                    metadata=meta,
                )
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
                nodes.append(node)

            store.upsert_nodes(nodes)

        # Build FTS index for keyword search
        store.create_fts_index()

        # Write index_metadata.json so _file_status_impl can read it
        meta_path = index_path / "index_metadata.json"
        doc_ids = store.list_doc_ids()
        chunk_count = store.count_chunks()
        from datetime import datetime, timezone
        meta = {
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "doc_count": len(doc_ids),
            "chunk_count": chunk_count,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Build a config dict pointing to temp dirs for MCP/REST
        test_config = dict(config)
        test_config["documents_root"] = docs_root
        test_config["vault_root"] = docs_root
        test_config["index_root"] = index_root

        # Wire up REST API app
        api_app = build_api_app(docs_path)

        # Wire up MCP _cache
        mcp_server._cache = (store, embed_provider, test_config)

        yield {
            "store": store,
            "embed_provider": embed_provider,
            "llm_generator": llm_generator,
            "config": test_config,
            "api_app": api_app,
            "docs_root": docs_root,
            "index_root": index_root,
            "doc_files": doc_files,
        }


# ===================================================================
# TEST 1: Upload -> Index -> Search -> Download (Single Document)
# ===================================================================


@pytest.mark.anyio
async def test_upload_index_search_download_with_real_providers(live_system):
    """E2E: Upload .md via REST API -> index with real OpenRouter embeddings
    + LLM enrichment -> search via MCP handler with DeepInfra reranker
    -> verify enrichment fields populated -> download original file via REST.

    This is the SINGLE most important production-readiness test.
    """
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    from doc_enrichment import enrich_document
    from extractors import extract_text, extract_title, derive_folder, normalize_tags
    import mcp_server

    store = live_system["store"]
    embed_provider = live_system["embed_provider"]
    llm_generator = live_system["llm_generator"]
    config = live_system["config"]
    api_app = live_system["api_app"]
    docs_root = live_system["docs_root"]

    # -- Act: Step 1 -- Upload via REST API --
    new_doc_content = _MD_WITH_FRONTMATTER
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/upload",
            files={"file": ("uploaded_financial.md", new_doc_content.encode(), "text/markdown")},
        )

    assert resp.status_code == 201, f"Upload failed: {resp.status_code} {resp.text}"
    upload_data = resp.json()
    assert upload_data["uploaded"] is True
    assert "doc_id" in upload_data
    assert upload_data["size"] > 0
    uploaded_doc_id = upload_data["doc_id"]

    # Verify file exists on disk
    uploaded_path = Path(docs_root) / uploaded_doc_id
    assert uploaded_path.exists(), f"Uploaded file not found at {uploaded_path}"

    # -- Act: Step 2 -- Index the uploaded file --
    enrichment_cfg = config.get("enrichment", {})
    chunking_cfg = config.get("chunking", {})
    splitter = SentenceSplitter(
        chunk_size=chunking_cfg.get("max_chars", 1800),
        chunk_overlap=chunking_cfg.get("overlap", 200),
    )

    result = extract_text(file_path=str(uploaded_path), ext="md")
    full_text = result.full_text
    fm = result.frontmatter
    title = fm.get("title") or extract_title(full_text, uploaded_doc_id)
    tags = normalize_tags(fm.get("tags"))
    folder = derive_folder(uploaded_doc_id)
    status_val = fm.get("status", "active")
    created = str(fm["created"]) if "created" in fm else ""
    author = str(fm.get("author", "")).strip()

    doc_meta = {
        "doc_id": uploaded_doc_id,
        "source_type": "md",
        "mtime": time.time(),
        "size": len(new_doc_content),
        "title": title,
        "tags": tags,
        "folder": folder,
        "status": status_val,
        "created": created,
        "description": "",
        "author": author,
        "keywords": "",
        "custom_meta": "",
        "section": "",
    }

    # Enrich via real LLM
    enrichment = {}
    if llm_generator:
        enrichment = enrich_document(
            text=full_text,
            title=title,
            source_type="md",
            generator=llm_generator,
            max_input_chars=enrichment_cfg.get("max_input_chars", 20000),
            max_output_tokens=enrichment_cfg.get("max_output_tokens", 5000),
        )
        enrichment.pop("_enrichment_failed", None)
    doc_meta.update(enrichment)

    # Chunk and embed
    chunks_text = splitter.split_text(full_text)
    vectors = embed_provider.embed_texts(chunks_text)

    nodes = []
    for i, (chunk_text, vector) in enumerate(zip(chunks_text, vectors)):
        loc = f"c:{i}"
        chunk_uid = f"{uploaded_doc_id}::{loc}"
        snippet = (chunk_text[:200] + "...") if len(chunk_text) > 200 else chunk_text
        meta = {**doc_meta, "loc": loc, "snippet": snippet}
        node = TextNode(
            text=chunk_text,
            id_=chunk_uid,
            embedding=vector,
            metadata=meta,
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=uploaded_doc_id)
        nodes.append(node)

    store.upsert_nodes(nodes)
    store.create_fts_index()

    # Verify document appears in store
    doc_ids = store.list_doc_ids()
    assert uploaded_doc_id in doc_ids, f"Uploaded doc {uploaded_doc_id} not in store: {doc_ids}"

    # -- Act: Step 3 -- Search via MCP handler --
    mcp_server._cache = (store, embed_provider, config)
    search_response = mcp_server._file_search_impl(
        "quarterly revenue SaaS enterprise", top_k=5
    )

    assert "results" in search_response, f"Search returned error: {search_response}"
    results = search_response["results"]
    assert len(results) >= 1, "Expected at least 1 search result"

    # Uploaded doc should appear in results (top_k=5)
    result_doc_ids = [r["doc_id"] for r in results]
    assert uploaded_doc_id in result_doc_ids, (
        f"Expected {uploaded_doc_id} in results, got {result_doc_ids}"
    )

    # Verify diagnostics
    diag = search_response["diagnostics"]
    assert diag.get("reranker_applied") is True, (
        f"Reranker should be applied, diagnostics: {diag}"
    )
    assert diag.get("degraded") is False
    assert diag.get("vector_search_active") is True
    assert diag.get("keyword_search_active") is True

    # -- Act: Step 4 -- Verify enrichment fields --
    # Find the uploaded doc's result
    uploaded_result = next(
        (r for r in results if r["doc_id"] == uploaded_doc_id), results[0]
    )

    assert uploaded_result.get("enr_summary"), "enr_summary should be non-empty"
    assert uploaded_result.get("enr_doc_type"), "enr_doc_type should be non-empty"
    assert uploaded_result.get("enr_topics"), "enr_topics should be non-empty"
    assert uploaded_result.get("enr_keywords"), "enr_keywords should be non-empty"
    assert uploaded_result.get("enr_key_facts"), "enr_key_facts should be non-empty"

    # -- Act: Step 5 -- Download via REST API --
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        download_resp = await client.get(f"/documents/{uploaded_doc_id}")

    assert download_resp.status_code == 200, (
        f"Download failed: {download_resp.status_code} {download_resp.text}"
    )
    assert download_resp.content == new_doc_content.encode(), (
        "Downloaded content does not match uploaded content"
    )


# ===================================================================
# TEST 2: Multi-Document Cross-Document Search Quality
# ===================================================================


@pytest.mark.anyio
async def test_multi_document_search_quality_with_reranker(live_system):
    """E2E: Index 3 diverse documents (finance, networking, baking) -> verify
    targeted queries find correct documents with reranker improving precision.

    Tests that the system can discriminate between semantically distinct
    documents and that the DeepInfra reranker produces meaningful score
    separation between relevant and irrelevant results.
    """
    import mcp_server

    store = live_system["store"]
    embed_provider = live_system["embed_provider"]
    config = live_system["config"]

    mcp_server._cache = (store, embed_provider, config)

    # -- Act & Assert: Query 1 -- Finance-specific query --
    finance_resp = mcp_server._file_search_impl(
        "quarterly revenue ARR enterprise SaaS growth", top_k=10
    )
    assert "results" in finance_resp, f"Finance search error: {finance_resp}"
    finance_results = finance_resp["results"]
    assert len(finance_results) >= 1

    # financial_summary.md should be top result
    assert finance_results[0]["doc_id"] == "financial_summary.md", (
        f"Expected financial_summary.md as top result, got {finance_results[0]['doc_id']}"
    )
    assert finance_resp["diagnostics"]["reranker_applied"] is True

    # -- Act & Assert: Query 2 -- Networking-specific query --
    network_resp = mcp_server._file_search_impl(
        "WiFi router DNS troubleshooting port forwarding", top_k=10
    )
    assert "results" in network_resp, f"Network search error: {network_resp}"
    network_results = network_resp["results"]
    assert len(network_results) >= 1

    assert network_results[0]["doc_id"] == "network_guide.md", (
        f"Expected network_guide.md as top result, got {network_results[0]['doc_id']}"
    )
    assert network_resp["diagnostics"]["reranker_applied"] is True

    # -- Act & Assert: Query 3 -- Baking-specific query --
    baking_resp = mcp_server._file_search_impl(
        "sourdough starter fermentation bread baking", top_k=10
    )
    assert "results" in baking_resp, f"Baking search error: {baking_resp}"
    baking_results = baking_resp["results"]
    assert len(baking_results) >= 1

    assert baking_results[0]["doc_id"] == "sourdough_notes.md", (
        f"Expected sourdough_notes.md as top result, got {baking_results[0]['doc_id']}"
    )
    assert baking_resp["diagnostics"]["reranker_applied"] is True

    # -- Act & Assert: Cross-document search --
    broad_resp = mcp_server._file_search_impl(
        "document report content", top_k=10
    )
    assert "results" in broad_resp, f"Broad search error: {broad_resp}"
    broad_results = broad_resp["results"]
    broad_doc_ids = {r["doc_id"] for r in broad_results}

    assert len(broad_doc_ids) >= 2, (
        f"Expected results from at least 2 distinct docs, got {broad_doc_ids}"
    )


# ===================================================================
# TEST 3: MCP Handlers Return Complete Metadata with Real Data
# ===================================================================


@pytest.mark.anyio
async def test_mcp_metadata_handlers_with_real_indexed_data(live_system):
    """E2E: Verify MCP list, facets, and status handlers return correct
    metadata after indexing with real providers.

    Confirms that enrichment-derived fields (enr_doc_type, enr_topics, etc.)
    are aggregated into facets and that document counts are accurate.
    """
    import mcp_server

    store = live_system["store"]
    embed_provider = live_system["embed_provider"]
    config = live_system["config"]

    mcp_server._cache = (store, embed_provider, config)

    # -- Act & Assert: file_list_documents --
    list_result = mcp_server._file_list_documents_impl(offset=0, limit=10)
    assert isinstance(list_result, dict), f"list_documents returned non-dict: {list_result}"
    assert "error" not in list_result, f"list_documents error: {list_result}"
    assert list_result["total"] >= 3, (
        f"Expected >= 3 docs, got {list_result['total']}"
    )
    assert len(list_result["documents"]) >= 3
    for doc in list_result["documents"]:
        assert "doc_id" in doc, f"Document missing doc_id: {doc}"

    # At least one document should have tags (from frontmatter)
    docs_with_tags = [
        d for d in list_result["documents"]
        if d.get("tags") and len(d["tags"]) > 0
    ]
    assert len(docs_with_tags) >= 1, "Expected at least one doc with tags from frontmatter"

    # -- Act & Assert: file_facets --
    facets_result = mcp_server._file_facets_impl()
    assert isinstance(facets_result, dict), f"facets returned non-dict: {facets_result}"
    assert "error" not in facets_result, f"facets error: {facets_result}"
    assert facets_result["total_docs"] >= 3, (
        f"Expected >= 3 total_docs, got {facets_result['total_docs']}"
    )
    assert facets_result["total_chunks"] > 3, (
        f"Expected > 3 total_chunks (each doc produces multiple), got {facets_result['total_chunks']}"
    )

    # source_types should contain "md"
    source_types = facets_result.get("source_types", [])
    source_type_values = [s["value"] for s in source_types]
    assert "md" in source_type_values, (
        f"Expected 'md' in source_types, got {source_type_values}"
    )

    # tags should be non-empty (from frontmatter)
    tags_facets = facets_result.get("tags", [])
    assert len(tags_facets) > 0, "Expected non-empty tags facet from frontmatter"

    # doc_types should be non-empty (from enr_doc_type via real LLM)
    doc_types = facets_result.get("doc_types", [])
    assert len(doc_types) > 0, (
        f"Expected non-empty doc_types from LLM enrichment, got {doc_types}"
    )

    # topics should be non-empty (from enr_topics via real LLM)
    topics = facets_result.get("topics", [])
    assert len(topics) > 0, (
        f"Expected non-empty topics from LLM enrichment, got {topics}"
    )

    # -- Act & Assert: file_status --
    status_result = mcp_server._file_status_impl()
    assert isinstance(status_result, dict), f"status returned non-dict: {status_result}"
    assert "error" not in status_result, f"status error: {status_result}"
    assert status_result["doc_count"] >= 3, (
        f"Expected >= 3 doc_count, got {status_result['doc_count']}"
    )
    assert status_result["chunk_count"] > 3
    assert status_result["embeddings_provider"], "embeddings_provider should be non-empty"

    # -- Act & Assert: file_get_doc_chunks --
    chunks_result = mcp_server._file_get_doc_chunks_impl("financial_summary.md")
    assert isinstance(chunks_result, list), (
        f"get_doc_chunks returned non-list: {chunks_result}"
    )
    assert len(chunks_result) >= 1, "Expected at least 1 chunk for financial_summary.md"

    for chunk in chunks_result:
        assert "text" in chunk, f"Chunk missing 'text': {chunk.keys()}"
        assert "loc" in chunk, f"Chunk missing 'loc': {chunk.keys()}"
        assert "doc_id" in chunk, f"Chunk missing 'doc_id': {chunk.keys()}"

    # Chunks should be ordered by loc
    locs = [c["loc"] for c in chunks_result]
    assert locs == sorted(locs), f"Chunks not ordered by loc: {locs}"

    # At least one chunk should contain "revenue" (content preserved)
    has_revenue = any("revenue" in c["text"].lower() for c in chunks_result)
    assert has_revenue, "Expected at least one chunk to contain 'revenue'"


# ===================================================================
# TEST 4: Taxonomy-Guided Enrichment with Real Embeddings
# ===================================================================


@pytest.mark.anyio
async def test_taxonomy_guided_enrichment_with_real_providers(live_system):
    """E2E: Create taxonomy entries with real embeddings -> enrich a document
    with real LLM -> verify suggestions align with taxonomy vocabulary.

    Tests that the taxonomy-to-LLM-prompt pipeline works end-to-end:
    taxonomy entries are embedded, format_for_prompt renders them,
    and the LLM's suggestions reference the provided vocabulary.
    """
    from taxonomy_store import TaxonomyStore
    from doc_enrichment import enrich_document

    embed_provider = live_system["embed_provider"]
    llm_generator = live_system["llm_generator"]
    config = live_system["config"]
    enrichment_cfg = config.get("enrichment", {})

    if not llm_generator:
        pytest.skip("LLM generator not available")

    def embed_fn(text):
        return embed_provider.embed_texts([text])[0]

    with tempfile.TemporaryDirectory() as tax_dir:
        # -- Arrange: Create taxonomy store with real embeddings --
        tax_store = TaxonomyStore(tax_dir, "taxonomy", embed_fn=embed_fn)

        # Add taxonomy entries
        tax_store.add("tag", "finance", "Financial documents, budgets, revenue reports")
        tax_store.add("tag", "cooking", "Recipes, food preparation, culinary techniques")
        tax_store.add("folder", "Reports/Financial", "Quarterly and annual financial reports")
        tax_store.add("folder", "Recipes", "Food and cooking recipes")

        # -- Act: format_for_prompt --
        prompt_text = tax_store.format_for_prompt()
        assert "finance" in prompt_text.lower(), (
            f"format_for_prompt should contain 'finance', got: {prompt_text[:300]}"
        )
        assert "cooking" in prompt_text.lower(), (
            f"format_for_prompt should contain 'cooking', got: {prompt_text[:300]}"
        )

        # -- Act: Enrich document with taxonomy --
        enrichment = enrich_document(
            text=_MD_WITH_FRONTMATTER,
            title="financial_summary.md",
            source_type="md",
            generator=llm_generator,
            max_input_chars=enrichment_cfg.get("max_input_chars", 20000),
            max_output_tokens=enrichment_cfg.get("max_output_tokens", 5000),
            taxonomy_store=tax_store,
        )

        # Verify enrichment succeeded
        assert "_enrichment_failed" not in enrichment, (
            f"Enrichment failed: {enrichment.get('_enrichment_failed')}"
        )
        assert enrichment.get("enr_summary"), "enr_summary should be non-empty"
        assert enrichment.get("enr_suggested_tags"), "enr_suggested_tags should be non-empty"

        # Verify taxonomy vocabulary is referenced in suggestions
        suggested_tags = enrichment.get("enr_suggested_tags", "").lower()
        suggested_folder = enrichment.get("enr_suggested_folder", "").lower()
        has_taxonomy_ref = (
            "finance" in suggested_tags
            or "financial" in suggested_tags
            or "finance" in suggested_folder
            or "financial" in suggested_folder
        )
        assert has_taxonomy_ref, (
            f"Expected taxonomy reference in suggestions. "
            f"suggested_tags='{enrichment.get('enr_suggested_tags')}', "
            f"suggested_folder='{enrichment.get('enr_suggested_folder')}'"
        )

        # -- Act: Semantic search on taxonomy --
        search_results = tax_store.search("budget revenue quarterly", kind="tag", top_k=5)
        assert len(search_results) > 0, "Expected taxonomy search results"

        # Finance tag should be found (semantic match)
        tag_names = [r["name"] for r in search_results]
        assert "finance" in tag_names, (
            f"Expected 'finance' in taxonomy search results, got {tag_names}"
        )

        # Finance should rank higher than cooking for a financial query
        if len(search_results) >= 2:
            finance_idx = next(
                (i for i, r in enumerate(search_results) if r["name"] == "finance"), None
            )
            cooking_idx = next(
                (i for i, r in enumerate(search_results) if r["name"] == "cooking"), None
            )
            if finance_idx is not None and cooking_idx is not None:
                assert finance_idx < cooking_idx, (
                    f"Finance should rank higher than cooking for financial query. "
                    f"finance_idx={finance_idx}, cooking_idx={cooking_idx}"
                )
