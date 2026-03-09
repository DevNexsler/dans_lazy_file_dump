# Design Doc: Document Organizer — Full System

**Status:** Active
**Created:** 2026-03-08
**Scope:** Entire system — indexing pipeline, search, MCP server, REST API, taxonomy

---

## 1. Overview

The Document Organizer (Obsidian Vault Semantic Index) is a full-pipeline semantic document indexing and retrieval system. It extracts text from Markdown, PDF, and image files, enriches metadata via LLM, chunks and embeds content, stores it in LanceDB, and exposes hybrid search through MCP tools and a REST API.

### 1.1 Goals

- Index a heterogeneous document vault (MD, PDF, images) with rich metadata
- Provide hybrid search (vector + BM25 keyword + RRF fusion + cross-encoder reranking)
- Expose search and management through MCP tools (for AI assistants) and REST API
- Maintain a taxonomy of tags, folders, and doc_types for consistent classification
- Support graceful degradation when providers are unavailable

---

## 2. Architecture

### 2.1 Components

| Component | Module | Purpose |
|-----------|--------|---------|
| Config | `core/config.py` | YAML config loading + validation |
| Extractors | `extractors.py` | MD/PDF/image text extraction, frontmatter parsing |
| Enrichment | `doc_enrichment.py` | LLM-based metadata extraction (summary, entities, topics) |
| Indexing Flow | `flow_index_vault.py` | Prefect-orchestrated pipeline: scan → diff → extract → enrich → chunk → embed → store |
| Vector Store | `lancedb_store.py` | LanceDB wrapper: vector search, keyword search (tantivy), CRUD, schema evolution |
| Hybrid Search | `search_hybrid.py` | Parallel vector+BM25 → RRF fusion → optional reranking → SearchResult |
| Taxonomy | `taxonomy_store.py` | LanceDB-backed CRUD for tags/folders/doc_types with semantic search |
| MCP Server | `mcp_server.py` | FastMCP tool handlers: search, list, facets, status, index, taxonomy CRUD |
| REST API | `api_server.py` | Starlette: file upload, download, directory listing |
| Unified Server | `server.py` | Combines MCP + REST on single port (default 7788) |
| Providers | `providers/` | Pluggable backends for embeddings, OCR, LLM |

### 2.2 Data Flow

```
Files on disk
  → scan_vault_task (glob matching, symlink following)
  → diff_index_task (mtime comparison with stored index)
  → process_doc_task per file:
      → extract text (MD frontmatter, PDF pages+OCR, image OCR+description)
      → LLM enrichment (summary, doc_type, entities, topics, keywords, key_facts, suggested_tags/folder)
      → chunk (heading-aware MD, page-aware PDF, SentenceSplitter)
      → prepend contextual headers to chunks
      → embed via provider
      → upsert TextNodes into LanceDB
  → rebuild FTS index
  → write index_metadata.json
```

### 2.3 Search Pipeline

```
Query
  → build WHERE clause (pre-filters: source_type, folder, tags, enr_doc_type, enr_topics, metadata_filters)
  → parallel: vector search (ANN) + keyword search (BM25/tantivy)
  → Reciprocal Rank Fusion (k=60)
  → length normalization (log-based penalty for long chunks, anchor=800)
  → importance weighting (score *= (1-w + w*importance); configurable field + weight)
  → optional recency boost + time decay with floor (when prefer_recent=true; old docs keep ≥50% score)
  → optional cross-encoder reranking (60/40 blend with original scores; cosine fallback on failure)
  → MMR diversity (defers near-duplicate chunks, cosine threshold=0.85)
  → minimum score threshold (discard results below configurable cutoff; default 0.0 = disabled)
  → SearchResult with hits[] + diagnostics{}
```

---

## 3. Component Specifications

### 3.1 REST API (`api_server.py`)

**Factory:** `build_api_app(documents_root: Path) -> Starlette`

#### AC-REST-1: File Upload
- `POST /api/upload` with multipart form data
- Fields: `file` (required), `directory` (optional relative path)
- Validates: max 100 MB, allowed extensions (.md, .pdf, .png, .jpg, .jpeg)
- Path traversal protection via `_safe_subpath()`
- Sanitizes filenames (no leading dots, no directory components)
- Returns 201: `{uploaded: true, doc_id: str, size: int}`
- Returns 400: `{error: true, code: str, message: str}` for validation failures

#### AC-REST-2: File Download
- `GET /api/documents/{doc_id:path}`
- Path traversal protection
- Returns FileResponse on success, 404 if not found

#### AC-REST-3: Directory Listing
- `GET /api/documents/` with query params: `directory`, `limit` (max 200), `offset`
- Returns: `{directory, files: [{name, type, path, size}], total, offset, limit}`
- Only shows files with allowed extensions; directories always shown
- Sorted alphabetically

#### AC-REST-4: Authentication
- Bearer token auth when `API_KEY` env var is set
- Applied to all endpoints uniformly

### 3.2 MCP Server (`mcp_server.py`)

#### AC-MCP-1: file_search
- Parameters: query (required), top_k, doc_id_prefix, source_type, tags, status, folder, prefer_recent, metadata_filters (JSON), enr_doc_type, enr_topics
- Returns: `{results: [{doc_id, loc, snippet, score, title, tags, folder, status, source_type, enr_* fields, ...}], diagnostics: {vector_search_active, keyword_search_active, reranker_applied, degraded}}`
- Error on empty query with code "empty_query"

#### AC-MCP-2: file_get_chunk
- Parameters: doc_id, loc
- Returns: single SearchHit dict with full `text` field

#### AC-MCP-3: file_get_doc_chunks
- Parameters: doc_id
- Returns: list of all chunks for document, sorted in document order

#### AC-MCP-4: file_list_documents
- Parameters: offset, limit (1-200), source_type, folder
- Returns: `{documents: [{doc_id, title, source_type, folder, tags, status, created, mtime, mtime_iso, size}], total, offset, limit}`

#### AC-MCP-5: file_recent
- Parameters: limit (1-100), source_type, folder
- Returns: list of document metadata sorted by mtime desc

#### AC-MCP-6: file_facets
- No parameters
- Returns: `{total_docs, total_chunks, tags[], folders[], source_types[], statuses[], doc_types[], authors[], topics[], keywords[], entities_*[]}`

#### AC-MCP-7: file_folders
- No parameters
- Scans filesystem (not index) for folders with supported files
- Returns: `{documents_root, folders: [{path, file_count}], total_folders, total_files}`

#### AC-MCP-8: file_status
- No parameters
- Returns: `{doc_count, chunk_count, last_run_at, embeddings_provider, metadata_fields, health: {fts_available, reranker_enabled, reranker_responsive, last_index_failed_count, last_index_warnings}}`

#### AC-MCP-9: file_index_update
- No parameters, blocking operation
- Runs full indexing pipeline
- Returns: `{status: "completed"|"completed_with_errors", elapsed_seconds, doc_count, chunk_count, last_run_at, failed_count?, failed_docs?}`

#### AC-MCP-10: Taxonomy CRUD
- `file_taxonomy_list(kind?, status?)` → list of entries
- `file_taxonomy_get(id)` → single entry
- `file_taxonomy_search(query, kind?, top_k?)` → semantic search results
- `file_taxonomy_add(kind, name, description, ...)` → created entry
- `file_taxonomy_update(id, ...)` → updated entry
- `file_taxonomy_delete(id)` → `{deleted: true}`
- `file_taxonomy_import()` → import stats from SQLite seed DBs

### 3.3 Hybrid Search (`search_hybrid.py`)

#### AC-SEARCH-1: hybrid_search()
- Step 1: Build WHERE clause from pre-filters
- Step 2: Parallel vector (ANN) + keyword (BM25/tantivy) retrieval
- Step 3: Reciprocal Rank Fusion (k=60)
- Step 4: Length normalization (log-based penalty, anchor=800 chars)
- Step 5: Importance weighting (configurable field + weight)
- Step 6: Optional recency boost + time decay with floor
- Step 7: Optional cross-encoder reranking (60/40 blend; cosine fallback on failure)
- Step 8: MMR diversity filtering (cosine threshold=0.85)
- Step 9: Minimum score threshold (discard noise below cutoff)
- Step 10: Return SearchResult with hits[] and diagnostics{}

#### AC-SEARCH-2: Graceful Degradation
- Vector search failure → keyword-only, diagnostics.vector_search_active=false
- Keyword search failure → vector-only, diagnostics.keyword_search_active=false
- Reranker failure → cosine fallback rerank (70/30 blend with original scores), diagnostics.reranker_applied=false
- Cosine fallback failure → original fusion order preserved
- Any failure → diagnostics.degraded=true

#### AC-SEARCH-3: Pre-filtering
- SQL WHERE clause built from: source_type, folder, tags, status, doc_id_prefix, enr_doc_type, enr_topics, metadata_filters
- Applied at LanceDB level before ANN/FTS scoring (prefilter=True)
- Comma-separated fields use LIKE '%value%' matching

#### AC-SEARCH-4: Length Normalization
- Formula: `score *= 1 / (1 + 0.5 * log2(len/anchor))` where anchor=800 chars
- Chunks at or below anchor length are unaffected (penalty is zero or negligible)
- Applied after RRF fusion (Step 4), before recency boost and reranking

#### AC-SEARCH-5: Time Decay Floor
- Activated when `prefer_recent=true` (default: false)
- Defaults: `half_life_days=90`, `weight=0.3`
- Multiplicative decay: `factor = 0.5 + 0.5 * exp(-age_days / half_life)`
- Floor at 0.5x — old documents never lose more than 50% relevance
- Additive recency bonus: `bonus = weight * exp(-age_days / half_life)`
- Combined: `score = score * decay_factor + recency_bonus`

#### AC-SEARCH-6: Cross-Encoder Blend
- Reranker blends 60% reranker score + 40% original fusion score
- Both score sets normalized to [0,1] before blending
- Preserves retrieval-stage signal; prevents cross-encoder from fully overriding fusion

#### AC-SEARCH-7: Cosine Similarity Fallback
- When cross-encoder fails, falls back to cosine similarity rerank
- Blend: 70% original score + 30% cosine similarity with query vector
- Looks up vectors by chunk_uid format `{doc_id}::{loc}` via `get_vector()` on LanceDBStore
- Falls back silently to original order if vectors cannot be retrieved

#### AC-SEARCH-8: MMR Diversity
- Greedy selection: for each candidate, compute cosine similarity with already-selected results
- If similarity > 0.85 to any selected result, defer (append at end of results list)
- Prevents near-duplicate chunks from consuming all top-K slots
- Looks up vectors by chunk_uid format `{doc_id}::{loc}` via `get_vector()` on LanceDBStore

#### AC-SEARCH-9: Importance Weighting
- Formula: `score *= (1 - weight + weight * importance)` where importance is [0, 1]
- Reads from configurable metadata field (default: `importance`); checked on hit attributes then `extra_metadata`
- Missing or non-numeric values default to 0.5 (neutral — no boost or penalty)
- Values outside [0, 1] clamped to range
- Default weight=0.3: importance=1.0 → 1.0x, importance=0.5 → 0.85x, importance=0.0 → 0.7x
- Config: `search.importance.field` (default: "importance"), `search.importance.weight` (default: 0.3)
- Applied after length normalization (Step 5), before recency boost

#### AC-SEARCH-10: Minimum Score Threshold
- Discards results with score below configurable threshold
- Default: 0.0 (disabled — all results pass through)
- Config: `search.min_score_threshold` (default: 0.0)
- Applied after all scoring and diversity steps (Step 9), before final top_k
- Typical tuning range: 0.01–0.05 for RRF-based scores

### 3.4 Indexing Pipeline (`flow_index_vault.py`)

#### AC-INDEX-1: Scan & Diff
- Glob-based file discovery with include/exclude patterns
- Symlink following with deduplication by real path
- mtime-based change detection (new, modified, deleted)

#### AC-INDEX-2: Text Extraction
- Markdown: YAML frontmatter parsing, heading-aware section splitting
- PDF: PyMuPDF text extraction + OCR fallback for scanned pages
- Images: OCR text extraction + visual description

#### AC-INDEX-3: Chunking
- SentenceSplitter with configurable max_chars and overlap
- Heading-aware splitting for Markdown (preserves section boundaries)
- Page-aware splitting for PDFs (page:chunk locator format)
- Contextual headers prepended before embedding

#### AC-INDEX-4: Enrichment
- LLM extracts: enr_summary, enr_doc_type, enr_entities_*, enr_topics, enr_keywords, enr_key_facts, enr_suggested_tags, enr_suggested_folder
- Taxonomy-aware: active taxonomy entries injected into LLM prompt
- Head+tail sampling for documents exceeding max_input_chars

#### AC-INDEX-5: Storage
- Upsert TextNodes into LanceDB with full metadata
- Schema evolution: new fields auto-added, old fields preserved
- FTS index rebuild after changes
- Auto-recovery on LanceDB corruption (version rollback)

#### AC-INDEX-6: Metadata
- index_metadata.json written after each run
- Fields: last_run_at, doc_count, chunk_count, failed_count, warnings

### 3.5 Configuration (`core/config.py`)

#### AC-CONFIG-1: load_config()
- Required: documents_root (or vault_root alias), index_root
- documents_root must exist on filesystem
- vault_root normalized to documents_root
- Environment variable overrides: DOCUMENTS_ROOT, INDEX_ROOT

#### AC-CONFIG-2: Validation
- chunking.max_chars > 0
- chunking.overlap >= 0 and < max_chars
- search.vector_top_k, keyword_top_k, final_top_k > 0

### 3.6 Vector Store (`lancedb_store.py`)

#### AC-STORE-1: LanceDBStore
- Implements StorageInterface protocol
- Vector search (ANN), keyword search (tantivy BM25), scalar index on doc_id
- Upsert with deduplication by chunk_uid
- Delete by doc_id (removes all chunks for document)
- Schema evolution: new metadata fields auto-promoted to columns
- `get_vector(chunk_uid: str) -> list[float] | None`: Retrieve stored embedding by chunk UID (format: `{doc_id}::{loc}`, matched against `id` column); used by MMR diversity (AC-SEARCH-8) and cosine fallback reranking (AC-SEARCH-7)

#### AC-STORE-2: Facets
- Aggregate counts for: tags, folders, source_types, statuses, doc_types, authors, topics, keywords, entities
- Comma-separated fields split and counted individually

### 3.7 Taxonomy (`taxonomy_store.py`)

#### AC-TAX-1: TaxonomyStore
- LanceDB-backed CRUD for entries with kind (tag/folder/doc_type)
- Embedded descriptions for semantic search
- Alias support (comma-separated, searched during lookup)
- format_for_prompt() renders active entries as text for LLM injection

### 3.8 Provider System (`providers/`)

#### AC-PROV-1: Embed Providers
- Protocol: `embed(texts: list[str]) -> list[list[float]]`
- Implementations: OpenRouter, Ollama, Baseten, LlamaIndex wrapper

#### AC-PROV-2: OCR Providers
- Protocol: `extract(image_path) -> str`, `describe(image_path) -> str`
- Implementations: Gemini Vision, DeepSeek OCR2 (local)

#### AC-PROV-3: LLM Providers
- Protocol: `generate(prompt: str) -> str`
- Implementations: OpenRouter, Ollama, Baseten

---

## 4. Error Handling Patterns

- **Standard error dict:** `{error: true, code: str, message: str, fix?: str}`
- **Diagnostics in search:** `{vector_search_active, keyword_search_active, reranker_applied, degraded}`
- **Index metadata:** `failed_count`, `warnings[]` in index_metadata.json
- **Auto-recovery:** LanceDB version rollback on corruption detection

---

## 5. Non-Functional Requirements

- **Latency:** Search should complete in < 5s for typical queries (excluding reranker timeout of 30s)
- **Throughput:** Indexing should handle 1000+ document vaults
- **Resilience:** Graceful degradation when any single provider is unavailable
- **Security:** Path traversal protection on all file operations, auth on all endpoints

---

## 6. Change History

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2026-03-08 | 1.0 | Initial creation | Claude |
| 2026-03-08 | 1.1 | Added 5 search enhancements (length normalization, MMR diversity, cross-encoder blend, cosine fallback, time decay floor) inspired by memory-lancedb-pro | Claude |
| 2026-03-08 | 1.2 | Added importance weighting (AC-SEARCH-9) and minimum score threshold (AC-SEARCH-10); pipeline now 10-step | Claude |
