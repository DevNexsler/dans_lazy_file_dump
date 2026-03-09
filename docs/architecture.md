# Architecture Reference

Developer-facing details about the search pipeline, storage schema, taxonomy system, providers, and configuration.

## Env vars

| Variable             | When needed                                              |
|----------------------|----------------------------------------------------------|
| `GEMINI_API_KEY`     | Default cloud config — OCR via Gemini Vision (not needed if using local DeepSeek OCR2) |
| `OPENROUTER_API_KEY` | Default cloud config — embeddings (Qwen3-Embedding-8B) + enrichment (GPT-4.1 Mini) |
| `DEEPINFRA_API_KEY`  | Default cloud config — reranker (Qwen3-Reranker-8B via DeepInfra) |

Store these in a `.env` file in the project root. The MCP server and indexer load it automatically.

## Key components

| What              | Component                                                     |
|-------------------|---------------------------------------------------------------|
| Chunking          | Heading-aware (MD) + page-aware (PDF) + `SentenceSplitter`. Semantic fallback via Ollama 0.6B for large sections. Contextual headers on all chunks. |
| Semantic chunking | Qwen3-Embedding-0.6B via Ollama (local). Detects topic boundaries for sections >3600 chars. |
| LLM enrichment   | GPT-4.1 Mini via OpenRouter. Structured JSON schema output. Extracts summary, doc_type, entities, topics, keywords, key_facts, suggested_tags, suggested_folder per document. |
| Taxonomy          | LanceDB table with embedded descriptions. Tags, folders, doc_types with vector search + FTS. Injected into LLM prompt for consistent classification. 7 MCP CRUD tools. |
| Embeddings        | Qwen3-Embedding-8B via OpenRouter (batch_size: 64, concurrency: 2) |
| Vector store      | LanceDB (`LanceDBStore` — direct search with native pre-filters) |
| Full-text search  | LanceDB + tantivy (BM25)                                     |
| OCR (PDF pages)   | Gemini Vision (cloud, default) or DeepSeek OCR2 (local) — text extraction from scanned PDFs/images |
| Image description | Gemini Vision `describe()` — text + detailed visual description (when Gemini OCR enabled) |
| Image metadata    | Pillow EXIF extraction (camera, date, GPS, dimensions)        |
| PDF metadata      | PyMuPDF (title, author, dates, page count)                    |
| Reranking         | Qwen3-Reranker-8B via DeepInfra (cloud, always-on, batch scoring, 30s timeout) |
| Orchestration     | Prefect 3.x — flow/task logging, retry, dashboard at `http://127.0.0.1:4200` |

## Search pipeline

1. **Pre-filter:** Build SQL WHERE clause from filters (source_type, folder, tags, enr_doc_type, enr_topics, etc.) and apply at LanceDB level via `.where(clause, prefilter=True)` — filters run before ANN/FTS scoring so full `top_k` results match the criteria
2. **Parallel retrieval:** vector search (semantic) + BM25/FTS (keyword) — run concurrently, both with pre-filters applied
3. **RRF fusion:** Reciprocal Rank Fusion merges both ranked lists (k=60)
4. **Length normalization:** Log-based penalty for long chunks — `score *= 1 / (1 + 0.5 * log2(len/anchor))` with anchor=800 chars. Chunks at or below anchor are unaffected.
5. **Importance weighting:** Boosts docs with higher importance/priority metadata — `score *= (0.7 + 0.3 * importance)`. Reads configurable metadata field (default: `importance`). Missing values treated as neutral (0.5).
6. **Recency boost (optional):** When `prefer_recent=true`, multiplicative time decay with floor (`0.5 + 0.5 * exp(-age/half_life)`) + additive bonus. Old docs keep ≥50% score.
7. **Cross-encoder reranking:** Qwen3-Reranker-8B (DeepInfra) with 60/40 blend — 60% reranker score + 40% original fusion score (both normalized to [0,1]). On failure, falls back to cosine similarity rerank (70% original + 30% cosine) instead of giving up.
8. **MMR diversity:** Greedy selection defers near-duplicate chunks (cosine similarity > 0.85) to end of results list, preventing redundancy in top-K.
9. **Minimum score threshold:** Discards results below configurable threshold (default: 0.0 = disabled). Filters noise from low-relevance matches.
10. **Diagnostics:** Every response includes `{vector_search_active, keyword_search_active, reranker_applied, degraded}` so callers detect silent degradation

## Taxonomy system

The taxonomy provides a controlled vocabulary for consistent document classification. It lives in a separate LanceDB table (`taxonomy`) alongside the main `chunks` table.

### How it works

1. **Taxonomy entries** (tags, folders, doc_types) have embedded descriptions for semantic matching
2. During enrichment, `taxonomy_store.format_for_prompt()` renders active entries as text injected into the LLM prompt
3. The LLM picks from available tags/folders (or suggests new ones) for `suggested_tags` and `suggested_folder`
4. After enrichment, `usage_count` is incremented for matched entries
5. Managed via 7 MCP tools (`file_taxonomy_*`) — no file editing needed

### Taxonomy table schema

| Column | Type | Example |
|--------|------|---------|
| `id` | string (PK) | `"tag:machine-learning"` |
| `kind` | string | `"tag"`, `"folder"`, `"doc_type"` |
| `name` | string | `"machine-learning"` |
| `description` | string | `"ML models, training, inference..."` |
| `aliases` | string | `"ml, deep-learning"` (comma-separated) |
| `parent` | string | `""` (for hierarchy) |
| `status` | string | `"active"` or `"archived"` |
| `usage_count` | int | `3` |
| `ai_managed` | int | `1` (1=AI, 0=human) |
| `contents_type` | string | `"notes"`, `"data"`, etc. (folders only) |
| `created_by` | string | `"AI"` or human name |
| `vector` | float[] | Embedded description |

### Seed data

The taxonomy can be seeded from existing SQLite databases via `file_taxonomy_import` or `scripts/seed_taxonomy.py`:
- `tags.db` — 21 tags with descriptions and usage counts
- `directory.db` — 62 folders with descriptions, purposes, and ai_managed flags

## LanceDB schema (per chunk)

| Field | Type | Description |
|-------|------|-------------|
| `chunk_uid` | string | Unique ID: `doc_id::loc` |
| `doc_id` | string | Path-relative to documents root (e.g. `Projects/notes.md`) |
| `loc` | string | Location within doc (`c:0`, `p:2:c:1`, `img:c:0`) |
| `text` | string | Contextual header + extracted text (header aids search retrieval) |
| `snippet` | string | First ~200 chars of raw text (without header) for clean display |
| `section` | string | Heading breadcrumb for MD chunks (e.g. `Setup > Prerequisites`) |
| `source_type` | string | `md`, `pdf`, or `img` |
| `mtime` | float | Source file's last-modified time |
| `size` | int | Source file size in bytes |
| `title` | string | From YAML frontmatter, first `# heading`, or filename |
| `tags` | string | From YAML frontmatter, comma-separated (e.g. `"recipe,korean"`) |
| `folder` | string | Top-level directory from document path |
| `status` | string | From YAML frontmatter (e.g. `active`, `archived`) |
| `created` | string | From YAML frontmatter (e.g. `2026-01-15`) |
| `description` | string | From YAML frontmatter `description` field |
| `author` | string | From YAML frontmatter `author` field |
| `keywords` | string | From YAML frontmatter, comma-separated. Separate from LLM `enr_keywords`. |
| `custom_meta` | string | JSON dict of remaining frontmatter fields |
| `enr_summary` | string | LLM-generated document summary (empty if not enriched) |
| `enr_doc_type` | string | Comma-separated document types (e.g. `report,engineering`) |
| `enr_entities_people` | string | Comma-separated person names |
| `enr_entities_places` | string | Comma-separated locations/addresses |
| `enr_entities_orgs` | string | Comma-separated organization names |
| `enr_entities_dates` | string | Comma-separated dates (YYYY-MM-DD) |
| `enr_topics` | string | Comma-separated high-level topics |
| `enr_keywords` | string | Comma-separated specific terms and phrases |
| `enr_key_facts` | string | Comma-separated key facts/conclusions |
| `enr_suggested_tags` | string | Comma-separated tags suggested by LLM (from taxonomy when available) |
| `enr_suggested_folder` | string | Best folder path suggested by LLM (from taxonomy when available) |
| `vector` | float[] | Embedding vector (2560-dim for Qwen3, 768-dim for Gemini) |

Metadata fields (`title`, `tags`, `status`, `created`, `description`, `author`, `keywords`, `folder`) are automatically enriched during indexing from YAML frontmatter and file path. LLM enrichment fields (prefixed `enr_`) are extracted by GPT-4.1 Mini via OpenRouter during indexing when `enrichment.enabled` is true. The `enr_` prefix avoids collisions with user frontmatter fields. All metadata is returned in search results and can be used as filters. Dynamic metadata fields from frontmatter (e.g. `priority`, `category`) are automatically promoted to LanceDB columns and appear in search results, facets, and filters.
