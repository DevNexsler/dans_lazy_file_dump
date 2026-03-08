# Architecture Diagram

## Current State (Local Mac)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Local Machine                            │
│                                                                 │
│  Documents Folder            ┌──────────────┐                   │
│  (/Users/.../Primary/)       │  Prefect     │                   │
│  ├── Markdown (.md)          │  Orchestrator│                   │
│  ├── PDFs                    │  (temp server)│                  │
│  └── Images (.png/.jpg)      └──────┬───────┘                   │
│           │                         │                           │
│           ▼                         ▼                           │
│  ┌─────────────────────────────────────────┐                    │
│  │              Indexing Pipeline           │                    │
│  │                                         │                    │
│  │  1. Scan & diff (incremental, mtime)    │                    │
│  │  2. Extract text (PyMuPDF, EXIF, regex) │                    │
│  │  3. OCR scanned pages ──────────────────┼──► DeepSeek OCR2   │
│  │  4. LLM enrichment ────────────────────┼──► OpenRouter       │
│  │     (summary, entities, topics,         │    (GPT-4.1 Mini)  │
│  │      suggested_tags, suggested_folder)  │                    │
│  │        ▲                                │                    │
│  │        │ taxonomy context               │                    │
│  │  ┌─────┴──────┐                         │                    │
│  │  │  Taxonomy   │                         │                    │
│  │  │  Store      │ (21 tags, 62 folders)   │                    │
│  │  └────────────┘                         │                    │
│  │  5. Chunk (heading/page-aware)          │                    │
│  │  6. Semantic split ─────────────────────┼──► Ollama (0.6B)   │
│  │  7. Embed ──────────────────────────────┼──► OpenRouter       │
│  │     (Qwen3-Embedding-8B)               │                    │
│  └─────────────────┬───────────────────────┘                    │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │           LanceDB (local disk)          │                    │
│  │                                         │                    │
│  │  ┌─────────────┐  ┌──────────────────┐  │                    │
│  │  │ chunks table │  │ taxonomy table   │  │                    │
│  │  │ vectors      │  │ embedded descs   │  │                    │
│  │  │ metadata     │  │ tags/folders/    │  │                    │
│  │  │ BM25 (FTS)   │  │  doc_types      │  │                    │
│  │  └─────────────┘  └──────────────────┘  │                    │
│  └─────────────────┬───────────────────────┘                    │
│                    │                                            │
│                    ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │         MCP Server (17 tools)           │                    │
│  │                                         │                    │
│  │  Document tools (9)  Taxonomy tools (7) │                    │
│  │  ├─ file_search      ├─ file_taxonomy_  │                    │
│  │  ├─ file_get_chunk   │   list/get/      │                    │
│  │  ├─ file_get_doc_    │   search/add/    │                    │
│  │  │   chunks          │   update/delete/ │                    │
│  │  ├─ file_list_docs   │   import         │                    │
│  │  ├─ file_recent      │                  │                    │
│  │  ├─ file_facets      Index tool (1)     │                    │
│  │  ├─ file_folders     ├─ file_index_     │                    │
│  │  └─ file_status      │   update         │                    │
│  └──────────┬──────────────────────────────┘                    │
│             │ stdio or HTTP (:7788)                             │
└─────────────┼──────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────┐
│     AI Assistants        │
│  ├─ Claude Code          │
│  ├─ OpenClaw             │
│  ├─ Claude Desktop       │
│  └─ Cursor / Windsurf    │
└──────────────────────────┘
```

## Search Pipeline (per query)

```
User query: "property insurance claim in Springfield"
                │
                ▼
┌───────────────────────────────────────────────────────┐
│  1. PRE-FILTER                                        │
│     Build SQL WHERE from params:                      │
│     folder="Legal", tags="insurance", source_type=pdf │
│     Applied at LanceDB level (prefilter=True)         │
└───────────────────┬───────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│ 2a. VECTOR      │  │ 2b. KEYWORD     │
│ (semantic)      │  │ (BM25/FTS)      │
│                 │  │                 │
│ Embed query via │  │ tantivy FTS     │
│ OpenRouter      │  │ index search    │
│ → ANN search    │  │                 │
│ top_k results   │  │ top_k results   │
└────────┬────────┘  └────────┬────────┘
         │    run in parallel  │
         └─────────┬──────────┘
                   ▼
┌───────────────────────────────────────┐
│  3. RRF FUSION                        │
│     Reciprocal Rank Fusion merges     │
│     both ranked lists (k=60)          │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│  4. CROSS-ENCODER RERANKING           │
│     Qwen3-Reranker-8B (DeepInfra)    │
│     Scores each query-doc pair        │
│     Re-sorts by relevance score       │
│     (timeout: 30s, graceful fallback) │
└───────────────────┬───────────────────┘
                    ▼
┌───────────────────────────────────────┐
│  5. RESPONSE + DIAGNOSTICS            │
│     Results with metadata + scores    │
│     + { keyword_search_active: true,  │
│         reranker_applied: true,       │
│         degraded: false }             │
└───────────────────────────────────────┘
```

## Enrichment + Taxonomy Flow (per document)

```
New document text
        │
        ▼
┌───────────────────────────────────┐
│  Taxonomy Store                   │
│  format_for_prompt()              │
│  ┌─────────────────────────────┐  │
│  │ Available Tags:             │  │
│  │  work — Work-related items  │  │
│  │  legal — Legal documents    │  │
│  │  insurance — Insurance docs │  │
│  │  ...21 tags                 │  │
│  │ Available Folders:          │  │
│  │  2-Area/Legal/              │  │
│  │  2-Area/Finance/            │  │
│  │  ...62 folders              │  │
│  └─────────────────────────────┘  │
└───────────────┬───────────────────┘
                │ injected into prompt
                ▼
┌───────────────────────────────────────────┐
│  GPT-4.1 Mini (OpenRouter)                │
│  Structured JSON schema output            │
│                                           │
│  {                                        │
│    "summary": "...",                      │
│    "doc_type": ["report", "legal"],       │
│    "entities_people": ["John Smith"],     │
│    "entities_places": ["Springfield"],    │
│    "topics": ["insurance", "claims"],     │
│    "keywords": ["hailstorm", "roof"],     │
│    "key_facts": ["$15,000 damage"],       │
│    "suggested_tags": ["insurance","legal"],│  ◄── from taxonomy
│    "suggested_folder": "2-Area/Legal/"    │  ◄── from taxonomy
│  }                                        │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────┐
│  Post-enrichment                  │
│  • Normalize to enr_* fields     │
│  • Increment usage_count for     │
│    matched taxonomy entries       │
│  • Store in LanceDB chunks table  │
└───────────────────────────────────┘
```

## VPS Target Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   VPS / Render.com                        │
│                                                          │
│   ┌────────────────────────────────────────────────┐     │
│   │              server.py (single process)         │     │
│   │                                                │     │
│   │  ┌──────────────────┐  ┌────────────────────┐  │     │
│   │  │   MCP Server     │  │  REST API (Ph.2)   │  │     │
│   │  │   (HTTP/SSE)     │  │  /api/upload       │  │     │
│   │  │   17 tools       │  │  /api/search       │  │     │
│   │  │                  │  │  /api/status        │  │     │
│   │  └────────┬─────────┘  └────────┬───────────┘  │     │
│   │           │     Auth: Bearer $API_KEY           │     │
│   │           └──────────┬──────────┘               │     │
│   │                      ▼                          │     │
│   │  ┌─────────────────────────────────────────┐    │     │
│   │  │           Core Engine                    │    │     │
│   │  │  indexer / search / enrichment /         │    │     │
│   │  │  taxonomy                                │    │     │
│   │  └──────────────────┬──────────────────────┘    │     │
│   └─────────────────────┼──────────────────────────┘     │
│                         ▼                                │
│   ┌─────────────────────────────────────────┐            │
│   │      Persistent Disk (/data)            │            │
│   │                                         │            │
│   │   /data/documents/  ← uploaded/synced   │            │
│   │   /data/index/      ← LanceDB tables    │            │
│   │     ├── chunks (vectors + metadata)     │            │
│   │     └── taxonomy (tags + folders)       │            │
│   └─────────────────────────────────────────┘            │
│                                                          │
│   ⚠ Single instance (Render Disk = 1 container)         │
│   Future: S3-backed LanceDB for horizontal scaling       │
└──────────────────────────────────────────────────────────┘
         │                          │
         │ HTTP/SSE (:7788)         │ HTTPS API calls
         ▼                          ▼
┌──────────────────┐    ┌──────────────────────────┐
│  AI Assistants   │    │  Cloud APIs              │
│  (remote MCP)    │    │  ├─ OpenRouter           │
│  ├─ Claude Code  │    │  │  (embed + enrich)     │
│  ├─ OpenClaw     │    │  ├─ DeepInfra (reranker) │
│  └─ Claude Desktop│   │  └─ Gemini (OCR)         │
└──────────────────┘    └──────────────────────────┘
```

### Render.com Specifics

```
Render handles:                     We handle:
├─ Container lifecycle              ├─ Single server.py process
├─ Zero-downtime deploys            ├─ Auth (API_KEY bearer token)
├─ Health checks + auto-restart     ├─ LanceDB on Render Disk
├─ TLS termination                  ├─ Config via env vars
├─ Env var management               └─ Backups (optional rsync to S3)
├─ Git push → auto-deploy
└─ Logging

Render does NOT handle:
├─ Horizontal scaling with local disk (Render Disk = 1 instance)
├─ Built-in DB (we use LanceDB, not Render's Postgres)
└─ Cron jobs for indexing (use Render Cron Job or external trigger)
```

### Scaling Path (if ever needed)

```
Phase 1: Single instance + Render Disk       ← start here
         (handles thousands of documents)
              │
              ▼ (only if needed)
Phase 2: S3-backed LanceDB (lance + S3/R2)
         Multiple instances, shared storage
         Render auto-scaling kicks in
```
