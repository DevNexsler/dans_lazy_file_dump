# VPS Architecture — Cloud-Native Document Organizer

## Goals

1. **No Obsidian dependency** — works with any markdown source
2. **VPS-ready** — runs on any Linux server with persistent disk
3. **Stateless compute, persistent data** — server can restart without data loss

---

## What's Already Done (VPS-Version branch)

These items from the original plan are **complete**:

- [x] `documents_root` config key (replaces `vault_root`, with fallback alias)
- [x] All MCP tools renamed from `vault_*` to `file_*`
- [x] Cloud providers (OpenRouter embeddings + enrichment, Baseten reranker) — no local GPU needed
- [x] Taxonomy system for consistent tagging (7 MCP CRUD tools)
- [x] MCP HTTP mode already works (`--http` flag)
- [x] 358 tests passing
- [x] **Phase 1: Deployable Server** — env var overrides, Bearer token auth, `server.py` entrypoint, Dockerfile, Render.com config

---

## Current Architecture

```
Local Machine / VPS
  Documents folder (any path)
       |
  [Indexer] → LanceDB (local .lance/ files)
       |            + Taxonomy table
  [MCP Server] ← AI assistants (stdio or HTTP)
       |
  Cloud APIs:
    OpenRouter (embeddings + enrichment)
    Baseten (reranker)
    Gemini or DeepSeek OCR2 (OCR)
```

The core engine is already cloud-API-only (no local GPU). The remaining VPS work is just **deployment packaging** — how to run it remotely, accept files, and secure it.

---

## Remaining Work — 3 Phases

### Phase 1: Deployable Server (DONE)

**Goal:** Run on a VPS, accept MCP connections over HTTP, serve from persistent disk.

| Task | Status | Details |
|------|--------|---------|
| **`server.py`** | Done | Unified entrypoint: starts MCP HTTP on `$PORT` (default 7788) |
| **`Dockerfile`** | Done | Python 3.13-slim, filters out macOS-only deps (mlx-*) |
| **Auth middleware** | Done | ASGI middleware in `mcp_server.py` — checks `Authorization: Bearer <API_KEY>` env var. No auth when unset. |
| **Env var overrides** | Done | `DOCUMENTS_ROOT`, `INDEX_ROOT`, `PORT` override config.yaml in `core/config.py` |
| **`config.vps.yaml.example`** | Done | VPS paths + S3 URI examples |
| **`render.yaml`** | Done | Render.com deployment descriptor with persistent disk |

**Usage:** `docker build -t doc-organizer . && docker run -v /tmp/data:/data -p 7788:7788 -e API_KEY=secret doc-organizer`

#### LanceDB S3 Storage (config-only, no code changes)

LanceDB v0.29.2 (already installed) natively supports `s3://`, `gs://`, and `az://` URIs. Switching from local disk to shared cloud storage is a **config-only change**:

```yaml
# Just change the URI:
index_root: "s3://my-bucket/doc-organizer-index"

# Add credentials:
storage_options:
  aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
  aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
  region: "us-east-1"

# For read consistency across instances:
lancedb:
  read_consistency_interval_seconds: 5
```

Both `lancedb_store.py` and `taxonomy_store.py` call `lancedb.connect(index_root)` — S3 URIs work transparently. The only code addition needed is passing `storage_options` and `read_consistency_interval` from config to the `lancedb.connect()` calls.

### Phase 2: Document Ingestion API

**Goal:** Accept files from external sources (upload, git sync, web UI).

| Task | What to do | Complexity |
|------|-----------|------------|
| **`api_server.py`** | FastAPI app with REST routes. Mount alongside MCP server in `server.py`. | Medium |
| `POST /api/upload` | Multipart file upload → save to `documents_root` → trigger incremental index | Medium |
| `POST /api/sync` | Trigger `file_index_update` (re-scan documents_root) | Small — calls existing flow |
| `GET /api/search` | REST wrapper around `file_search` for non-MCP clients | Small |
| `GET /api/status` | REST wrapper around `file_status` | Small |
| `DELETE /api/documents/{doc_id}` | Remove file from documents_root + delete from index | Small |

**Note:** The REST API is optional. MCP HTTP already provides full functionality for AI assistants. The REST API is for web UIs and scripts that don't speak MCP.

### Phase 3: Platform Configs (as needed)

| Platform | Config file | Status |
|----------|------------|--------|
| **Render.com** | `render.yaml` | Done (Phase 1) |
| **Docker** | `Dockerfile` | Done (Phase 1) |
| **Docker Compose** | `docker-compose.yml` | Add when needed |
| **Systemd** | `doc-organizer.service` | Add when needed |
| **Fly.io** | `fly.toml` | Add when needed |

---

## Design Principles

**Keep it simple:**
- One process serves everything (MCP + optional REST API)
- LanceDB is the only data store (no Postgres, no Redis, no external DBs)
- Cloud APIs handle all ML inference (no GPU, no model downloads on server)
- Persistent disk at `/data` — documents + index side by side

**Evolve incrementally:**
- Phase 1 is a weekend project — it's just Docker + auth
- Phase 2 adds ingestion only when you need non-filesystem sources
- Phase 3 is platform-specific glue, done only for platforms you actually deploy to

**Don't over-engineer:**
- No multi-tenant auth until you actually have multiple users
- No S3-backed LanceDB until you need multi-instance scaling
- No message queue for indexing until throughput demands it
- No Kubernetes — a single Docker container is fine for thousands of documents

---

## Storage

LanceDB stores everything as local `.lance/` files. This must survive container restarts.

| Platform | Approach |
|----------|---------|
| **Any VPS** | Regular disk at `/data` (ext4/ZFS). Back up with rsync/restic. |
| **Render.com** | Render Disk (persistent SSD at `/data`). Survives deploys. |
| **Fly.io** | Fly Volume (NVMe). |
| **Docker** | Bind mount or named volume at `/data`. |

```yaml
# config.vps.yaml
documents_root: "/data/documents"
index_root: "/data/index"
```

---

## Environment Variables (VPS)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Embeddings + enrichment |
| `BASETEN_API_KEY` | Yes | Reranker |
| `GEMINI_API_KEY` | No | OCR (only if using Gemini instead of DeepSeek OCR2) |
| `API_KEY` | Yes (VPS) | Bearer token auth for HTTP access |
| `DOCUMENTS_ROOT` | No | Override config (default: from config.yaml) |
| `INDEX_ROOT` | No | Override config (default: from config.yaml) |
| `PORT` | No | Server port (default: 7788) |

---

## What Stays the Same (no changes needed)

- LanceDB as the storage engine
- All cloud providers (already stateless API calls)
- Hybrid search pipeline (vector + BM25 + RRF + reranker)
- LLM enrichment + taxonomy integration
- Frontmatter extraction (works with any YAML frontmatter)
- Chunking strategies
- Full test suite

---

## Ingestion Methods (Phase 2+)

| Method | How it works | When to add |
|--------|-------------|-------------|
| **Filesystem scan** | Point `documents_root` at a folder, run indexer | Already works (Phase 1) |
| **Upload API** | `POST /api/upload` with file | Phase 2 |
| **Git sync** | Clone/pull a repo to `documents_root` on schedule | Phase 2 (cron + git pull) |
| **Watch directory** | inotify/fswatch triggers re-index | Phase 2 (optional) |
| **S3/R2 sync** | Pull from bucket on schedule | Future (if needed) |

---

## Compatible Software

Works with any tool that produces markdown with optional YAML frontmatter:

Obsidian, HackMD/CodiMD, Notion (export), Logseq, GitBook, Typora, iA Writer, VS Code, Google Docs (export), Jekyll, Hugo — all use standard `---` delimited YAML frontmatter.
