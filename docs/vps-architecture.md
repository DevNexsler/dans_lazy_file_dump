# VPS Architecture — Cloud-Native Document Organizer

## Goals

1. **No Obsidian dependency** — works with any markdown source (HackMD, Notion exports, GitBook, plain `.md` files, Google Docs exports, etc.)
2. **VPS / Render.com ready** — all services run in cloud with persistent storage
3. **Collaborative** — multiple users can feed documents; not tied to a single local vault
4. **Stateless compute, persistent data** — app servers can restart/redeploy without data loss

---

## Current State (Local)

```
Local Machine
  Obsidian Vault (local folder)
       |
  [Indexer] → LanceDB (local folder)
       |
  [MCP Server] ← AI assistants (stdio)
```

**Problems for VPS deployment:**
- `vault_root` / `index_root` are local filesystem paths
- LanceDB index is a local directory (`.lance/` files)
- MCP server uses stdio transport (requires co-located process)
- Prefect runs a temp local server
- Config references "Obsidian vault" everywhere
- No document ingestion API — relies on filesystem scanning

---

## Target Architecture (VPS)

```
┌─────────────────────────────────────────────────────────┐
│                     VPS / Render.com                     │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Web API      │    │  MCP Server  │ (SSE transport)   │
│  │  (FastAPI)    │    │  (HTTP mode) │                   │
│  │  /upload      │    │              │                   │
│  │  /search      │    └──────┬───────┘                   │
│  │  /status      │           │                           │
│  └──────┬───────┘           │                           │
│         │                    │                           │
│         ▼                    ▼                           │
│  ┌─────────────────────────────────┐                    │
│  │         Core Engine              │                    │
│  │  indexer / search / enrichment   │                    │
│  └──────────────┬──────────────────┘                    │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────────┐                    │
│  │     Persistent Storage           │                    │
│  │  Option A: Render Disk           │                    │
│  │  Option B: S3/R2 + local cache   │                    │
│  │  Option C: VPS disk (ext4/ZFS)   │                    │
│  └─────────────────────────────────┘                    │
│                                                         │
│  Documents dir:  /data/documents/                       │
│  LanceDB index:  /data/index/                           │
└─────────────────────────────────────────────────────────┘

External (cloud APIs — same as current):
  - OpenRouter (embeddings + enrichment)
  - Baseten (reranker)
  - Gemini (OCR)
```

---

## Key Changes from Local Version

### 1. Document Source — Filesystem-Agnostic

**Current:** Scans a local Obsidian vault directory via glob patterns.

**VPS Version:** Documents come from multiple sources. The indexer operates on a **documents directory** (`/data/documents/`) regardless of how files got there.

**Ingestion methods (any combination):**

| Method | How it works | Best for |
|--------|-------------|----------|
| **Upload API** | `POST /api/documents/upload` — multipart file upload | Web UI, scripts, integrations |
| **Git sync** | Clone/pull a git repo of markdown files on a schedule | Teams using git-based MD (HackMD, GitBook, Obsidian Git) |
| **Watch directory** | Mount a volume or sync folder; indexer detects changes | VPS with shared storage |
| **S3/R2 sync** | Pull from object storage bucket on schedule or webhook | Cloud-native workflows |
| **CLI push** | `curl` or client script pushes files to upload API | Quick one-off imports |

The scanner already uses glob patterns — just point `documents_root` at `/data/documents/` instead of an Obsidian vault. No Obsidian-specific logic to remove (frontmatter parsing works on any YAML-frontmatter markdown, which HackMD/Jekyll/Hugo/etc. all use).

### 2. Storage Strategy

LanceDB stores everything as local files (`.lance/` directory). This must survive container restarts and redeployments.

| Platform | Storage approach | Notes |
|----------|-----------------|-------|
| **Render.com** | Render Disk (persistent SSD mounted at `/data`) | Survives deploys. 1 service per disk. |
| **VPS (Hetzner, DigitalOcean, etc.)** | Regular disk (`/data` on ext4/ZFS) | Most straightforward. Back up with rsync/restic. |
| **Fly.io** | Fly Volume (persistent NVMe) | Attached to one machine. |
| **Railway** | Volume mount | Similar to Render Disk. |
| **S3-backed (future)** | LanceDB-Cloud or lance + S3 object store | LanceDB supports remote storage natively. Best for multi-instance. |

**Config change:**
```yaml
# Old
vault_root: "/Users/dan/obsidian-vault"
index_root: "/Users/dan/index"

# New (VPS)
documents_root: "/data/documents"
index_root: "/data/index"
```

### 3. Transport — MCP over HTTP/SSE

**Current:** MCP server uses stdio (process launched by AI assistant locally).

**VPS Version:** MCP server runs as a persistent HTTP service. Already supported via `--http` flag. For remote AI assistants, expose as SSE (Server-Sent Events) transport — the MCP spec supports this.

```
AI Assistant ──(SSE/HTTP)──> VPS:7788 ──> MCP Server ──> LanceDB
```

**Also add:** A REST API layer (FastAPI) for non-MCP clients:
- `POST /api/documents/upload` — upload files
- `POST /api/documents/sync` — trigger git pull / re-scan
- `GET /api/search?q=...&filters=...` — search without MCP
- `GET /api/status` — index health
- `DELETE /api/documents/{doc_id}` — remove a document

### 4. Naming — Drop "Obsidian" / "Vault" Terminology

| Old | New |
|-----|-----|
| `vault_root` | `documents_root` |
| `vault_search` | `doc_search` |
| `vault_status` | `index_status` |
| `vault_recent` | `doc_recent` |
| `vault_facets` | `index_facets` |
| `vault_folders` | `doc_folders` |
| `vault_get_chunk` | `doc_get_chunk` |
| `vault_get_doc_chunks` | `doc_get_chunks` |
| `vault_list_documents` | `doc_list` |
| `vault_index_update` | `index_update` |
| Obsidian Vault Semantic Index | Document Organizer |

Keep backward compatibility: accept both old and new tool names in MCP server during transition.

### 5. Process Management

**Current:** Prefect starts a temp local server. Works for CLI usage.

**VPS Version:**

| Component | How to run |
|-----------|-----------|
| **MCP Server** | Long-running process (`python mcp_server.py --http`) — Render Web Service or `systemd` on VPS |
| **API Server** | FastAPI app (`uvicorn api_server:app`) — can combine with MCP in one process |
| **Indexer** | Triggered via API call, cron, or file-watch. Not a persistent process. |
| **Prefect** | Optional. For VPS, a simple cron + logging may suffice. Keep Prefect support for users who want dashboards. |

**Render.com deployment:**
```
# render.yaml
services:
  - type: web
    name: doc-organizer
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python server.py
    disk:
      name: data
      mountPath: /data
      sizeGB: 10
    envVars:
      - key: OPENROUTER_API_KEY
        sync: false
      - key: BASETEN_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
```

**Docker (VPS):**
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
VOLUME /data
EXPOSE 7788
CMD ["python", "server.py"]
```

### 6. Authentication

Local version has none (localhost only). VPS needs basic auth at minimum.

**Phase 1:** API key in header (`Authorization: Bearer <key>`) — single shared key from env var `API_KEY`.

**Phase 2 (later):** Per-user keys, JWT, or OAuth if multi-tenant.

---

## File Structure (New/Modified)

```
server.py                    NEW — unified entrypoint (starts API + MCP HTTP server)
api_server.py                NEW — FastAPI REST API (upload, search, sync, status)
core/config.py               MODIFY — support documents_root, detect VPS vs local mode
core/storage.py              MODIFY — ensure paths work with /data mount
core/auth.py                 NEW — API key middleware
mcp_server.py                MODIFY — rename tools, keep backward compat aliases
flow_index_vault.py          MODIFY — rename to flow_index.py, accept documents_root
Dockerfile                   NEW
docker-compose.yml           NEW (optional — for multi-container with Prefect)
render.yaml                  NEW — Render.com deploy config
config.vps.yaml.example      NEW — VPS config template
docs/vps-architecture.md     THIS FILE
```

---

## Migration Path

1. **Rename config keys** — `vault_root` → `documents_root` (keep `vault_root` as fallback alias)
2. **Rename MCP tools** — add new names, keep old as aliases
3. **Add `server.py`** — unified HTTP entrypoint combining MCP + REST API
4. **Add `api_server.py`** — FastAPI routes for upload/search/sync
5. **Add auth middleware** — API key from env var
6. **Add Dockerfile + render.yaml** — containerized deployment
7. **Add `config.vps.yaml.example`** — cloud config pointing to `/data/` paths
8. **Update README** — document both local and VPS deployment

---

## What Stays the Same

- LanceDB as the storage engine (works great with local files on persistent disk)
- All cloud providers (OpenRouter, Baseten, Gemini) — already stateless API calls
- Hybrid search pipeline (vector + BM25 + RRF + reranker)
- LLM enrichment pipeline
- Frontmatter extraction (works with any YAML frontmatter, not Obsidian-specific)
- Chunking strategies
- Test suite (point at `/data/` in test config)

---

## Environment Variables (VPS)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Embeddings + enrichment |
| `BASETEN_API_KEY` | Yes | Reranker |
| `GEMINI_API_KEY` | Yes | OCR |
| `API_KEY` | Yes (VPS) | Auth for REST API + MCP HTTP |
| `DOCUMENTS_ROOT` | No | Override config (default: `/data/documents`) |
| `INDEX_ROOT` | No | Override config (default: `/data/index`) |
| `PORT` | No | Server port (default: 7788, Render sets this) |

---

## Collaborative MD Software Compatibility

The system works with any tool that produces standard markdown files with optional YAML frontmatter:

| Software | Integration method |
|----------|-------------------|
| **HackMD / CodiMD** | Git sync (HackMD pushes to git), or export + upload API |
| **Notion** | Export as markdown + upload, or use Notion API → markdown converter |
| **Obsidian** | Git sync (Obsidian Git plugin), or point documents_root at vault |
| **Logseq** | Git sync (Logseq uses git natively) |
| **GitBook** | Git sync (GitBook stores as markdown in git) |
| **Typora / iA Writer** | Sync folder or upload API |
| **Google Docs** | Export as markdown via API/Zapier + upload |
| **VS Code / any editor** | Direct file access or git sync |

The frontmatter parser already handles standard YAML frontmatter (`---` delimited), which is the universal standard across all these tools. No Obsidian-specific syntax (like `[[wikilinks]]`) is required for indexing to work.
