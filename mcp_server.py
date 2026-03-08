"""MCP server: file search, retrieval, browsing, status, and indexing tools.

Single service: both querying and indexing via MCP tools. Same config as indexer."""

import logging
import os
import time
from pathlib import Path

from core.config import load_config
from core.storage import SearchHit
from lancedb_store import LanceDBStore
from providers.embed import build_embed_provider
from search_hybrid import hybrid_search, build_reranker

logger = logging.getLogger(__name__)

# Optional: use FastMCP when mcp package is installed
try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    FastMCP = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SOURCE_TYPES = {"md", "pdf", "img"}
_MAX_TOP_K = 100
_MAX_LIMIT = 200

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(code: str, message: str, fix: str | None = None) -> dict:
    """Build a structured error response for MCP tool callers."""
    err: dict = {"error": True, "code": code, "message": message}
    if fix:
        err["fix"] = fix
    return err


def _validate_source_type(source_type: str | None) -> dict | None:
    """Return an error dict if source_type is invalid, else None."""
    if source_type and source_type not in _VALID_SOURCE_TYPES:
        return _error(
            "invalid_parameter",
            f"source_type must be one of: {', '.join(sorted(_VALID_SOURCE_TYPES))}. Got: '{source_type}'.",
            "Valid values: 'md' (Markdown), 'pdf' (PDF documents), 'img' (images).",
        )
    return None


def _hit_to_dict(h: SearchHit, include_text: bool = False) -> dict:
    """Convert a SearchHit to a response dict."""
    d: dict = {
        "doc_id": h.doc_id,
        "loc": h.loc,
        "snippet": h.snippet,
        "score": h.score,
        "title": h.title,
        "tags": h.tags.split(",") if h.tags else [],
        "folder": h.folder,
        "status": h.status,
        "source_type": h.source_type,
        "description": h.description,
        "author": h.author,
        "keywords": h.keywords.split(",") if h.keywords else [],
        "custom_meta": h.custom_meta,
        "enr_summary": h.enr_summary,
        "enr_doc_type": h.enr_doc_type,
        "enr_topics": h.enr_topics,
        "enr_keywords": h.enr_keywords,
        "enr_entities_people": h.enr_entities_people,
        "enr_entities_places": h.enr_entities_places,
        "enr_entities_orgs": h.enr_entities_orgs,
        "enr_entities_dates": h.enr_entities_dates,
        "enr_key_facts": h.enr_key_facts,
        "enr_suggested_tags": h.enr_suggested_tags,
        "enr_suggested_folder": h.enr_suggested_folder,
    }
    # Include dynamic metadata fields (e.g. section, sentiment)
    for k, v in (h.extra_metadata or {}).items():
        if k not in d:
            d[k] = v
    if include_text:
        d["text"] = h.text
    return d


def _enrich_doc_list(docs: list[dict]) -> None:
    """Add mtime_iso and split comma-separated tags on a list of doc dicts (in-place)."""
    from datetime import datetime, timezone
    for d in docs:
        mtime = d.get("mtime")
        if mtime is not None:
            try:
                d["mtime_iso"] = datetime.fromtimestamp(float(mtime), tz=timezone.utc).isoformat()
            except (ValueError, OSError):
                d["mtime_iso"] = None
        else:
            d["mtime_iso"] = None
        if d.get("tags") and isinstance(d["tags"], str):
            d["tags"] = d["tags"].split(",")


# ---------------------------------------------------------------------------
# Lazy init
# ---------------------------------------------------------------------------


def _build_store_and_embed(config_path: str = "config.yaml"):
    """Load config and build store + embed provider."""
    config = load_config(config_path)
    index_root = Path(config["index_root"])
    table = config.get("lancedb", {}).get("table", "chunks")
    store = LanceDBStore(index_root, table)

    embed_provider = build_embed_provider(config)
    return store, embed_provider, config


_cache: tuple | None = None


def _get_deps(config_path: str = "config.yaml"):
    global _cache
    if _cache is None:
        _cache = _build_store_and_embed(config_path)
    return _cache[0], _cache[1], _cache[2]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _file_search_impl(
    query: str,
    top_k: int = 10,
    doc_id_prefix: str | None = None,
    source_type: str | None = None,
    tags: str | None = None,
    status: str | None = None,
    folder: str | None = None,
    prefer_recent: bool = False,
    metadata_filters: str | None = None,
    enr_doc_type: str | None = None,
    enr_topics: str | None = None,
) -> dict:
    # Validate
    if not query or not query.strip():
        return _error(
            "empty_query",
            "Query must not be empty.",
            "Provide a natural-language search query (e.g., 'machine learning papers').",
        )
    top_k = max(1, min(top_k, _MAX_TOP_K))
    err = _validate_source_type(source_type)
    if err:
        return err

    # Parse metadata_filters JSON string
    parsed_filters: dict[str, str] | None = None
    if metadata_filters:
        import json
        try:
            parsed_filters = json.loads(metadata_filters)
            if not isinstance(parsed_filters, dict):
                return _error(
                    "invalid_parameter",
                    "metadata_filters must be a JSON object (e.g. '{\"section\": \"Introduction\"}').",
                )
        except (json.JSONDecodeError, TypeError):
            return _error(
                "invalid_parameter",
                "metadata_filters is not valid JSON.",
                'Provide a JSON object string, e.g. \'{"section": "Introduction"}\'.',
            )

    try:
        store, embed_provider, config = _get_deps()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize: {exc}",
                       "Check config.yaml paths and ensure configured services are running.")

    try:
        search_cfg = config.get("search", {})
        recency_cfg = search_cfg.get("recency", {})
        importance_cfg = search_cfg.get("importance", {})

        reranker = build_reranker(config)

        result = hybrid_search(
            store,
            embed_provider,
            query,
            vector_top_k=search_cfg.get("vector_top_k", 50),
            keyword_top_k=search_cfg.get("keyword_top_k", 50),
            final_top_k=top_k,
            rrf_k=search_cfg.get("rrf_k", 60),
            doc_id_prefix=doc_id_prefix,
            source_type=source_type,
            tags=tags,
            status=status,
            folder=folder,
            reranker=reranker,
            prefer_recent=prefer_recent,
            recency_half_life_days=recency_cfg.get("half_life_days", 90.0),
            recency_weight=recency_cfg.get("weight", 0.3),
            metadata_filters=parsed_filters,
            enr_doc_type=enr_doc_type,
            enr_topics=enr_topics,
            importance_field=importance_cfg.get("field", "enr_importance"),
            importance_weight=importance_cfg.get("weight", 0.3),
            min_score_threshold=search_cfg.get("min_score_threshold", 0.0),
        )
    except Exception as exc:
        return _error("search_failed", f"Search operation failed: {exc}",
                       "Check that the index exists (run file_index_update) and configured services are running.")
    return {
        "results": [_hit_to_dict(h) for h in result.hits],
        "diagnostics": result.diagnostics,
    }


def _file_get_chunk_impl(doc_id: str, loc: str) -> dict:
    if not doc_id or not doc_id.strip():
        return _error(
            "invalid_parameter",
            "doc_id must not be empty.",
            "Provide the document-relative file path (e.g., 'Projects/recipe.md'). "
            "Use file_search or file_list_documents to find valid doc_ids.",
        )
    if not loc or not loc.strip():
        return _error(
            "invalid_parameter",
            "loc must not be empty.",
            "Provide the chunk locator (e.g., 'c:0' or 'p:3:c:1'). "
            "Use file_search to find valid doc_id + loc pairs.",
        )

    try:
        store, _, _ = _get_deps()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize: {exc}",
                       "Check config.yaml paths and ensure configured services are running.")

    try:
        hit = store.get_chunk(doc_id, loc)
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to retrieve chunk: {exc}",
                       "Check that the index exists (run file_index_update).")
    if hit is not None:
        return _hit_to_dict(hit, include_text=True)

    # Determine why it wasn't found
    try:
        doc_ids = store.list_doc_ids()
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to list documents: {exc}",
                       "Check that the index exists (run file_index_update).")
    if not doc_ids:
        return _error(
            "index_empty",
            "No documents indexed yet.",
            "Run file_index_update to index your documents first.",
        )
    if doc_id not in doc_ids:
        return _error(
            "not_found",
            f"Document '{doc_id}' not found in the index.",
            "Use file_search or file_list_documents to find valid doc_ids.",
        )
    return _error(
        "not_found",
        f"Chunk loc '{loc}' not found in document '{doc_id}'.",
        "Use file_search to find valid loc values, or file_get_doc_chunks "
        f"to see all chunks for this document.",
    )


def _file_get_doc_chunks_impl(doc_id: str) -> list[dict] | dict:
    if not doc_id or not doc_id.strip():
        return _error(
            "invalid_parameter",
            "doc_id must not be empty.",
            "Provide the document-relative file path (e.g., 'Projects/recipe.md'). "
            "Use file_search or file_list_documents to find valid doc_ids.",
        )

    try:
        store, _, _ = _get_deps()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize: {exc}",
                       "Check config.yaml paths and ensure configured services are running.")

    try:
        chunks = store.get_doc_chunks(doc_id)
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to retrieve chunks: {exc}",
                       "Check that the index exists (run file_index_update).")
    if chunks:
        return [_hit_to_dict(h, include_text=True) for h in chunks]

    try:
        doc_ids = store.list_doc_ids()
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to list documents: {exc}",
                       "Check that the index exists (run file_index_update).")
    if not doc_ids:
        return _error(
            "index_empty",
            "No documents indexed yet.",
            "Run file_index_update to index your documents first.",
        )
    return _error(
        "not_found",
        f"Document '{doc_id}' not found in the index.",
        "Use file_search or file_list_documents to find valid doc_ids.",
    )


def _file_list_documents_impl(
    offset: int = 0,
    limit: int = 50,
    source_type: str | None = None,
    folder: str | None = None,
) -> dict:
    err = _validate_source_type(source_type)
    if err:
        return err
    offset = max(0, offset)
    limit = max(1, min(limit, _MAX_LIMIT))

    try:
        store, _, _ = _get_deps()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize: {exc}",
                       "Check config.yaml paths and ensure configured services are running.")
    # Get all docs (list_recent_docs sorts by mtime desc)
    try:
        all_docs = store.list_recent_docs(limit=100_000, source_type=source_type, folder=folder)
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to list documents: {exc}",
                       "Check that the index exists (run file_index_update).")
    total = len(all_docs)
    page = all_docs[offset : offset + limit]
    _enrich_doc_list(page)

    return {"documents": page, "total": total, "offset": offset, "limit": limit}


def _file_status_impl() -> dict:
    try:
        store, _, config = _get_deps()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize: {exc}",
                       "Check config.yaml paths and ensure configured services are running.")

    try:
        doc_ids = store.list_doc_ids()
        chunk_count = store.count_chunks()
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to read index status: {exc}",
                       "Check that the index exists (run file_index_update).")

    import json
    index_root = Path(config["index_root"])
    meta_path = index_root / "index_metadata.json"
    last_run_at = None
    last_index_failed_count = 0
    last_index_warnings: list[str] = []
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                last_run_at = meta.get("last_run_at")
                last_index_failed_count = meta.get("failed_count", 0)
                last_index_warnings = meta.get("warnings", [])
        except Exception as exc:
            logger.warning("Failed to read index_metadata.json: %s", exc)

    # Health: FTS availability
    fts_ok = store.fts_available()

    # Health: reranker status
    reranker_cfg = config.get("search", {}).get("reranker", {})
    reranker_enabled = reranker_cfg.get("enabled", False)
    reranker_responsive = None
    if reranker_enabled:
        try:
            import httpx
            api_key = os.environ.get("DEEPINFRA_API_KEY", "")
            model = reranker_cfg.get("model", "Qwen/Qwen3-Reranker-8B")
            resp = httpx.post(
                f"https://api.deepinfra.com/v1/inference/{model}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "queries": ["test"],
                    "documents": ["test"],
                },
                timeout=10.0,
            )
            reranker_responsive = resp.status_code == 200
        except Exception:
            reranker_responsive = False

    return {
        "doc_count": len(doc_ids),
        "chunk_count": chunk_count,
        "last_run_at": last_run_at,
        "embeddings_provider": config.get("embeddings", {}).get("provider"),
        "metadata_fields": sorted(store._metadata_subfields()),
        "health": {
            "fts_available": fts_ok,
            "reranker_enabled": reranker_enabled,
            "reranker_responsive": reranker_responsive,
            "last_index_failed_count": last_index_failed_count,
            "last_index_warnings": last_index_warnings,
        },
    }


def _file_recent_impl(
    limit: int = 20,
    source_type: str | None = None,
    folder: str | None = None,
) -> list[dict] | dict:
    err = _validate_source_type(source_type)
    if err:
        return err
    limit = max(1, min(limit, _MAX_TOP_K))

    try:
        store, _, _ = _get_deps()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize: {exc}",
                       "Check config.yaml paths and ensure configured services are running.")
    try:
        docs = store.list_recent_docs(limit=limit, source_type=source_type, folder=folder)
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to list recent documents: {exc}",
                       "Check that the index exists (run file_index_update).")
    _enrich_doc_list(docs)
    return docs


def _file_facets_impl() -> dict:
    try:
        store, _, _ = _get_deps()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize: {exc}",
                       "Check config.yaml paths and ensure configured services are running.")
    try:
        return store.facets()
    except Exception as exc:
        return _error("retrieval_failed", f"Failed to retrieve facets: {exc}",
                       "Check that the index exists (run file_index_update).")


def _file_folders_impl() -> dict:
    """Scan the documents directory and return folder tree with file counts."""
    import os
    from collections import Counter
    from flow_index_vault import _matches_any

    config = load_config()
    docs_root = Path(config["documents_root"])
    if not docs_root.exists():
        return _error(
            "not_found",
            f"Documents root directory not found: {docs_root}",
            "Check documents_root (or vault_root) in config.yaml.",
        )

    scan_cfg = config.get("scan", {})
    include = scan_cfg.get("include", ["**/*.md"])
    exclude = scan_cfg.get("exclude", [])

    folder_counts: Counter = Counter()
    total_files = 0
    visited: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(docs_root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            continue
        visited.add(real)
        for fname in filenames:
            full_path = Path(dirpath) / fname
            rel_str = str(full_path.relative_to(docs_root)).replace("\\", "/")
            if _matches_any(rel_str, exclude):
                continue
            if not _matches_any(rel_str, include):
                continue
            total_files += 1
            rel_dir = str(Path(dirpath).relative_to(docs_root)).replace("\\", "/")
            if rel_dir == ".":
                rel_dir = "."
            folder_counts[rel_dir] += 1

    folders = sorted(
        [{"path": p, "file_count": c} for p, c in folder_counts.items()],
        key=lambda x: x["path"],
    )
    return {
        "documents_root": str(docs_root),
        "folders": folders,
        "total_folders": len(folders),
        "total_files": total_files,
    }


def _file_index_update_impl(config_path: str = "config.yaml") -> dict:
    from flow_index_vault import index_vault_flow

    t0 = time.perf_counter()
    logger.info("file_index_update: starting indexer flow")
    try:
        index_vault_flow(config_path)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("file_index_update failed after %.1fs: %s", elapsed, exc)
        return _error(
            "index_failed",
            f"Index update failed after {elapsed:.1f}s: {exc}",
            "Common causes: cloud API keys not set, configured services unreachable, "
            "documents_root path incorrect in config.yaml, or disk full. Check logs for details.",
        )
    elapsed = time.perf_counter() - t0

    # Invalidate cache so next query rebuilds store/embed with fresh data
    global _cache
    _cache = None

    # Read results from index_metadata.json written by the flow
    import json
    config = load_config(config_path)
    index_root = Path(config["index_root"])
    meta_path = index_root / "index_metadata.json"
    result: dict = {"status": "completed", "elapsed_seconds": round(elapsed, 1)}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            result["last_run_at"] = meta.get("last_run_at")
            result["doc_count"] = meta.get("doc_count")
            result["chunk_count"] = meta.get("chunk_count")
            failed_count = meta.get("failed_count", 0)
            if failed_count:
                result["status"] = "completed_with_errors"
                result["failed_count"] = failed_count
                result["failed_docs"] = meta.get("failed_docs", [])
        except Exception as exc:
            logger.warning("Failed to read index_metadata.json after indexing: %s", exc)
    logger.info("file_index_update: completed in %.1fs", elapsed)
    return result


# ---------------------------------------------------------------------------
# Taxonomy tool implementations
# ---------------------------------------------------------------------------


def _get_taxonomy_store():
    """Lazily build and cache the taxonomy store."""
    from core.taxonomy import load_taxonomy_store
    _, _, config = _get_deps()
    return load_taxonomy_store(config)


def _file_taxonomy_list_impl(kind: str | None = None, status: str = "active") -> list[dict] | dict:
    try:
        store = _get_taxonomy_store()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize taxonomy: {exc}")
    if kind:
        return store.list_by_kind(kind, status=status)
    # List all kinds
    results = []
    for k in ("tag", "folder", "doc_type"):
        results.extend(store.list_by_kind(k, status=status))
    return results


def _file_taxonomy_get_impl(entry_id: str) -> dict:
    if not entry_id or not entry_id.strip():
        return _error("invalid_parameter", "id must not be empty.")
    try:
        store = _get_taxonomy_store()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize taxonomy: {exc}")
    result = store.get(entry_id)
    if result is None:
        return _error("not_found", f"Taxonomy entry '{entry_id}' not found.",
                       "Use file_taxonomy_list to see available entries.")
    return result


def _file_taxonomy_search_impl(query: str, kind: str | None = None, top_k: int = 10) -> list[dict] | dict:
    if not query or not query.strip():
        return _error("empty_query", "Query must not be empty.")
    try:
        store = _get_taxonomy_store()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize taxonomy: {exc}")
    return store.search(query, kind=kind, top_k=top_k)


def _file_taxonomy_add_impl(
    kind: str, name: str, description: str,
    aliases: str = "", parent: str = "", status: str = "active",
    contents_type: str = "", created_by: str = "AI",
) -> dict:
    if kind not in ("tag", "folder", "doc_type"):
        return _error("invalid_parameter", f"kind must be 'tag', 'folder', or 'doc_type'. Got: '{kind}'.")
    if not name or not name.strip():
        return _error("invalid_parameter", "name must not be empty.")
    try:
        store = _get_taxonomy_store()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize taxonomy: {exc}")
    # Check for duplicates
    entry_id = f"{kind}:{name}"
    if store.get(entry_id) is not None:
        return _error("duplicate", f"Entry '{entry_id}' already exists.",
                       "Use file_taxonomy_update to modify existing entries.")
    row = store.add(kind, name, description, aliases, parent, status,
                    0, 1, contents_type, created_by)
    row.pop("vector", None)
    return row


def _file_taxonomy_update_impl(
    entry_id: str,
    description: str | None = None,
    aliases: str | None = None,
    status: str | None = None,
    parent: str | None = None,
    contents_type: str | None = None,
) -> dict:
    if not entry_id or not entry_id.strip():
        return _error("invalid_parameter", "id must not be empty.")
    try:
        store = _get_taxonomy_store()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize taxonomy: {exc}")
    fields = {}
    if description is not None:
        fields["description"] = description
    if aliases is not None:
        fields["aliases"] = aliases
    if status is not None:
        fields["status"] = status
    if parent is not None:
        fields["parent"] = parent
    if contents_type is not None:
        fields["contents_type"] = contents_type
    if not fields:
        return _error("invalid_parameter", "No fields to update. Provide at least one field.")
    result = store.update(entry_id, **fields)
    if result is None:
        return _error("not_found", f"Taxonomy entry '{entry_id}' not found.")
    return result


def _file_taxonomy_delete_impl(entry_id: str) -> dict:
    if not entry_id or not entry_id.strip():
        return _error("invalid_parameter", "id must not be empty.")
    try:
        store = _get_taxonomy_store()
    except Exception as exc:
        return _error("service_unavailable", f"Failed to initialize taxonomy: {exc}")
    if store.delete(entry_id):
        return {"deleted": True}
    return _error("not_found", f"Taxonomy entry '{entry_id}' not found.")


def _file_taxonomy_import_impl() -> dict:
    try:
        from scripts.seed_taxonomy import seed_taxonomy
        return seed_taxonomy()
    except Exception as exc:
        return _error("import_failed", f"Taxonomy import failed: {exc}",
                       "Check that SQLite databases exist at ~/Documents/Primary/0-AI/directory_info/.")


# ---------------------------------------------------------------------------
# MCP tool registration (FastMCP)
# ---------------------------------------------------------------------------

if HAS_MCP and FastMCP is not None:
    mcp = FastMCP("file-index-mcp", json_response=True)

    @mcp.tool()
    def file_search(
        query: str,
        top_k: int = 10,
        doc_id_prefix: str | None = None,
        source_type: str | None = None,
        tags: str | None = None,
        status: str | None = None,
        folder: str | None = None,
        prefer_recent: bool = False,
        metadata_filters: str | None = None,
        enr_doc_type: str | None = None,
        enr_topics: str | None = None,
    ) -> dict:
        """Hybrid semantic + keyword search over the indexed documents.

        Combines vector similarity search with BM25 keyword search using
        Reciprocal Rank Fusion (RRF), with optional cross-encoder re-ranking.

        Args:
            query: Natural-language search query (required, non-empty).
            top_k: Max results to return (1-100, default 10).
            doc_id_prefix: Filter to docs under this path prefix
                (e.g., "Projects/" to search only the Projects folder).
            source_type: Filter by file type: "md", "pdf", or "img".
            tags: Comma-separated tags to filter by. Matches documents that
                have ANY of the listed tags (OR logic). Example: "recipe,korean".
            status: Filter by document status (e.g., "active", "archived", "draft").
                Values come from the "status" field in document frontmatter.
            folder: Filter by top-level folder name (e.g., "Projects", "Archive").
            prefer_recent: If true, boost recently modified documents in ranking.
                Uses time-decay scoring (not a strict sort by date).
            metadata_filters: JSON string of {field: value} pairs for filtering on
                any metadata field, including dynamic fields added by the indexer.
                Example: '{"section": "Introduction"}'. Use file_facets or
                file_status to discover available fields. Combined with AND logic.
            enr_doc_type: Filter by LLM-enriched document type (e.g.,
                "Geotechnical Report", "Recipe"). Comma-separated for OR logic.
                Values come from file_facets doc_types. Uses LIKE matching.
            enr_topics: Filter by LLM-enriched topic (e.g., "machine learning",
                "Korean cooking"). Comma-separated for OR logic. Values come
                from file_facets topics. Uses LIKE matching.

        Returns a dict with:
            - results: List of result dicts, each containing:
                - doc_id: Document-relative file path (e.g., "Projects/recipe.md").
                - loc: Chunk locator within the document (e.g., "c:0" for first chunk,
                  "p:3:c:1" for page 3 chunk 1 of a PDF).
                - snippet: First ~200 characters of the chunk text (preview).
                - score: Relevance score (higher is better).
                - title, folder, status, source_type: Document metadata.
                - tags: Array of tag strings from frontmatter.
                - enr_summary, enr_doc_type, enr_topics, enr_keywords: LLM-generated
                  enrichment (comma-separated strings; empty if not yet enriched).
                - enr_entities_people, enr_entities_places, enr_entities_orgs,
                  enr_entities_dates: Named entities extracted by LLM (comma-separated).
                - enr_key_facts: Key facts from the document (comma-separated string).
                - description, author, custom_meta: Additional frontmatter fields.
                - Any dynamic metadata fields (e.g., section) are also included.
            - diagnostics: Search pipeline health signals:
                - vector_search_active: true if vector (semantic) search ran successfully.
                - keyword_search_active: true if BM25/FTS ran successfully.
                - reranker_applied: true if cross-encoder reranking ran successfully.
                - degraded: true if any retrieval stage failed silently. When true,
                  results are still returned but may be lower quality (e.g., vector-only
                  without keyword boost, or RRF order without reranking).

        Filters are combined with AND logic (e.g., source_type="md" AND folder="Projects").
        Only the tags filter uses OR logic within its values.

        To get the full text of a result, call file_get_chunk with the doc_id and loc.
        To see all available filter values, call file_facets first.

        On error, returns {"error": true, "code": "...", "message": "...", "fix": "..."}.
        """
        return _file_search_impl(
            query, top_k, doc_id_prefix, source_type, tags, status, folder,
            prefer_recent, metadata_filters, enr_doc_type, enr_topics,
        )

    @mcp.tool()
    def file_get_chunk(doc_id: str, loc: str) -> dict:
        """Retrieve the full text and metadata for a single chunk.

        Use this after file_search to get the complete text of a result.

        Args:
            doc_id: Document-relative file path, exactly as returned by file_search
                (e.g., "Projects/recipe.md" or "Archive/2024/report.pdf").
            loc: Chunk locator, exactly as returned by file_search
                (e.g., "c:0" for chunk 0, "p:3:c:1" for page 3 chunk 1).

        Returns a dict with all fields from file_search, plus:
            - text: Full chunk text (including contextual header with doc metadata).

        On error, returns {"error": true, "code": "...", "message": "...", "fix": "..."}.
        Common errors:
            - not_found: The doc_id or loc does not exist. Use file_search to
              find valid doc_id + loc pairs.
            - index_empty: No documents indexed. Run file_index_update first.
        """
        return _file_get_chunk_impl(doc_id, loc)

    @mcp.tool()
    def file_get_doc_chunks(doc_id: str) -> list[dict] | dict:
        """Retrieve all chunks for a document, sorted by position.

        Returns every chunk of the given document with full text. Useful for
        reading an entire document or understanding its structure.

        Args:
            doc_id: Document-relative file path (e.g., "Projects/recipe.md").
                Use file_search or file_list_documents to find valid doc_ids.

        Returns a list of chunk dicts, each containing:
            - loc: Chunk position (e.g., "c:0", "c:1", "p:2:c:0"). Sorted
              in document order.
            - text: Full chunk text (including contextual header).
            - snippet, title, tags, and all metadata/enrichment fields.

        On error, returns {"error": true, "code": "...", "message": "...", "fix": "..."}.
        """
        return _file_get_doc_chunks_impl(doc_id)

    @mcp.tool()
    def file_list_documents(
        offset: int = 0,
        limit: int = 50,
        source_type: str | None = None,
        folder: str | None = None,
    ) -> dict:
        """Browse all indexed documents with pagination.

        Lists documents sorted by modification time (newest first). Returns
        document-level metadata only (no chunk text or embeddings).

        Args:
            offset: Number of documents to skip (for pagination, default 0).
            limit: Max documents to return (1-200, default 50).
            source_type: Filter by file type: "md", "pdf", or "img".
            folder: Filter by top-level folder name (e.g., "Projects").

        Returns a dict:
            - documents: List of document metadata dicts, each with:
                doc_id, title, source_type, folder, tags (array), status,
                created, mtime (unix timestamp), mtime_iso (ISO 8601 UTC),
                size (bytes).
            - total: Total number of matching documents (for pagination).
            - offset: The offset used.
            - limit: The limit used.

        Use total and offset to paginate: next page is offset + limit.
        On error, returns {"error": true, "code": "...", "message": "...", "fix": "..."}.
        """
        return _file_list_documents_impl(offset, limit, source_type, folder)

    @mcp.tool()
    def file_recent(
        limit: int = 20,
        source_type: str | None = None,
        folder: str | None = None,
    ) -> list[dict] | dict:
        """List recently modified or indexed documents, newest first.

        Returns document-level metadata (no chunk text). Useful for seeing
        what was recently added, modified, or re-indexed.

        Args:
            limit: Max results (1-100, default 20).
            source_type: Filter by file type: "md", "pdf", or "img".
            folder: Filter by top-level folder name (e.g., "Projects").

        Returns a list of document metadata dicts, each containing:
            - doc_id: Document-relative path (e.g., "Archive/notes.md").
            - title, source_type, folder, tags (array), status, created.
            - mtime: Unix timestamp of last modification.
            - mtime_iso: ISO 8601 UTC string (e.g., "2026-02-20T15:30:00+00:00").
            - size: File size in bytes.

        On error, returns {"error": true, "code": "...", "message": "...", "fix": "..."}.
        """
        return _file_recent_impl(limit, source_type, folder)

    @mcp.tool()
    def file_facets() -> dict:
        """Discover all available filter values, tags, topics, and document counts.

        Call this before searching to see what tags, folders, topics, keywords,
        and other filterable fields exist in the index. Counts reflect documents
        (not chunks) — a document with 10 chunks counts once.

        Takes no arguments.

        Returns a dict with:
            - total_docs: Total number of indexed documents.
            - total_chunks: Total number of text chunks across all documents.
            - tags: [{value: "recipe", count: 5}, ...] sorted by count desc.
            - folders: [{value: "Projects", count: 12}, ...].
            - source_types: [{value: "md", count: 30}, ...].
            - statuses: [{value: "active", count: 38}, ...].
            - doc_types: [{value: "tutorial", count: 8}, ...] (from enr_doc_type).
            - authors: [{value: "Dan Park", count: 15}, ...] (from frontmatter).
            - topics: [{value: "machine learning", count: 12}, ...] (from enr_topics).
            - keywords: [{value: "transformer", count: 8}, ...] (from enr_keywords).
            - entities_people: [{value: "Alan Turing", count: 3}, ...] (from enr_entities_people).
            - entities_places: [{value: "Seoul", count: 5}, ...] (from enr_entities_places).
            - entities_orgs: [{value: "OpenAI", count: 4}, ...] (from enr_entities_orgs).

        Use these values as filter parameters in file_search, file_recent,
        and file_list_documents. Use file_folders to browse the directory tree.
        """
        return _file_facets_impl()

    @mcp.tool()
    def file_folders() -> dict:
        """Browse the documents folder/directory structure with file counts.

        Scans the actual documents filesystem (not the index) and returns every
        folder that contains supported files, with a count of files per folder.
        Shows the full nested hierarchy, including folders with unindexed files.

        Takes no arguments.

        Returns a dict with:
            - documents_root: Absolute path to the documents root directory.
            - folders: List of {path, file_count} sorted alphabetically.
              "path" is relative to documents_root (e.g., "Projects/DocOrganizer").
              Root-level files show as path ".".
            - total_folders: Number of folders containing files.
            - total_files: Total supported files across all folders.

        Use folder paths with file_search (doc_id_prefix) or file_list_documents
        (folder) to narrow results to a specific directory.
        """
        return _file_folders_impl()

    @mcp.tool()
    def file_status() -> dict:
        """Return index health information and configuration.

        Takes no arguments.

        Returns a dict with:
            - doc_count: Number of distinct documents (files) in the index.
            - chunk_count: Total text chunks across all documents.
            - last_run_at: ISO 8601 timestamp of the last file_index_update run
              (null if never indexed).
            - embeddings_provider: Configured embedding backend (e.g., "openrouter").
            - metadata_fields: List of all metadata field names in the index schema
              (e.g., ["doc_id", "folder", "section", "source_type", ...]). Use these
              field names with file_search's metadata_filters parameter.
            - health: Service health diagnostics:
                - fts_available: true if the full-text search (BM25/tantivy) index
                  is operational. False means keyword search is down.
                - reranker_enabled: true if cross-encoder reranking is configured.
                - reranker_responsive: true/false if enabled and server is/isn't
                  responding, null if reranker is not enabled.
                - last_index_failed_count: Number of documents that failed during
                  the last indexing run (0 = clean).

        Troubleshooting:
            - If doc_count is 0, run file_index_update to index your documents.
            - If last_run_at is null, the index has never been built.
            - If health.fts_available is false, run file_index_update to rebuild
              the FTS index.
            - If health.reranker_responsive is false, check DEEPINFRA_API_KEY is set
              and DeepInfra API is reachable.
        """
        return _file_status_impl()

    # --- Taxonomy tools ---

    @mcp.tool()
    def file_taxonomy_list(
        kind: str | None = None,
        status: str = "active",
    ) -> dict | list[dict]:
        """List taxonomy entries filtered by kind and/or status.

        Args:
            kind: Filter by kind: "tag", "folder", or "doc_type". If omitted, lists all kinds.
            status: Filter by status (default "active"). Use "archived" to see archived entries.

        Returns a list of taxonomy entry dicts, each containing:
            - id: Unique identifier (e.g., "tag:machine-learning").
            - kind: Entry type ("tag", "folder", "doc_type").
            - name: Canonical name.
            - description: Human-readable description.
            - aliases: Comma-separated alternative names.
            - status: "active" or "archived".
            - usage_count: Number of times used in enrichment.
            - ai_managed: 1 if AI-managed, 0 if human-managed.
            - contents_type: For folders, the type of contents (e.g., "notes", "data").
            - created_by: Who created the entry.
        """
        return _file_taxonomy_list_impl(kind, status)

    @mcp.tool()
    def file_taxonomy_get(id: str) -> dict:
        """Get a single taxonomy entry by its id.

        Args:
            id: The taxonomy entry id (e.g., "tag:machine-learning", "folder:Projects/").

        Returns the entry dict, or an error if not found.
        """
        return _file_taxonomy_get_impl(id)

    @mcp.tool()
    def file_taxonomy_search(
        query: str,
        kind: str | None = None,
        top_k: int = 10,
    ) -> list[dict] | dict:
        """Semantic search over taxonomy descriptions.

        Finds entries whose descriptions are semantically similar to the query.

        Args:
            query: Natural-language search query (e.g., "property management").
            kind: Optional filter by kind: "tag", "folder", or "doc_type".
            top_k: Max results to return (default 10).

        Returns a list of matching taxonomy entries ranked by relevance.
        """
        return _file_taxonomy_search_impl(query, kind, top_k)

    @mcp.tool()
    def file_taxonomy_add(
        kind: str,
        name: str,
        description: str,
        aliases: str = "",
        parent: str = "",
        status: str = "active",
        contents_type: str = "",
        created_by: str = "AI",
    ) -> dict:
        """Add a new taxonomy entry.

        Args:
            kind: Entry type: "tag", "folder", or "doc_type".
            name: Canonical name (e.g., "machine-learning", "Projects/").
            description: Human-readable description for semantic matching.
            aliases: Comma-separated alternative names (e.g., "ml,deep-learning").
            parent: Parent entry id for hierarchy (optional, for future use).
            status: "active" (default) or "archived".
            contents_type: For folders, the type of contents (e.g., "notes", "data").
            created_by: Who is creating this entry (default "AI").

        Returns the created entry dict.
        """
        return _file_taxonomy_add_impl(kind, name, description, aliases, parent, status, contents_type, created_by)

    @mcp.tool()
    def file_taxonomy_update(
        id: str,
        description: str | None = None,
        aliases: str | None = None,
        status: str | None = None,
        parent: str | None = None,
        contents_type: str | None = None,
    ) -> dict:
        """Update an existing taxonomy entry.

        Args:
            id: The taxonomy entry id (e.g., "tag:machine-learning").
            description: New description (re-embeds for semantic search).
            aliases: New comma-separated aliases.
            status: New status ("active" or "archived"). Use "archived" to soft-delete.
            parent: New parent entry id.
            contents_type: New contents type (folders only).

        Returns the updated entry dict, or an error if not found.
        """
        return _file_taxonomy_update_impl(id, description, aliases, status, parent, contents_type)

    @mcp.tool()
    def file_taxonomy_delete(id: str) -> dict:
        """Hard delete a taxonomy entry.

        Args:
            id: The taxonomy entry id (e.g., "tag:old-tag").

        Returns {"deleted": true} on success, or an error if not found.
        Consider using file_taxonomy_update with status="archived" instead.
        """
        return _file_taxonomy_delete_impl(id)

    @mcp.tool()
    def file_taxonomy_import() -> dict:
        """Import taxonomy data from existing SQLite databases.

        Reads tags from ~/Documents/Primary/0-AI/directory_info/tags.db and
        directories from ~/Documents/Primary/0-AI/directory_info/directory.db.
        Idempotent: skips entries that already exist.

        Returns import statistics (counts added, skipped, totals).
        """
        return _file_taxonomy_import_impl()

    @mcp.tool()
    def file_index_update() -> dict:
        """Incrementally update the index based on file changes.

        Scans the documents directory, diffs against the current index, and only processes
        new, modified, or deleted files. Does NOT rebuild from scratch — existing
        unchanged documents keep their current index entries.

        WARNING: This is a blocking operation that may take several minutes for
        large document sets with many changes.

        Takes no arguments. Pipeline steps:
            1. Scans the documents directory for all supported files.
            2. Diffs against the current index (by file modification time) to find
               new, modified, and deleted files.
            3. Extracts text from new/modified files only (Markdown, PDF, images via OCR).
            4. Runs LLM enrichment (enr_summary, enr_entities_*, enr_topics) on new/modified files if enabled.
            5. Chunks text and generates embeddings for new/modified files.
            6. Removes deleted files from the index.
            7. Rebuilds the keyword search (FTS) index if any changes occurred.

        Returns a dict with:
            - status: "completed" on success.
            - elapsed_seconds: Time taken (float).
            - doc_count: Total documents after updating.
            - chunk_count: Total chunks after updating.
            - last_run_at: ISO 8601 timestamp of this run.

        On error, returns {"error": true, "code": "index_failed", "message": "...",
        "fix": "..."} with guidance (e.g., check configured services are running, verify config paths).
        """
        return _file_index_update_impl()

    def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 7788):
        """Run the MCP server.

        Args:
            transport: "stdio" (for OpenClaw/Claude Desktop) or "streamable-http" (for testing).
            host: Host for HTTP transport (ignored for stdio).
            port: Port for HTTP transport (ignored for stdio).
        """
        if transport == "stdio":
            mcp.run(transport="stdio")
        else:
            import anyio

            async def _run_http():
                import hmac
                import uvicorn
                from starlette.applications import Starlette
                from starlette.responses import JSONResponse
                from starlette.routing import Mount

                from api_server import build_api_app

                api_key = os.environ.get("API_KEY")
                config = load_config()

                mcp_app = mcp.streamable_http_app()
                api_app = build_api_app(documents_root=Path(config["documents_root"]))

                # Compose: /api/* → REST API, everything else → MCP
                app = Starlette(routes=[
                    Mount("/api", app=api_app),
                    Mount("/", app=mcp_app),
                ])

                if api_key:
                    original_app = app
                    expected = f"Bearer {api_key}".encode()

                    async def auth_app(scope, receive, send):
                        if scope["type"] in ("http", "websocket"):
                            auth_value = b""
                            for name, value in scope.get("headers", []):
                                if name == b"authorization":
                                    auth_value = value
                                    break
                            if not hmac.compare_digest(auth_value, expected):
                                response = JSONResponse(
                                    {"error": "Unauthorized"}, status_code=401
                                )
                                await response(scope, receive, send)
                                return
                        await original_app(scope, receive, send)

                    app = auth_app

                uvi_config = uvicorn.Config(app, host=host, port=port, log_level="info")
                server = uvicorn.Server(uvi_config)
                await server.serve()

            anyio.run(_run_http)
else:
    def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 7788):
        print("Install mcp to run the server: pip install mcp")
        print("Tool implementations available: _file_search_impl, _file_get_chunk_impl, _file_status_impl")


if __name__ == "__main__":
    import sys
    from prefect_server import PrefectServer

    config = load_config()

    # Configure root logger from config (Prefect flow/task logs are independent)
    log_level = config.get("logging", {}).get("level", "WARNING").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))

    mcp_cfg = config.get("mcp", {})

    with PrefectServer():
        # Default: stdio (for OpenClaw). Pass --http for browser/testing.
        if "--http" in sys.argv:
            host = mcp_cfg.get("host", "0.0.0.0")
            port = int(os.environ.get("PORT", mcp_cfg.get("port", 7788)))
            run_server(transport="streamable-http", host=host, port=port)
        else:
            run_server(transport="stdio")
