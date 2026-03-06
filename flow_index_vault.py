"""Prefect flow: scan vault, diff with store, process docs, delete removed, log stats.

Uses LlamaIndex for chunking (SentenceSplitter) and embeddings.
Storage via our LanceDBStore (which wraps LlamaIndex's LanceDBVectorStore).
Extraction via extractors.py (Markdown, PDF with PyMuPDF, images with OCR).

Chunking enhancements:
- Contextual headers: each chunk is prepended with document metadata
  (title, type, folder, tags, page/section) before embedding so that
  chunks are self-describing and retrieve better in isolation.
- Heading-aware splitting for Markdown: text is first split at heading
  boundaries (h1-h3), then SentenceSplitter runs within each section.
  Each chunk carries its heading hierarchy as context.

LLM document enrichment (optional):
- When enrichment.enabled is true, the configured LLM provider extracts
  structured metadata (summary, doc_type, entities, topics, keywords,
  key_facts) from every new/modified document before chunking.
- Summary and topics are prepended to chunk contextual headers.
- All enrichment fields are stored in LanceDB for filtering and search.

Complex objects (store, embed_provider, splitter, ocr_provider) are built inside
the flow and passed to tasks via a shared module-level dict (_RUNTIME) rather
than as task arguments.  This avoids Prefect 3's input-serialisation warnings
while keeping the flow easy to read.
"""

import os
import re
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

from core.config import load_config
from doc_enrichment import enrich_document, empty_enrichment, ENRICHMENT_FIELDS, failed_enrichment
from extractors import extract_text, extract_title, derive_folder, normalize_tags
from providers.embed import build_embed_provider
from providers.embed.base import EmbedProvider
from providers.ocr import build_ocr_provider
from lancedb_store import LanceDBStore


# Module-level runtime context populated by the flow, read by tasks.
# Avoids passing unpickleable objects as Prefect task arguments.
_RUNTIME: dict[str, Any] = {}


# --- Helpers ---


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_markdown_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown at heading boundaries (h1-h3).

    Returns (heading_breadcrumb, section_text) pairs.  The breadcrumb tracks
    the heading hierarchy, e.g. "Setup > Prerequisites".  Text before the
    first heading gets an empty breadcrumb.  Each section includes its own
    heading line so the content is self-contained.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if section_text:
            breadcrumb = " > ".join(h[1] for h in heading_stack)
            sections.append((breadcrumb, section_text))

    return sections


def _build_chunk_context(
    doc_meta: dict,
    page: int | None = None,
    section: str | None = None,
) -> str:
    """Build a minimal contextual header prepended to chunk text before embedding.

    Keeps only what's needed to situate the chunk within its parent document.
    Deliberately lean to avoid diluting lexical (FTS/BM25) retrieval with
    metadata terms — all metadata is stored in separate LanceDB columns and
    searchable via filters.

    Example output::

        [Document: TaxReturn | Topics: tax filing | Page: 31]
        Summary: Joint federal tax filing for 2022.

        <actual chunk text>
    """
    parts: list[str] = []
    if doc_meta.get("title"):
        parts.append(f"Document: {doc_meta['title']}")
    if doc_meta.get("enr_topics"):
        parts.append(f"Topics: {doc_meta['enr_topics']}")
    if page is not None:
        parts.append(f"Page: {page}")
    if section:
        parts.append(f"Section: {section}")
    if not parts:
        return ""

    header = f"[{' | '.join(parts)}]"
    summary = doc_meta.get("enr_summary", "")
    if summary:
        header += f"\nSummary: {summary}"
    return header + "\n\n"


def _semantic_subsplit(text: str, semantic_splitter) -> list[str]:
    """Use SemanticSplitterNodeParser to find topic boundaries in large text."""
    from llama_index.core.schema import Document as LIDocument

    doc = LIDocument(text=text)
    nodes = semantic_splitter.get_nodes_from_documents([doc])
    return [n.text for n in nodes if n.text.strip()]


def _split_section(
    text: str,
    splitter,
    semantic_splitter=None,
    semantic_threshold: int = 0,
) -> list[str]:
    """Split a text section into chunks.

    If the section exceeds *semantic_threshold* and a semantic splitter is
    available, it first finds topic boundaries via embedding similarity,
    then runs SentenceSplitter within each topic sub-section.  Otherwise
    falls back to SentenceSplitter directly.
    """
    if semantic_splitter and semantic_threshold and len(text) > semantic_threshold:
        sub_sections = _semantic_subsplit(text, semantic_splitter)
        chunks: list[str] = []
        for sub in sub_sections:
            chunks.extend(splitter.split_text(sub))
        return chunks
    return splitter.split_text(text)


def _matches_any(rel_str: str, patterns: list[str]) -> bool:
    """Check if a vault-relative path matches any glob pattern (supports ** for any depth)."""
    import fnmatch
    for pat in patterns:
        # fnmatch doesn't handle ** well; expand: **/*.md should match both root and nested
        if pat.startswith("**/"):
            # Match the suffix at any depth: *.md for root, **/*.md for nested
            suffix = pat[3:]  # e.g. "*.md"
            if fnmatch.fnmatch(rel_str, suffix):
                return True
            if fnmatch.fnmatch(rel_str, pat):
                return True
            # Also try: any/path/*.md via full glob
            if fnmatch.fnmatch(rel_str, "*/" + suffix):
                return True
            # Recursive match via os-level
            parts = rel_str.split("/")
            if fnmatch.fnmatch(parts[-1], suffix):
                return True
        else:
            if fnmatch.fnmatch(rel_str, pat):
                return True
    return False


# --- Tasks (one responsibility each) ---


@task
def scan_vault_task(vault_root: str | Path, include: list[str], exclude: list[str]) -> list[dict]:
    """Scan vault; return list of file records as dicts."""
    root = Path(vault_root)
    if not root.exists():
        return []
    records = []
    # os.walk with followlinks=True so symlinked directories (e.g. NAS mounts) are traversed.
    # Path.rglob does not follow symlinks in Python 3.13+.
    # Track visited real paths to guard against symlink cycles.
    visited: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            continue
        visited.add(real)
        for fname in filenames:
            full_path = Path(dirpath) / fname
            rel_str = str(full_path.relative_to(root)).replace("\\", "/")
            if _matches_any(rel_str, exclude):
                continue
            if not _matches_any(rel_str, include):
                continue
            try:
                stat = full_path.stat()
            except OSError:
                continue
            records.append({
                "doc_id": rel_str,
                "abs_path": str(full_path.resolve()),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "ext": full_path.suffix.lower().lstrip(".") or "bin",
            })
    return records


@task
def diff_index_task(
    scanned: list[dict],
    stored_doc_mtimes: dict[str, float],
) -> tuple[list[dict], list[str]]:
    """Compare scanned files vs stored docs. Return (to_add_or_update, to_delete).

    A file is added/updated if:
      - It's new (not in store), OR
      - Its mtime has changed since last index
    A file is deleted if it's in the store but no longer on disk.
    """
    scanned_ids = {r["doc_id"] for r in scanned}
    stored_ids = set(stored_doc_mtimes.keys())

    to_add_or_update = []
    for r in scanned:
        doc_id = r["doc_id"]
        if doc_id not in stored_ids:
            # New file
            to_add_or_update.append(r)
        elif r["mtime"] != stored_doc_mtimes.get(doc_id, 0.0):
            # Modified file (mtime changed)
            to_add_or_update.append(r)
        # else: unchanged, skip

    to_delete = list(stored_ids - scanned_ids)
    return to_add_or_update, to_delete


@task(retries=1, timeout_seconds=1800)
def process_doc_task(doc: dict) -> None:
    """Extract text, chunk with LlamaIndex, embed, upsert into store.

    Reads store / embed_provider / splitter / ocr_provider / config from _RUNTIME.
    Handles Markdown, PDF (with page-level chunking), and images (via OCR).
    """
    store: LanceDBStore = _RUNTIME["store"]
    embed_provider: EmbedProvider = _RUNTIME["embed_provider"]
    splitter: SentenceSplitter = _RUNTIME["splitter"]
    semantic_splitter = _RUNTIME.get("semantic_splitter")
    semantic_threshold: int = _RUNTIME.get("semantic_threshold", 0)
    ocr_provider = _RUNTIME.get("ocr_provider")  # may be None
    config: dict = _RUNTIME.get("config", {})

    logger = get_run_logger()
    doc_id = doc["doc_id"]
    abs_path = doc["abs_path"]
    mtime = doc["mtime"]
    size = doc["size"]
    ext = doc["ext"]
    logger.info(f"Processing: {doc_id}")

    # --- Determine source_type ---
    source_type = "md" if ext == "md" else "pdf" if ext == "pdf" else "img"

    # --- Extract text (Markdown / PDF / Image) ---
    pdf_cfg = config.get("pdf", {})
    result = extract_text(
        file_path=abs_path,
        ext=ext,
        ocr_provider=ocr_provider,
        pdf_strategy=pdf_cfg.get("strategy", "text_then_ocr"),
        min_text_chars=pdf_cfg.get("min_text_chars_before_ocr", 200),
        ocr_page_limit=pdf_cfg.get("ocr_page_limit", 200),
    )

    if not result.full_text.strip():
        logger.warning(f"No text extracted: {doc_id}")
        return

    # --- Extract document-level metadata ---
    fm = result.frontmatter  # from Markdown frontmatter; empty dict for PDF/images
    title = fm.get("title") or extract_title(result.full_text, doc_id)
    tags = normalize_tags(fm.get("tags"))
    folder = derive_folder(doc_id)
    status = fm.get("status", "archived" if folder.lower() in ("archive", "archived") else "active")
    created = str(fm["created"]) if "created" in fm else ""
    description = str(fm.get("description", "")).strip()
    author = str(fm.get("author", "")).strip()
    keywords = normalize_tags(fm.get("keywords"))

    # Collect remaining frontmatter fields into custom_meta JSON
    _KNOWN_FM_KEYS = {"title", "tags", "status", "created", "description", "author", "keywords"}
    extra_fm = {k: str(v) for k, v in fm.items() if k not in _KNOWN_FM_KEYS and v is not None}
    import json as _json
    custom_meta = _json.dumps(extra_fm, default=str) if extra_fm else ""

    # Shared metadata for all chunks of this doc
    doc_meta = {
        "doc_id": doc_id,
        "source_type": source_type,
        "mtime": mtime,
        "size": size,
        "title": title,
        "tags": tags,
        "folder": folder,
        "status": status,
        "created": created,
        "description": description,
        "author": author,
        "keywords": keywords,
        "custom_meta": custom_meta,
        "section": "",
    }
    # Promote extra frontmatter to real columns (skip collisions with reserved keys)
    for k, v in extra_fm.items():
        if k not in doc_meta:
            doc_meta[k] = v

    # --- LLM document enrichment (summary, entities, topics, etc.) ---
    llm_generator = _RUNTIME.get("llm_generator")
    enrichment_cfg = _RUNTIME.get("config", {}).get("enrichment", {})
    if llm_generator:
        enrichment = enrich_document(
            text=result.full_text,
            title=title,
            source_type=source_type,
            generator=llm_generator,
            max_input_chars=enrichment_cfg.get("max_input_chars", 4000),
            max_output_tokens=enrichment_cfg.get("max_output_tokens", 512),
        )
        if enrichment.get("_enrichment_failed"):
            reason = enrichment.pop("_enrichment_failed")
            logger.warning("Enrichment failed for '%s': %s", doc_id, reason)
            _RUNTIME.setdefault("_warnings", []).append(
                f"enrichment_failed:{doc_id}:{reason}"
            )
        elif not enrichment.get("enr_summary"):
            logger.warning("Enrichment returned empty summary for '%s' — LLM may have failed silently", doc_id)
        enrichment.pop("_enrichment_failed", None)
        doc_meta.update(enrichment)
    else:
        doc_meta.update(empty_enrichment())

    # --- Chunk and build nodes ---
    # Three paths:
    #   1. Multi-page PDF  → page-aware chunks, context header per page
    #   2. Markdown         → heading-aware sections, context header per section
    #   3. Images / 1-page PDF → flat chunks with context header
    # Every chunk is prepended with a contextual header before embedding
    # so it is self-describing in isolation.
    nodes: list[TextNode] = []

    if ext == "pdf" and len(result.pages) > 1:
        for page_text in result.pages:
            if not page_text.text.strip():
                continue
            raw_chunks = _split_section(
                page_text.text, splitter, semantic_splitter, semantic_threshold
            )
            ctx = _build_chunk_context(doc_meta, page=page_text.page)
            contextualized = [ctx + c for c in raw_chunks]
            page_vectors = embed_provider.embed_texts(contextualized)

            for i, (ctx_text, raw_text, vector) in enumerate(
                zip(contextualized, raw_chunks, page_vectors, strict=True)
            ):
                loc = f"p:{page_text.page}:c:{i}"
                chunk_uid = f"{doc_id}::{loc}"
                snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
                node = TextNode(
                    text=ctx_text,
                    id_=chunk_uid,
                    embedding=vector,
                    metadata={**doc_meta, "loc": loc, "snippet": snippet},
                )
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
                nodes.append(node)

    elif source_type == "md":
        # Heading-aware: split at h1-h3 boundaries, then SentenceSplitter within.
        sections = _split_markdown_by_headings(result.full_text)
        all_raw: list[str] = []
        all_ctx: list[str] = []
        all_sections: list[str] = []

        for heading_ctx, section_text in sections:
            raw_chunks = _split_section(
                section_text, splitter, semantic_splitter, semantic_threshold
            )
            ctx = _build_chunk_context(
                doc_meta, section=heading_ctx if heading_ctx else None
            )
            for raw in raw_chunks:
                all_raw.append(raw)
                all_ctx.append(ctx + raw)
                all_sections.append(heading_ctx)

        vectors = embed_provider.embed_texts(all_ctx)

        for i, (ctx_text, raw_text, sec, vector) in enumerate(
            zip(all_ctx, all_raw, all_sections, vectors, strict=True)
        ):
            loc = f"c:{i}"
            chunk_uid = f"{doc_id}::{loc}"
            snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
            meta = {**doc_meta, "loc": loc, "snippet": snippet}
            meta["section"] = sec
            node = TextNode(
                text=ctx_text,
                id_=chunk_uid,
                embedding=vector,
                metadata=meta,
            )
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
            nodes.append(node)

    else:
        # Images or single-page PDFs
        loc_prefix = "img" if source_type == "img" else ""
        raw_chunks = _split_section(
            result.full_text, splitter, semantic_splitter, semantic_threshold
        )
        ctx = _build_chunk_context(doc_meta)
        contextualized = [ctx + c for c in raw_chunks]
        vectors = embed_provider.embed_texts(contextualized)

        for i, (ctx_text, raw_text, vector) in enumerate(
            zip(contextualized, raw_chunks, vectors, strict=True)
        ):
            loc = f"{loc_prefix}:c:{i}" if loc_prefix else f"c:{i}"
            chunk_uid = f"{doc_id}::{loc}"
            snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
            node = TextNode(
                text=ctx_text,
                id_=chunk_uid,
                embedding=vector,
                metadata={**doc_meta, "loc": loc, "snippet": snippet},
            )
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
            nodes.append(node)

    # --- Upsert into store ---
    store.upsert_nodes(nodes)
    logger.info(f"Upserted {len(nodes)} chunks: {doc_id}")


@task
def delete_docs_task(doc_ids: list[str]) -> None:
    """Remove all chunk nodes for the given doc_ids."""
    if doc_ids:
        store: LanceDBStore = _RUNTIME["store"]
        store.delete_by_doc_ids(doc_ids)


@task
def index_stats_task(
    to_add_count: int,
    to_delete_count: int,
    run_seconds: float | None = None,
) -> None:
    """Log counts and optional duration."""
    logger = get_run_logger()
    logger.info(
        f"Index stats: added/updated={to_add_count}, deleted={to_delete_count}, "
        f"seconds={run_seconds:.1f}" if run_seconds else
        f"Index stats: added/updated={to_add_count}, deleted={to_delete_count}"
    )


@task
def write_index_metadata_task(
    index_root: str | Path,
    doc_count: int,
    chunk_count: int | None,
    failed_docs: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Write index_metadata.json for file_status (last_run_at, counts, failures, warnings)."""
    import json
    from datetime import datetime, timezone
    path = Path(index_root) / "index_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": doc_count,
        "chunk_count": chunk_count,
    }
    if failed_docs:
        meta["failed_count"] = len(failed_docs)
        meta["failed_docs"] = failed_docs[:20]
    if warnings:
        meta["warnings"] = warnings[:50]
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# --- Flow ---


@flow(name="index_vault_flow")
def index_vault_flow(config_path: str = "config.yaml") -> None:
    """Scan vault, diff with store, process new/updated docs, delete removed, log stats."""
    _RUNTIME.clear()
    import time
    logger = get_run_logger()
    config = load_config(config_path)
    vault_root = Path(config["documents_root"])
    index_root = Path(config["index_root"])
    scan_cfg = config.get("scan", {})
    include = scan_cfg.get("include", ["**/*.md"])
    exclude = scan_cfg.get("exclude", [".obsidian/**", ".trash/**", "**/.DS_Store"])
    chunk_cfg = config.get("chunking", {})

    # --- Build components from config (stored in _RUNTIME for tasks) ---
    store = LanceDBStore(index_root, config.get("lancedb", {}).get("table", "chunks"))

    embed_provider = build_embed_provider(config)

    splitter = SentenceSplitter(
        chunk_size=chunk_cfg.get("max_chars", 1800),
        chunk_overlap=chunk_cfg.get("overlap", 200),
    )

    # OCR provider (may be None if disabled or provider="none")
    ocr_provider = build_ocr_provider(config)
    if ocr_provider:
        logger.info(f"OCR enabled: {config.get('ocr', {}).get('provider', 'none')}")
    else:
        logger.info("OCR disabled (set ocr.enabled=true in config to enable)")

    # Semantic splitter (optional — for large sections that need topic-boundary detection)
    semantic_cfg = chunk_cfg.get("semantic", {})
    semantic_splitter = None
    semantic_threshold = 0
    if semantic_cfg.get("enabled", False):
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

            # Build dedicated embed provider for semantic chunking (defaults to main provider)
            from providers.embed.semantic_adapter import SemanticEmbeddingAdapter

            sem_provider = semantic_cfg.get("provider")
            if sem_provider == "ollama":
                from providers.embed.ollama_embed import OllamaEmbedProvider

                sem_embed_provider = OllamaEmbedProvider(
                    base_url=semantic_cfg.get("base_url", "http://localhost:11434"),
                    model_name=semantic_cfg.get("model", "qwen3-embedding:0.6b"),
                )
                semantic_embed = SemanticEmbeddingAdapter(sem_embed_provider)
            else:
                semantic_embed = SemanticEmbeddingAdapter(embed_provider)

            semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=semantic_cfg.get("buffer_size", 1),
                breakpoint_percentile_threshold=semantic_cfg.get("breakpoint_percentile", 95),
                embed_model=semantic_embed,
            )
            semantic_threshold = semantic_cfg.get(
                "threshold", chunk_cfg.get("max_chars", 1800) * 2
            )
            logger.info(
                "Semantic chunking enabled (model=%s, threshold=%d chars)",
                semantic_embed.model_name, semantic_threshold,
            )
        except Exception as exc:
            logger.warning("Failed to load semantic chunking model, falling back to SentenceSplitter: %s", exc)
            _RUNTIME.setdefault("_warnings", []).append(f"semantic_chunking_failed: {exc}")

    # LLM enrichment (optional — extracts summary, entities, topics from each doc)
    llm_generator = None
    enrichment_cfg = config.get("enrichment", {})
    if enrichment_cfg.get("enabled", False):
        try:
            from providers.llm import build_llm_provider
            llm_generator = build_llm_provider(config)
            if llm_generator:
                logger.info(
                    "LLM enrichment enabled (model=%s)",
                    enrichment_cfg.get("model", "qwen3:14b-udq6"),
                )
        except Exception as exc:
            logger.warning("Failed to load LLM enrichment model: %s", exc)

    _RUNTIME["store"] = store
    _RUNTIME["embed_provider"] = embed_provider
    _RUNTIME["splitter"] = splitter
    _RUNTIME["semantic_splitter"] = semantic_splitter
    _RUNTIME["semantic_threshold"] = semantic_threshold
    _RUNTIME["ocr_provider"] = ocr_provider
    _RUNTIME["llm_generator"] = llm_generator
    _RUNTIME["config"] = config

    # --- Run pipeline ---
    t0 = time.perf_counter()
    scanned = scan_vault_task(vault_root, include, exclude)
    stored_mtimes = store.list_doc_mtimes()
    to_add_or_update, to_delete = diff_index_task(scanned, stored_mtimes)

    failed_docs: list[str] = []
    for doc in to_add_or_update:
        try:
            process_doc_task(doc)
        except Exception as exc:
            failed_docs.append(doc["doc_id"])
            logger.error("Skipping %s after retries exhausted: %s", doc["doc_id"], exc)

    if failed_docs:
        logger.warning("Failed to process %d docs: %s", len(failed_docs), failed_docs[:20])

    delete_docs_task(to_delete)

    # Rebuild FTS index for keyword search (BM25/tantivy)
    if to_add_or_update or to_delete:
        logger.info("Rebuilding FTS index...")
        try:
            store.create_fts_index()
        except Exception as exc:
            logger.error("FTS index rebuild failed: %s", exc)
            _RUNTIME.setdefault("_warnings", []).append(f"fts_rebuild_failed: {exc}")

    run_seconds = time.perf_counter() - t0
    index_stats_task(len(to_add_or_update), len(to_delete), run_seconds)
    doc_count = len(store.list_doc_ids())
    write_index_metadata_task(
        index_root, doc_count, store.count_chunks(),
        failed_docs or None,
        _RUNTIME.get("_warnings") or None,
    )
    logger.info(f"index_vault_flow finished in {run_seconds:.1f}s")
