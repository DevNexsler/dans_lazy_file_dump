"""LanceDB storage via LlamaIndex's LanceDBVectorStore. Implements our StorageInterface."""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.vector_stores.lancedb.base import TableNotFoundError

from core.storage import SearchHit

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_ENRICHMENT_FIELDS = (
    "enr_summary", "enr_doc_type", "enr_entities_people", "enr_entities_places",
    "enr_entities_orgs", "enr_entities_dates", "enr_topics", "enr_keywords", "enr_key_facts",
    "enr_suggested_tags", "enr_suggested_folder",
)

_EXTRA_META_FIELDS = ("description", "author", "keywords", "custom_meta")

# All metadata keys that map to explicit SearchHit attributes.
# Anything NOT in this set goes into SearchHit.extra_metadata.
_CORE_META_KEYS = {
    "doc_id", "source_type", "mtime", "size", "title", "tags", "folder",
    "status", "created", "loc", "snippet",
    "description", "author", "keywords", "custom_meta",
    "enr_summary", "enr_doc_type", "enr_entities_people", "enr_entities_places",
    "enr_entities_orgs", "enr_entities_dates", "enr_topics", "enr_keywords", "enr_key_facts",
    "enr_suggested_tags", "enr_suggested_folder",
}


def _extract_enrichment(meta: dict) -> dict[str, str]:
    """Pull enrichment + extra metadata fields from metadata, defaulting to empty strings."""
    result = {f: meta.get(f, "") or "" for f in _ENRICHMENT_FIELDS}
    for f in _EXTRA_META_FIELDS:
        result[f] = meta.get(f, "") or ""
    return result


def _extract_extra_metadata(meta: dict) -> dict[str, str]:
    """Collect metadata fields not in the hardcoded core set (e.g. section, sentiment)."""
    return {k: str(v) for k, v in meta.items() if k not in _CORE_META_KEYS and v}


class LanceDBStore:
    """Implements StorageInterface using LlamaIndex's LanceDBVectorStore."""

    def __init__(self, index_root: str | Path, table_name: str = "chunks") -> None:
        self.index_root = str(Path(index_root))
        self.table_name = table_name
        self._vs = LanceDBVectorStore(
            uri=self.index_root,
            table_name=table_name,
            mode="create",  # "create" lets LanceDB create the table if missing, or open if exists
        )
        self._ensure_scalar_index()

    def _ensure_scalar_index(self) -> None:
        """Create a BTREE scalar index on doc_id for O(log n) filtered lookups."""
        try:
            self._vs.table.create_scalar_index("doc_id", index_type="BTREE", replace=True)
        except TableNotFoundError:
            pass  # Table not created yet on first run
        except Exception as e:
            logger.warning("Failed to create scalar index: %s", e)

    @staticmethod
    def _sql_escape(value: str) -> str:
        """Escape single quotes for safe use in SQL WHERE clauses."""
        return value.replace("'", "''")

    @staticmethod
    def _validate_identifier(key: str) -> None:
        """Raise ValueError if *key* is not a safe SQL identifier."""
        if not _SAFE_IDENTIFIER_RE.match(key):
            raise ValueError(f"Unsafe metadata filter key: {key!r}")

    def _metadata_subfields(self) -> set[str]:
        """Return the set of sub-field names inside the metadata struct column."""
        try:
            ds = self._vs.table.to_lance()
            for field in ds.schema:
                if field.name == "metadata":
                    return {sf.name for sf in field.type}
        except Exception:
            pass
        return set()

    def _evolve_metadata_schema(self, new_fields: set[str]) -> None:
        """Add new string sub-fields to the metadata struct column.

        Reads the entire table as Arrow, adds empty-string columns for each
        new field, reconstructs the metadata struct, and replaces the table.
        """
        import lancedb as ldb

        table = self._vs.table
        arrow_table = table.to_arrow()

        # Extract existing metadata struct arrays
        meta_chunked = arrow_table.column("metadata")
        meta_col = meta_chunked.combine_chunks()  # StructArray (not ChunkedArray)
        meta_type = meta_col.type
        existing_names = [meta_type.field(i).name for i in range(meta_type.num_fields)]

        # Build new struct arrays: existing + new fields filled with ""
        n_rows = len(arrow_table)
        arrays = [meta_col.field(name) for name in existing_names]
        fields = [meta_type.field(i) for i in range(meta_type.num_fields)]

        for fname in sorted(new_fields):
            if fname not in existing_names:
                arrays.append(pa.array([""] * n_rows, type=pa.utf8()))
                fields.append(pa.field(fname, pa.utf8()))

        new_struct = pa.StructArray.from_arrays(arrays, fields=fields)

        # Replace metadata column in the table
        col_idx = arrow_table.schema.get_field_index("metadata")
        new_arrow = arrow_table.set_column(col_idx, pa.field("metadata", new_struct.type), new_struct)

        # Drop and recreate the table
        db = ldb.connect(self.index_root)
        db.drop_table(self.table_name)
        db.create_table(self.table_name, new_arrow)

        # Reconnect LanceDBVectorStore to the new table
        self._vs = LanceDBVectorStore(
            uri=self.index_root,
            table_name=self.table_name,
            mode="create",
        )
        self._ensure_scalar_index()
        logger.info("Schema evolved: added metadata fields %s", new_fields)

    # --- WHERE clause builder ---

    def _build_where_clause(
        self,
        doc_id_prefix: str | None = None,
        source_type: str | None = None,
        status: str | None = None,
        folder: str | None = None,
        tags: str | None = None,
        enr_doc_type: str | None = None,
        enr_topics: str | None = None,
        metadata_filters: dict[str, str] | None = None,
    ) -> str | None:
        """Build a SQL WHERE clause from filter parameters for LanceDB prefilter.

        Returns None if no filters are active.
        """
        parts: list[str] = []

        # Exact match on metadata fields (case-insensitive)
        if source_type:
            parts.append(f"lower(metadata.source_type) = '{self._sql_escape(source_type.lower())}'")
        if status:
            parts.append(f"lower(metadata.status) = '{self._sql_escape(status.lower())}'")
        if folder:
            parts.append(f"lower(metadata.folder) = '{self._sql_escape(folder.lower())}'")

        # Prefix match on top-level doc_id column (case-sensitive — paths)
        if doc_id_prefix:
            parts.append(f"doc_id LIKE '{self._sql_escape(doc_id_prefix)}%'")

        # Comma-separated OR fields (tags, enr_doc_type, enr_topics) — case-insensitive
        for field, value in [
            ("tags", tags),
            ("enr_doc_type", enr_doc_type),
            ("enr_topics", enr_topics),
        ]:
            if value:
                items = [item.strip() for item in value.split(",") if item.strip()]
                if items:
                    or_parts = [
                        f"lower(metadata.{field}) LIKE '%{self._sql_escape(item.lower())}%'"
                        for item in items
                    ]
                    parts.append(f"({' OR '.join(or_parts)})")

        # Arbitrary metadata key=value pairs (case-insensitive)
        if metadata_filters:
            for key, val in metadata_filters.items():
                self._validate_identifier(key)
                parts.append(f"lower(metadata.{key}) = '{self._sql_escape(str(val).lower())}'")

        return " AND ".join(parts) if parts else None

    # --- Shared row-to-hit converter ---

    @staticmethod
    def _row_to_hit(row: dict) -> "SearchHit":
        """Convert a raw LanceDB row dict to a SearchHit.

        Handles both vector search (_distance → similarity) and FTS (_score).
        """
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        text = row.get("text", "") or ""

        doc_id = (
            meta.get("doc_id")
            or row.get("doc_id")
            or row.get("_node_ref_doc_id", "")
        )
        loc = meta.get("loc") or row.get("loc", "")
        snippet = meta.get("snippet") or text[:200]

        # Score: vector search returns _distance (cosine), FTS returns _score
        if "_distance" in row:
            score = 1.0 - float(row["_distance"])
        elif "_score" in row:
            score = float(row["_score"])
        elif "score" in row:
            score = float(row["score"])
        else:
            score = 0.0

        raw_mtime = meta.get("mtime") or row.get("mtime") or 0.0
        combined_meta = {**row, **meta} if meta else row

        return SearchHit(
            doc_id=doc_id,
            loc=loc,
            snippet=snippet,
            text=text,
            score=score,
            source_type=meta.get("source_type") or row.get("source_type"),
            title=meta.get("title") or row.get("title"),
            tags=meta.get("tags") or row.get("tags"),
            folder=meta.get("folder") or row.get("folder"),
            status=meta.get("status") or row.get("status"),
            created=meta.get("created") or row.get("created"),
            mtime=float(raw_mtime) if raw_mtime else 0.0,
            **_extract_enrichment(combined_meta),
            extra_metadata=_extract_extra_metadata(meta),
        )

    # --- StorageInterface methods ---

    def upsert_nodes(self, nodes: list[TextNode]) -> None:
        """Delete existing nodes for each doc_id, then add new ones."""
        if not nodes:
            return

        # Detect new metadata fields and evolve schema if needed
        existing_subfields = self._metadata_subfields()
        if existing_subfields:  # table already has data
            incoming_keys: set[str] = set()
            for n in nodes:
                if n.metadata:
                    incoming_keys.update(n.metadata.keys())
            new_fields = incoming_keys - existing_subfields
            if new_fields:
                self._evolve_metadata_schema(new_fields)

        # Collect distinct doc_ids from this batch
        doc_ids = {n.ref_doc_id for n in nodes if n.ref_doc_id}
        # Delete old data for those doc_ids first
        for doc_id in doc_ids:
            try:
                self._vs.delete(doc_id)
            except TableNotFoundError:
                pass  # Table not created yet on first run
            except Exception as e:
                logger.warning("Failed to delete old data for %s: %s", doc_id, e)
        # Add new nodes
        try:
            self._vs.add(nodes)
        except Exception:
            logger.critical(
                "Failed to add %d nodes for doc_ids=%s after old chunks were deleted; "
                "these docs will self-heal on the next index run",
                len(nodes), sorted(doc_ids),
            )
            raise

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Remove all nodes for the given doc_ids."""
        for doc_id in doc_ids:
            try:
                self._vs.delete(doc_id)
            except TableNotFoundError:
                pass  # Table not created yet — nothing to delete
            except Exception as e:
                logger.warning("Failed to delete doc %s: %s", doc_id, e)

    def list_doc_ids(self) -> list[str]:
        """Return all distinct doc_ids in the store.

        Uses Lance SQL DISTINCT — reads only the doc_id column, no vectors or text.
        Raises on failure so callers get a proper error (not empty results).
        """
        try:
            ds = self._vs.table.to_lance()
        except TableNotFoundError:
            return []  # Table not created yet — legitimately empty
        batches = ds.sql("SELECT DISTINCT doc_id FROM dataset WHERE doc_id IS NOT NULL").build().to_batch_records()
        if not batches:
            return []
        return pa.Table.from_batches(batches)["doc_id"].to_pylist()

    def list_doc_mtimes(self) -> dict[str, float]:
        """Return {doc_id: mtime} for all docs in the store.

        Uses Lance SQL GROUP BY — reads only doc_id + metadata.mtime, no vectors or text.
        Raises on failure so callers get a proper error (not empty results).
        """
        try:
            ds = self._vs.table.to_lance()
        except TableNotFoundError:
            return {}  # Table not created yet — legitimately empty
        batches = ds.sql(
            "SELECT doc_id, MAX(metadata.mtime) AS mtime "
            "FROM dataset WHERE doc_id IS NOT NULL GROUP BY doc_id"
        ).build().to_batch_records()
        if not batches:
            return {}
        t = pa.Table.from_batches(batches)
        doc_ids = t["doc_id"].to_pylist()
        mtimes = t["mtime"].to_pylist()
        return {d: float(m) if m is not None else 0.0 for d, m in zip(doc_ids, mtimes)}

    def count_chunks(self) -> int:
        """Return total number of chunks (rows) in the store. O(1) via LanceDB.

        Raises on failure so callers get a proper error (not zero).
        """
        try:
            return self._vs.table.count_rows()
        except TableNotFoundError:
            return 0  # Table not created yet — legitimately empty

    def vector_search(
        self, query_vector: list[float], top_k: int, where: str | None = None,
    ) -> list[SearchHit]:
        """Vector search via direct LanceDB query with optional prefilter.

        Raises on failure so the caller (hybrid_search) can track degradation.
        """
        try:
            table = self._vs.table
        except TableNotFoundError:
            return []  # Table not created yet — legitimately empty
        q = table.search(query_vector, query_type="vector")
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(top_k).to_list()
        return [self._row_to_hit(row) for row in rows]

    # --- Full-Text Search (BM25/tantivy) ---

    def create_fts_index(self) -> None:
        """Create or rebuild the tantivy FTS index on the text column.

        Should be called after indexing completes. Uses replace=True to rebuild
        from scratch (tantivy FTS does not support incremental updates).
        Raises on failure so the caller can track the error.
        """
        table = self._vs.table
        text_key = getattr(self._vs, "text_key", "text")
        table.create_fts_index(text_key, use_tantivy=True, replace=True)
        logger.info("FTS index created/rebuilt on column %r", text_key)

    def fts_available(self) -> bool:
        """Check if the FTS/tantivy index is operational (health check for file_status)."""
        try:
            self._vs.table.search("test", query_type="fts").limit(1).to_list()
            return True
        except Exception:
            return False

    def keyword_search(
        self, query: str, top_k: int = 50, where: str | None = None,
    ) -> list[SearchHit]:
        """BM25/FTS keyword search via LanceDB tantivy index with optional prefilter.

        Returns SearchHit list ranked by BM25 relevance. Raises on FTS failure
        so the caller (hybrid_search) can track degradation in diagnostics.
        """
        if not query.strip():
            return []
        try:
            table = self._vs.table
        except TableNotFoundError:
            return []  # Table not created yet — legitimately empty
        q = table.search(query, query_type="fts")
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(top_k).to_list()
        return [self._row_to_hit(row) for row in rows]

    _RECENT_DOC_FIELDS = ("title", "source_type", "folder", "tags", "status", "created", "keywords")

    def list_recent_docs(
        self,
        limit: int = 20,
        source_type: str | None = None,
        folder: str | None = None,
    ) -> list[dict]:
        """Return recently modified docs sorted by mtime descending.

        Uses Lance SQL with GROUP BY, ORDER BY, and LIMIT — all server-side.
        No vectors or text loaded. Raises on failure so callers get a proper error.
        """
        try:
            ds = self._vs.table.to_lance()
        except TableNotFoundError:
            return []  # Table not created yet — legitimately empty
        available = self._metadata_subfields()

        where_parts: list[str] = ["doc_id IS NOT NULL"]
        if source_type and "source_type" in available:
            where_parts.append(f"metadata.source_type = '{self._sql_escape(source_type)}'")
        if folder and "folder" in available:
            where_parts.append(f"metadata.folder = '{self._sql_escape(folder)}'")
        where_clause = " AND ".join(where_parts)

        # Build SELECT with only fields that exist in the metadata struct
        select_parts = ["doc_id"]
        output_fields = ["doc_id"]
        if "mtime" in available:
            select_parts.append("MAX(metadata.mtime) AS mtime")
            output_fields.append("mtime")
        for f in self._RECENT_DOC_FIELDS:
            if f in available:
                select_parts.append(f"MAX(metadata.{f}) AS {f}")
                output_fields.append(f)

        sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM dataset WHERE {where_clause} "
            "GROUP BY doc_id "
            f"ORDER BY mtime DESC LIMIT {limit}"
        )
        batches = ds.sql(sql).build().to_batch_records()
        if not batches:
            return []
        t = pa.Table.from_batches(batches)
        result: list[dict] = []
        for i in range(len(t)):
            rec: dict = {}
            for f in output_fields:
                try:
                    rec[f] = t[f][i].as_py()
                except (KeyError, IndexError):
                    rec[f] = None
            result.append(rec)
        return result

    _FACET_KEY_MAP = {
        "folder": "folders",
        "source_type": "source_types",
        "status": "statuses",
        "enr_doc_type": "doc_types",
        "author": "authors",
        "enr_topics": "topics",
        "enr_keywords": "keywords",
        "enr_entities_people": "entities_people",
        "enr_entities_places": "entities_places",
        "enr_entities_orgs": "entities_orgs",
    }

    # Metadata fields that should NOT be faceted (structural/numeric, not categorical)
    _NON_FACET_FIELDS = {"doc_id", "loc", "snippet", "mtime", "size", "title", "created"}

    def facets(self) -> dict:
        """Return distinct values and doc counts for all filterable fields.

        Uses column projection to load only metadata fields — no vectors or text.
        Deduplicates by doc_id so counts reflect documents, not chunks.
        Tags are split on commas and counted individually.
        Dynamic fields (not in _FACET_KEY_MAP) are automatically included.

        TODO(perf): This does a full table scan — O(chunks). Fine at ~15K chunks
        (~200ms) but will get slow at 50K+. When that happens, write a
        facets_cache.json at end of index_vault_flow and read from it here.
        Cache invalidation is trivial: indexing is the only writer.
        """
        total_chunks = self.count_chunks()
        if total_chunks == 0:
            return {"total_docs": 0, "total_chunks": 0}

        try:
            table = self._vs.table
        except TableNotFoundError:
            return {"total_docs": 0, "total_chunks": 0}  # Table not created yet

        available = self._metadata_subfields()

        # Discover dynamic fields: anything in schema but not in the hardcoded
        # facet map and not in the non-facet exclusion set
        all_facet_fields = set(self._FACET_KEY_MAP.keys())
        dynamic_fields = available - all_facet_fields - self._NON_FACET_FIELDS - {"tags"}

        projection = {"doc_id": "doc_id"}
        for field in self._FACET_KEY_MAP:
            if field in available:
                projection[field] = f"metadata.{field}"
        for field in dynamic_fields:
            projection[field] = f"metadata.{field}"
        if "tags" in available:
            projection["tags"] = "metadata.tags"

        t = table.search(None).select(projection).to_arrow()

        if len(t) == 0:
            return {"total_docs": 0, "total_chunks": total_chunks}

        # Deduplicate by doc_id and count facet values
        seen: set[str] = set()
        all_counted_fields = set(self._FACET_KEY_MAP.keys()) | dynamic_fields
        counters: dict[str, Counter] = {f: Counter() for f in all_counted_fields}
        tag_counter: Counter = Counter()

        doc_ids = t["doc_id"].to_pylist()
        field_columns: dict[str, list] = {}
        for f in all_counted_fields:
            try:
                field_columns[f] = t[f].to_pylist()
            except KeyError:
                field_columns[f] = [None] * len(t)
        try:
            tags_col = t["tags"].to_pylist()
        except KeyError:
            tags_col = [None] * len(t)

        for i, did in enumerate(doc_ids):
            if not did or did in seen:
                continue
            seen.add(did)

            for f in all_counted_fields:
                val = field_columns[f][i]
                if val and str(val).strip():
                    for item in str(val).split(","):
                        item = item.strip()
                        if item:
                            counters[f][item] += 1

            raw_tags = tags_col[i]
            if raw_tags and str(raw_tags).strip():
                for tag in str(raw_tags).split(","):
                    tag = tag.strip()
                    if tag:
                        tag_counter[tag] += 1

        def _to_list(counter: Counter) -> list[dict]:
            return [{"value": v, "count": c} for v, c in counter.most_common()]

        result: dict = {
            "tags": _to_list(tag_counter),
            "total_docs": len(seen),
            "total_chunks": total_chunks,
        }
        for f, key in self._FACET_KEY_MAP.items():
            result[key] = _to_list(counters[f])
        # Dynamic fields use their own name as the result key
        for f in dynamic_fields:
            facet_list = _to_list(counters[f])
            if facet_list:  # only include if there are non-empty values
                result[f] = facet_list

        return result

    def get_vector(self, chunk_uid: str) -> list[float] | None:
        """Retrieve the stored embedding vector for a chunk by its UID (doc_id::loc).

        Used by MMR diversity and cosine fallback reranking. Returns None if
        the chunk is not found or the table doesn't exist yet.
        """
        try:
            table = self._vs.table
        except TableNotFoundError:
            return None
        rows = (
            table.search(None)
            .where(f"id = '{self._sql_escape(chunk_uid)}'", prefilter=True)
            .select(["vector"])
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        vec = rows[0].get("vector")
        if vec is None:
            return None
        # Convert from Arrow/numpy to plain list if needed
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)

    def get_chunk(self, doc_id: str, loc: str) -> SearchHit | None:
        """Get a single chunk by doc_id and loc.

        Uses predicate pushdown with BTREE index — O(log n), no full scan.
        Loads text + metadata for the matching row only; no vectors.
        """
        try:
            table = self._vs.table
        except TableNotFoundError:
            return None  # Table not created yet — legitimately empty
        esc_doc = self._sql_escape(doc_id)
        esc_loc = self._sql_escape(loc)
        rows = (
            table.search(None)
            .where(f"doc_id = '{esc_doc}' AND metadata.loc = '{esc_loc}'", prefilter=True)
            .select(["doc_id", "text", "metadata"])
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        return self._row_to_hit(rows[0])

    def get_doc_chunks(self, doc_id: str) -> list[SearchHit]:
        """Get all chunks for a document, sorted by loc.

        Uses predicate pushdown with BTREE index — O(log n + k), no full scan.
        Loads text + metadata for matching rows only; no vectors.
        """
        try:
            table = self._vs.table
        except TableNotFoundError:
            return []  # Table not created yet — legitimately empty
        esc_doc = self._sql_escape(doc_id)
        rows = (
            table.search(None)
            .where(f"doc_id = '{esc_doc}'", prefilter=True)
            .select(["doc_id", "text", "metadata"])
            .to_list()
        )
        hits = [self._row_to_hit(row) for row in rows]
        hits.sort(key=lambda h: h.loc)
        return hits
