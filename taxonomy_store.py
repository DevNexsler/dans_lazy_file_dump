"""LanceDB-backed taxonomy store for tags, folders, and doc_types.

Manages a controlled vocabulary with embedded descriptions for semantic
matching. Uses lancedb directly (not LlamaIndex wrapper) since taxonomy
rows aren't document chunks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import lancedb as ldb
import pyarrow as pa

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_:-]*$")

_SCHEMA = pa.schema([
    pa.field("id", pa.utf8()),
    pa.field("kind", pa.utf8()),
    pa.field("name", pa.utf8()),
    pa.field("description", pa.utf8()),
    pa.field("aliases", pa.utf8()),
    pa.field("parent", pa.utf8()),
    pa.field("status", pa.utf8()),
    pa.field("usage_count", pa.int64()),
    pa.field("ai_managed", pa.int64()),
    pa.field("contents_type", pa.utf8()),
    pa.field("created_by", pa.utf8()),
    # vector column added dynamically based on embed_fn dimension
])


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


class TaxonomyStore:
    """CRUD store for taxonomy entries backed by a LanceDB table."""

    def __init__(
        self,
        index_root: str | Path,
        table_name: str = "taxonomy",
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.index_root = str(Path(index_root))
        self.table_name = table_name
        self.embed_fn = embed_fn
        self._db = ldb.connect(self.index_root)
        self._table = self._open_or_create()

    def _open_or_create(self):
        try:
            tables = self._db.list_tables()
        except Exception:
            tables = []
        if self.table_name in tables:
            return self._db.open_table(self.table_name)
        return None  # created lazily on first add

    def _ensure_table(self) -> None:
        """Re-check disk and open the table if it exists but _table is None.

        Handles stale connection state (container restart, volume remount,
        long idle) where list_tables() missed the table at init time but
        it exists on disk now.
        """
        if self._table is not None:
            return
        try:
            self._table = self._db.open_table(self.table_name)
            logger.info("Taxonomy table '%s' found on disk (was not open)", self.table_name)
        except Exception:
            pass  # table truly doesn't exist

    def _create_table(self, data: list[dict]) -> None:
        # Guard: table may exist on disk even though _table is None
        self._ensure_table()
        if self._table is not None:
            # Table exists — just add data instead of re-creating
            self._table.add(data)
            return
        self._table = self._db.create_table(self.table_name, data)
        try:
            self._table.create_scalar_index("id", index_type="BTREE", replace=True)
        except Exception as e:
            logger.warning("Failed to create scalar index on id: %s", e)
        try:
            self._table.create_scalar_index("kind", index_type="BTREE", replace=True)
        except Exception as e:
            logger.warning("Failed to create scalar index on kind: %s", e)

    def _embed(self, text: str) -> list[float]:
        if self.embed_fn is None:
            raise ValueError("embed_fn is required for taxonomy operations")
        return self.embed_fn(text)

    def _build_row(
        self,
        kind: str,
        name: str,
        description: str,
        aliases: str = "",
        parent: str = "",
        status: str = "active",
        usage_count: int = 0,
        ai_managed: int = 1,
        contents_type: str = "",
        created_by: str = "AI",
    ) -> dict:
        row_id = f"{kind}:{name}"
        vector = self._embed(description) if description.strip() else self._embed(name)
        return {
            "id": row_id,
            "kind": kind,
            "name": name,
            "description": description,
            "aliases": aliases,
            "parent": parent,
            "status": status,
            "usage_count": usage_count,
            "ai_managed": ai_managed,
            "contents_type": contents_type,
            "created_by": created_by,
            "vector": vector,
        }

    # --- CRUD ---

    def add(
        self,
        kind: str,
        name: str,
        description: str,
        aliases: str = "",
        parent: str = "",
        status: str = "active",
        usage_count: int = 0,
        ai_managed: int = 1,
        contents_type: str = "",
        created_by: str = "AI",
    ) -> dict:
        """Add a taxonomy entry. Returns the row dict."""
        row = self._build_row(
            kind, name, description, aliases, parent, status,
            usage_count, ai_managed, contents_type, created_by,
        )
        if self._table is None:
            self._create_table([row])
        else:
            self._table.add([row])
        return row

    def add_batch(self, rows: list[dict]) -> int:
        """Add multiple pre-built rows at once. Returns count added.

        Each row must have all required fields plus a 'vector' key.
        """
        if not rows:
            return 0
        if self._table is None:
            self._create_table(rows)
        else:
            self._table.add(rows)
        return len(rows)

    def get(self, entry_id: str) -> dict | None:
        """Get a single entry by id."""
        self._ensure_table()
        if self._table is None:
            return None
        esc = _sql_escape(entry_id)
        rows = (
            self._table.search(None)
            .where(f"id = '{esc}'", prefilter=True)
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        row = rows[0]
        row.pop("vector", None)
        row.pop("_distance", None)
        return row

    def update(self, entry_id: str, **fields) -> dict | None:
        """Update fields of an existing entry. Re-embeds if description changed."""
        existing = self.get(entry_id)
        if existing is None:
            return None

        existing.update(fields)

        if "description" in fields:
            desc = fields["description"]
            existing["vector"] = self._embed(desc) if desc.strip() else self._embed(existing["name"])

        # LanceDB doesn't have row-level update; delete + re-add
        esc = _sql_escape(entry_id)
        self._table.delete(f"id = '{esc}'")
        # Re-fetch vector if not already set
        if "vector" not in existing:
            existing["vector"] = self._embed(existing.get("description") or existing["name"])
        self._table.add([existing])
        result = dict(existing)
        result.pop("vector", None)
        return result

    def delete(self, entry_id: str) -> bool:
        """Hard delete an entry. Returns True if found and deleted."""
        self._ensure_table()
        if self._table is None:
            return False
        existing = self.get(entry_id)
        if existing is None:
            return False
        esc = _sql_escape(entry_id)
        self._table.delete(f"id = '{esc}'")
        return True

    def increment_usage(self, entry_id: str) -> None:
        """Bump usage_count by 1."""
        existing = self.get(entry_id)
        if existing is not None:
            self.update(entry_id, usage_count=existing.get("usage_count", 0) + 1)

    # --- Query ---

    def list_by_kind(self, kind: str, status: str = "active") -> list[dict]:
        """List entries filtered by kind and status."""
        self._ensure_table()
        if self._table is None:
            return []
        esc_kind = _sql_escape(kind)
        esc_status = _sql_escape(status)
        rows = (
            self._table.search(None)
            .where(f"kind = '{esc_kind}' AND status = '{esc_status}'", prefilter=True)
            .limit(1000)
            .to_list()
        )
        for r in rows:
            r.pop("vector", None)
            r.pop("_distance", None)
        return rows

    def search(self, query: str, kind: str | None = None, top_k: int = 10) -> list[dict]:
        """Semantic search on embedded descriptions."""
        self._ensure_table()
        if self._table is None:
            return []
        vector = self._embed(query)
        q = self._table.search(vector, query_type="vector")
        if kind:
            q = q.where(f"kind = '{_sql_escape(kind)}'", prefilter=True)
        rows = q.limit(top_k).to_list()
        for r in rows:
            r.pop("vector", None)
        return rows

    def fts_search(self, query: str, kind: str | None = None, top_k: int = 10) -> list[dict]:
        """Full-text search on description field using tantivy."""
        self._ensure_table()
        if self._table is None:
            return []
        try:
            q = self._table.search(query, query_type="fts")
            if kind:
                q = q.where(f"kind = '{_sql_escape(kind)}'", prefilter=True)
            rows = q.limit(top_k).to_list()
        except Exception:
            # FTS index may not exist yet
            return []
        for r in rows:
            r.pop("vector", None)
            r.pop("_score", None)
        return rows

    def create_fts_index(self) -> None:
        """Create or rebuild the tantivy FTS index on description."""
        if self._table is None:
            return
        self._table.create_fts_index("description", use_tantivy=True, replace=True)
        logger.info("Taxonomy FTS index created on description")

    def resolve_aliases(self, names: list[str]) -> list[str]:
        """Map alias names to canonical names. Unknown names pass through unchanged."""
        if self._table is None or not names:
            return names
        # Build alias -> canonical map
        all_rows = self._table.search(None).where("status = 'active'", prefilter=True).limit(10000).to_list()
        alias_map: dict[str, str] = {}
        for row in all_rows:
            canonical = row["name"]
            alias_map[canonical.lower()] = canonical
            for alias in (row.get("aliases") or "").split(","):
                alias = alias.strip().lower()
                if alias:
                    alias_map[alias] = canonical
        result = []
        for name in names:
            resolved = alias_map.get(name.strip().lower(), name)
            if resolved not in result:
                result.append(resolved)
        return result

    # --- Enrichment helper ---

    def format_for_prompt(self, kind: str | None = None) -> str:
        """Render active entries as compact text for LLM prompt injection.

        Returns a string block like:
            ## Available Tags
            - machine-learning: ML models, training, inference...
            - property-management: Real estate property operations...

            ## Available Folders
            - 0-AI/: AI collaboration hub
        """
        if self._table is None:
            return ""

        kinds = [kind] if kind else ["tag", "folder", "doc_type"]
        sections: list[str] = []

        kind_labels = {"tag": "Tags", "folder": "Folders", "doc_type": "Document Types"}

        for k in kinds:
            entries = self.list_by_kind(k, status="active")
            if not entries:
                continue
            lines = [f"## Available {kind_labels.get(k, k.title() + 's')}"]
            for e in sorted(entries, key=lambda x: x["name"]):
                desc = e.get("description", "")
                if desc:
                    lines.append(f"- {e['name']}: {desc[:80]}")
                else:
                    lines.append(f"- {e['name']}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def count(self, kind: str | None = None) -> int:
        """Count entries, optionally filtered by kind."""
        self._ensure_table()
        if self._table is None:
            return 0
        if kind:
            rows = self._table.search(None).where(f"kind = '{_sql_escape(kind)}'", prefilter=True).limit(100000).to_list()
            return len(rows)
        return self._table.count_rows()
