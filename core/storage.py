"""Storage interface: indexer and MCP depend on this, not on LanceDB directly."""

from typing import Protocol

from llama_index.core.schema import TextNode


class SearchHit:
    """One search result with document metadata."""

    def __init__(
        self,
        doc_id: str,
        loc: str,
        snippet: str,
        text: str,
        score: float,
        source_type: str | None = None,
        title: str | None = None,
        tags: str | None = None,
        folder: str | None = None,
        status: str | None = None,
        created: str | None = None,
        mtime: float = 0.0,
        # Frontmatter metadata
        description: str = "",
        author: str = "",
        keywords: str = "",
        custom_meta: str = "",
        # LLM enrichment fields (enr_ prefix separates from frontmatter namespace)
        enr_summary: str = "",
        enr_doc_type: str = "",
        enr_entities_people: str = "",
        enr_entities_places: str = "",
        enr_entities_orgs: str = "",
        enr_entities_dates: str = "",
        enr_topics: str = "",
        enr_keywords: str = "",
        enr_key_facts: str = "",
        # Taxonomy-suggested fields from enrichment
        enr_suggested_tags: str = "",
        enr_suggested_folder: str = "",
        # Importance score (0.0-1.0) and its source
        enr_importance: str = "",
        enr_importance_source: str = "",
        # Dynamic metadata (fields not in the hardcoded set above)
        extra_metadata: dict[str, str] | None = None,
    ):
        self.doc_id = doc_id
        self.loc = loc
        self.snippet = snippet
        self.text = text
        self.score = score
        self.source_type = source_type
        self.title = title
        self.tags = tags          # comma-separated string, e.g. "recipe,korean"
        self.folder = folder      # top-level folder, e.g. "Archive"
        self.status = status      # "active", "archived", "draft", etc.
        self.created = created    # from frontmatter, e.g. "2026-01-15"
        self.mtime = mtime        # file last-modified timestamp
        # Frontmatter metadata
        self.description = description  # from frontmatter "description" field
        self.author = author            # from frontmatter "author" field
        self.keywords = keywords        # from frontmatter "keywords" field, comma-separated
        self.custom_meta = custom_meta  # JSON dict of remaining frontmatter fields
        # LLM enrichment
        self.enr_summary = enr_summary
        self.enr_doc_type = enr_doc_type
        self.enr_entities_people = enr_entities_people
        self.enr_entities_places = enr_entities_places
        self.enr_entities_orgs = enr_entities_orgs
        self.enr_entities_dates = enr_entities_dates
        self.enr_topics = enr_topics
        self.enr_keywords = enr_keywords
        self.enr_key_facts = enr_key_facts
        self.enr_suggested_tags = enr_suggested_tags
        self.enr_suggested_folder = enr_suggested_folder
        self.enr_importance = enr_importance
        self.enr_importance_source = enr_importance_source
        # Dynamic metadata for fields added after initial schema
        self.extra_metadata = extra_metadata or {}

    def __getattr__(self, name: str):
        """Fall through to extra_metadata for dynamic fields like 'section'."""
        # __getattr__ is only called when normal attribute lookup fails
        extra = self.__dict__.get("extra_metadata", {})
        if name in extra:
            return extra[name]
        raise AttributeError(f"'SearchHit' has no attribute {name!r}")

    def __repr__(self) -> str:
        return (
            f"SearchHit(doc_id={self.doc_id!r}, loc={self.loc!r}, "
            f"score={self.score:.4f}, title={self.title!r}, source_type={self.source_type!r})"
        )


class StorageInterface(Protocol):
    """Thin interface for persisting and querying chunks. Implementations: LanceDB, etc."""

    def upsert_nodes(self, nodes: list[TextNode]) -> None:
        """Replace all nodes for each doc_id in this batch."""
        ...

    def delete_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Remove all node rows for the given doc_ids."""
        ...

    def list_doc_ids(self) -> list[str]:
        """Return all doc_ids currently in the store."""
        ...

    def list_doc_mtimes(self) -> dict[str, float]:
        """Return {doc_id: mtime} for all docs (used for incremental diffing)."""
        ...

    def vector_search(
        self, query_vector: list[float], top_k: int, where: str | None = None,
    ) -> list[SearchHit]:
        """Return top_k chunks by vector similarity, with optional SQL prefilter."""
        ...

    def keyword_search(
        self, query: str, top_k: int, where: str | None = None,
    ) -> list[SearchHit]:
        """Return top_k chunks by BM25/FTS keyword relevance, with optional SQL prefilter."""
        ...

    def create_fts_index(self) -> None:
        """Create or rebuild the full-text search index."""
        ...
