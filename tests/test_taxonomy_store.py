"""Tests for taxonomy_store.py — CRUD, search, alias resolution, format_for_prompt."""

import tempfile
from pathlib import Path

import pytest

from taxonomy_store import TaxonomyStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DIM = 8  # tiny vectors for testing


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedder: hash text to a fixed-dimension vector."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    return [float(b) / 255.0 for b in h[:_DIM]]


@pytest.fixture
def store(tmp_path):
    """Fresh TaxonomyStore in a temp directory."""
    return TaxonomyStore(str(tmp_path), table_name="taxonomy", embed_fn=_fake_embed)


@pytest.fixture
def populated_store(store):
    """Store with a few entries pre-loaded."""
    store.add("tag", "machine-learning", "ML models, training, inference", aliases="ml,deep-learning")
    store.add("tag", "property-management", "Real estate property operations")
    store.add("folder", "0-AI/", "AI collaboration hub", contents_type="notes", ai_managed=1)
    store.add("folder", "Projects/", "Active project files", contents_type="mixed")
    store.add("doc_type", "report", "Formal reports and analysis documents")
    return store


# ---------------------------------------------------------------------------
# CRUD Tests
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_creates_entry(self, store):
        row = store.add("tag", "test-tag", "A test tag")
        assert row["id"] == "tag:test-tag"
        assert row["kind"] == "tag"
        assert row["name"] == "test-tag"

    def test_add_returns_vector(self, store):
        row = store.add("tag", "test", "desc")
        assert "vector" in row
        assert len(row["vector"]) == _DIM

    def test_add_to_empty_store(self, store):
        """First add should create the table."""
        assert store._table is None
        store.add("tag", "first", "first entry")
        assert store._table is not None


class TestGet:
    def test_get_existing(self, populated_store):
        result = populated_store.get("tag:machine-learning")
        assert result is not None
        assert result["name"] == "machine-learning"
        assert result["description"] == "ML models, training, inference"
        assert "vector" not in result  # vector stripped from get()

    def test_get_nonexistent(self, populated_store):
        assert populated_store.get("tag:nonexistent") is None

    def test_get_from_empty(self, store):
        assert store.get("tag:anything") is None


class TestUpdate:
    def test_update_description(self, populated_store):
        result = populated_store.update("tag:machine-learning", description="Updated ML desc")
        assert result is not None
        assert result["description"] == "Updated ML desc"
        # Verify persisted
        fetched = populated_store.get("tag:machine-learning")
        assert fetched["description"] == "Updated ML desc"

    def test_update_status(self, populated_store):
        populated_store.update("tag:property-management", status="archived")
        fetched = populated_store.get("tag:property-management")
        assert fetched["status"] == "archived"

    def test_update_nonexistent_returns_none(self, populated_store):
        assert populated_store.update("tag:nope", description="x") is None


class TestDelete:
    def test_delete_existing(self, populated_store):
        assert populated_store.delete("tag:property-management") is True
        assert populated_store.get("tag:property-management") is None

    def test_delete_nonexistent(self, populated_store):
        assert populated_store.delete("tag:nope") is False

    def test_delete_from_empty(self, store):
        assert store.delete("tag:anything") is False


class TestIncrementUsage:
    def test_increment(self, populated_store):
        before = populated_store.get("tag:machine-learning")
        initial = before["usage_count"]
        populated_store.increment_usage("tag:machine-learning")
        after = populated_store.get("tag:machine-learning")
        assert after["usage_count"] == initial + 1

    def test_increment_nonexistent_no_error(self, populated_store):
        populated_store.increment_usage("tag:nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Query Tests
# ---------------------------------------------------------------------------


class TestListByKind:
    def test_list_tags(self, populated_store):
        tags = populated_store.list_by_kind("tag")
        names = [t["name"] for t in tags]
        assert "machine-learning" in names
        assert "property-management" in names

    def test_list_folders(self, populated_store):
        folders = populated_store.list_by_kind("folder")
        assert len(folders) == 2

    def test_list_filtered_by_status(self, populated_store):
        populated_store.update("tag:property-management", status="archived")
        active_tags = populated_store.list_by_kind("tag", status="active")
        names = [t["name"] for t in active_tags]
        assert "property-management" not in names
        assert "machine-learning" in names

    def test_list_empty_kind(self, populated_store):
        assert populated_store.list_by_kind("nonexistent_kind") == []

    def test_list_from_empty_store(self, store):
        assert store.list_by_kind("tag") == []


class TestSearch:
    def test_vector_search_returns_results(self, populated_store):
        results = populated_store.search("artificial intelligence", top_k=5)
        assert len(results) > 0

    def test_vector_search_with_kind_filter(self, populated_store):
        results = populated_store.search("files", kind="folder", top_k=5)
        for r in results:
            assert r["kind"] == "folder"

    def test_search_empty_store(self, store):
        assert store.search("anything") == []


class TestResolveAliases:
    def test_alias_resolved(self, populated_store):
        result = populated_store.resolve_aliases(["ml"])
        assert result == ["machine-learning"]

    def test_canonical_name_passes_through(self, populated_store):
        result = populated_store.resolve_aliases(["machine-learning"])
        assert result == ["machine-learning"]

    def test_unknown_passes_through(self, populated_store):
        result = populated_store.resolve_aliases(["unknown-tag"])
        assert result == ["unknown-tag"]

    def test_mixed_aliases_and_canonical(self, populated_store):
        result = populated_store.resolve_aliases(["ml", "property-management", "unknown"])
        assert "machine-learning" in result
        assert "property-management" in result
        assert "unknown" in result

    def test_deduplication(self, populated_store):
        result = populated_store.resolve_aliases(["ml", "deep-learning", "machine-learning"])
        assert result == ["machine-learning"]

    def test_empty_list(self, populated_store):
        assert populated_store.resolve_aliases([]) == []

    def test_empty_store(self, store):
        assert store.resolve_aliases(["test"]) == ["test"]


# ---------------------------------------------------------------------------
# Format for prompt
# ---------------------------------------------------------------------------


class TestFormatForPrompt:
    def test_format_all_kinds(self, populated_store):
        text = populated_store.format_for_prompt()
        assert "## Available Tags" in text
        assert "machine-learning" in text
        assert "## Available Folders" in text
        assert "0-AI/" in text
        assert "## Available Document Types" in text

    def test_format_single_kind(self, populated_store):
        text = populated_store.format_for_prompt(kind="tag")
        assert "## Available Tags" in text
        assert "## Available Folders" not in text

    def test_format_empty_store(self, store):
        assert store.format_for_prompt() == ""


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_all(self, populated_store):
        assert populated_store.count() == 5

    def test_count_by_kind(self, populated_store):
        assert populated_store.count("tag") == 2
        assert populated_store.count("folder") == 2
        assert populated_store.count("doc_type") == 1

    def test_count_empty(self, store):
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Batch add
# ---------------------------------------------------------------------------


class TestBatchAdd:
    def test_add_batch(self, store):
        rows = []
        for i in range(5):
            rows.append({
                "id": f"tag:batch-{i}",
                "kind": "tag",
                "name": f"batch-{i}",
                "description": f"Batch tag {i}",
                "aliases": "",
                "parent": "",
                "status": "active",
                "usage_count": 0,
                "ai_managed": 1,
                "contents_type": "",
                "created_by": "test",
                "vector": _fake_embed(f"Batch tag {i}"),
            })
        count = store.add_batch(rows)
        assert count == 5
        assert store.count() == 5

    def test_add_batch_empty(self, store):
        assert store.add_batch([]) == 0


# ---------------------------------------------------------------------------
# Stale connection / table-already-exists regression
# ---------------------------------------------------------------------------

class TestStaleConnectionRecovery:
    """Verify that a new TaxonomyStore instance can write to a table
    that was created by a previous instance (simulates container restart,
    volume remount, or MCP server re-init with existing data on disk)."""

    def test_second_instance_can_add(self, tmp_path):
        """Creating a second store on the same dir should not crash on add."""
        store1 = TaxonomyStore(str(tmp_path), table_name="taxonomy", embed_fn=_fake_embed)
        store1.add("tag", "first-tag", "Created by store1")
        assert store1.count() == 1

        # Simulate restart: new instance, same directory
        store2 = TaxonomyStore(str(tmp_path), table_name="taxonomy", embed_fn=_fake_embed)
        store2.add("tag", "second-tag", "Created by store2")
        assert store2.count() == 2

    def test_stale_table_handle_recovers_on_add(self, tmp_path):
        """If _table is forced to None (simulating stale state), add should recover."""
        store = TaxonomyStore(str(tmp_path), table_name="taxonomy", embed_fn=_fake_embed)
        store.add("tag", "existing", "Already here")
        assert store.count() == 1

        # Simulate stale state: table exists on disk but handle is lost
        store._table = None
        # This would previously crash with "Table 'taxonomy' already exists"
        store.add("tag", "new-tag", "Added after stale reset")
        assert store.count() == 2
        assert store.get("tag:new-tag") is not None

    def test_stale_table_handle_recovers_on_read(self, tmp_path):
        """If _table is forced to None, read operations should still find data."""
        store = TaxonomyStore(str(tmp_path), table_name="taxonomy", embed_fn=_fake_embed)
        store.add("tag", "visible", "Should be found")

        store._table = None
        # Should recover and find the entry
        result = store.get("tag:visible")
        assert result is not None
        assert result["name"] == "visible"

    def test_stale_table_handle_recovers_on_list(self, tmp_path):
        """list_by_kind should recover from stale _table = None."""
        store = TaxonomyStore(str(tmp_path), table_name="taxonomy", embed_fn=_fake_embed)
        store.add("tag", "listed", "Should appear in list")

        store._table = None
        results = store.list_by_kind("tag")
        assert len(results) == 1

    def test_stale_table_handle_recovers_on_count(self, tmp_path):
        """count() should recover from stale _table = None."""
        store = TaxonomyStore(str(tmp_path), table_name="taxonomy", embed_fn=_fake_embed)
        store.add("folder", "test-folder", "A folder")

        store._table = None
        assert store.count() == 1
        assert store.count("folder") == 1
