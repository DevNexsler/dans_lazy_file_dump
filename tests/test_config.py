"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest

from core.config import load_config


def _write_config(tmpdir: Path, content: str) -> Path:
    p = tmpdir / "config.yaml"
    p.write_text(content)
    return p


def test_load_valid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir) / "vault"
        vault.mkdir()
        cfg_path = _write_config(Path(tmpdir), f"""
vault_root: "{vault}"
index_root: "{tmpdir}/index"
""")
        config = load_config(cfg_path)
        assert config["vault_root"] == str(vault)
        assert config["documents_root"] == str(vault)


def test_load_documents_root_config():
    """documents_root is accepted as an alias for vault_root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = Path(tmpdir) / "documents"
        docs.mkdir()
        cfg_path = _write_config(Path(tmpdir), f"""
documents_root: "{docs}"
index_root: "{tmpdir}/index"
""")
        config = load_config(cfg_path)
        assert config["documents_root"] == str(docs)
        assert config["vault_root"] == str(docs)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_missing_documents_root_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(Path(tmpdir), 'index_root: "/tmp/idx"')
        with pytest.raises(ValueError, match="documents_root"):
            load_config(cfg_path)


def test_missing_index_root_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir) / "vault"
        vault.mkdir()
        cfg_path = _write_config(Path(tmpdir), f'vault_root: "{vault}"')
        with pytest.raises(ValueError, match="index_root"):
            load_config(cfg_path)


def test_documents_root_does_not_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(Path(tmpdir), """
vault_root: "/nonexistent/vault"
index_root: "/tmp/idx"
""")
        with pytest.raises(ValueError, match="documents_root does not exist"):
            load_config(cfg_path)


def test_empty_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(Path(tmpdir), "")
        with pytest.raises(ValueError, match="empty"):
            load_config(cfg_path)


# --- Chunking validation (Fix 6) ---


def _base_config(tmpdir: Path) -> str:
    """Return YAML string with valid vault_root and index_root."""
    vault = tmpdir / "vault"
    vault.mkdir(exist_ok=True)
    return f'vault_root: "{vault}"\nindex_root: "{tmpdir}/index"'


def test_chunking_max_chars_must_be_positive():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(Path(tmpdir), _base_config(Path(tmpdir)) + "\nchunking:\n  max_chars: 0")
        with pytest.raises(ValueError, match="max_chars"):
            load_config(cfg_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(Path(tmpdir), _base_config(Path(tmpdir)) + "\nchunking:\n  max_chars: -5")
        with pytest.raises(ValueError, match="max_chars"):
            load_config(cfg_path)


def test_chunking_overlap_negative():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(Path(tmpdir), _base_config(Path(tmpdir)) + "\nchunking:\n  overlap: -1")
        with pytest.raises(ValueError, match="overlap"):
            load_config(cfg_path)


def test_chunking_overlap_exceeds_max_chars():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(
            Path(tmpdir),
            _base_config(Path(tmpdir)) + "\nchunking:\n  max_chars: 100\n  overlap: 100",
        )
        with pytest.raises(ValueError, match="overlap.*must be less than max_chars"):
            load_config(cfg_path)


def test_search_top_k_must_be_positive():
    for key in ("vector_top_k", "keyword_top_k", "final_top_k"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = _write_config(
                Path(tmpdir),
                _base_config(Path(tmpdir)) + f"\nsearch:\n  {key}: 0",
            )
            with pytest.raises(ValueError, match=key):
                load_config(cfg_path)


def test_valid_config_with_optional_fields():
    """A config with valid chunking and search values should load without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(
            Path(tmpdir),
            _base_config(Path(tmpdir)) + """
chunking:
  max_chars: 2000
  overlap: 200
search:
  vector_top_k: 50
  keyword_top_k: 50
  final_top_k: 10
""",
        )
        config = load_config(cfg_path)
        assert config["chunking"]["max_chars"] == 2000
        assert config["search"]["final_top_k"] == 10
