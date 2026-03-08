# Config Edge Cases Integration Tests - Design Doc: system-design-doc.md
# Generated: 2026-03-08 | Budget Used: 1/3 integration, 0/2 E2E
#
# Tests config.load_config() edge cases NOT covered by test_config.py:
# - Environment variable overrides (DOCUMENTS_ROOT, INDEX_ROOT)
# - Interaction between env vars and YAML values
#
# Existing test_config.py covers: basic loading, missing keys, validation,
# chunking params, search top_k. This file covers ONLY env var overrides.
#
# Framework: pytest
# Run with: pytest tests/test_config_edge_cases.int.test.py -v

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from core.config import load_config


# ===================================================================
# TEST 1: Environment Variable Overrides
# ===================================================================

# AC-CONFIG-1: "Environment variable overrides: DOCUMENTS_ROOT, INDEX_ROOT"
# ROI: 38 | Business Value: 7 (deployment-critical for VPS/containers) | Frequency: 5
# Behavior: When DOCUMENTS_ROOT env var is set, it overrides YAML documents_root
#           When INDEX_ROOT env var is set, it overrides YAML index_root
#           Both vault_root and documents_root aliases update from env var
# @category: integration
# @dependency: core.config.load_config, os.environ
# @complexity: medium
#
# Verification items:
# - DOCUMENTS_ROOT env var overrides YAML documents_root value
# - INDEX_ROOT env var overrides YAML index_root value
# - vault_root alias also updated when DOCUMENTS_ROOT env var is set
# - Validation still runs against env-var-overridden path (nonexistent dir -> error)

def test_documents_root_env_var_overrides_yaml():
    """AC-CONFIG-1: DOCUMENTS_ROOT env var overrides YAML documents_root."""
    with tempfile.TemporaryDirectory() as yaml_docs_dir, \
         tempfile.TemporaryDirectory() as env_docs_dir, \
         tempfile.TemporaryDirectory() as tmpdir:
        # Arrange: write config.yaml with documents_root=yaml_docs_dir, set DOCUMENTS_ROOT=env_docs_dir
        config_path = Path(tmpdir) / "config.yaml"
        config_data = {
            "documents_root": yaml_docs_dir,
            "index_root": tmpdir,
        }
        config_path.write_text(yaml.dump(config_data))

        old_env = os.environ.get("DOCUMENTS_ROOT")
        os.environ["DOCUMENTS_ROOT"] = env_docs_dir
        try:
            # Act
            config = load_config(str(config_path))

            # Assert: env var overrides YAML value
            assert config["documents_root"] == env_docs_dir
            # vault_root alias also updated
            assert config["vault_root"] == env_docs_dir
        finally:
            if old_env is None:
                os.environ.pop("DOCUMENTS_ROOT", None)
            else:
                os.environ["DOCUMENTS_ROOT"] = old_env


def test_index_root_env_var_overrides_yaml():
    """AC-CONFIG-1: INDEX_ROOT env var overrides YAML index_root."""
    with tempfile.TemporaryDirectory() as docs_dir, \
         tempfile.TemporaryDirectory() as tmpdir:
        # Arrange: write config.yaml with index_root=A, set INDEX_ROOT=B
        env_index_root = str(Path(tmpdir) / "env_index")
        config_path = Path(tmpdir) / "config.yaml"
        config_data = {
            "documents_root": docs_dir,
            "index_root": str(Path(tmpdir) / "yaml_index"),
        }
        config_path.write_text(yaml.dump(config_data))

        old_env = os.environ.get("INDEX_ROOT")
        os.environ["INDEX_ROOT"] = env_index_root
        try:
            # Act
            config = load_config(str(config_path))

            # Assert: env var overrides YAML value
            assert config["index_root"] == env_index_root
        finally:
            if old_env is None:
                os.environ.pop("INDEX_ROOT", None)
            else:
                os.environ["INDEX_ROOT"] = old_env


def test_env_var_override_still_validates_path():
    """AC-CONFIG-1: DOCUMENTS_ROOT pointing to nonexistent dir still raises ValueError."""
    with tempfile.TemporaryDirectory() as docs_dir, \
         tempfile.TemporaryDirectory() as tmpdir:
        # Arrange: write valid config.yaml, set DOCUMENTS_ROOT=/nonexistent/path
        config_path = Path(tmpdir) / "config.yaml"
        config_data = {
            "documents_root": docs_dir,
            "index_root": tmpdir,
        }
        config_path.write_text(yaml.dump(config_data))

        old_env = os.environ.get("DOCUMENTS_ROOT")
        os.environ["DOCUMENTS_ROOT"] = "/nonexistent/path/that/does/not/exist"
        try:
            # Act & Assert: raises ValueError matching "documents_root does not exist"
            with pytest.raises(ValueError, match="documents_root does not exist"):
                load_config(str(config_path))
        finally:
            if old_env is None:
                os.environ.pop("DOCUMENTS_ROOT", None)
            else:
                os.environ["DOCUMENTS_ROOT"] = old_env
