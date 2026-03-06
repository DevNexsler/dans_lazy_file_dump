"""Load and validate config from config.yaml. Fail fast on missing/invalid keys."""

from pathlib import Path
from typing import Any

import yaml

# Load .env file if present (for GEMINI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set manually


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML config. Raises if file missing or required keys absent.

    Accepts either ``documents_root`` or ``vault_root`` for the source
    directory.  Internally both are normalised so that either key works
    everywhere (``config["documents_root"]`` and ``config["vault_root"]``
    both resolve to the same path).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("Config file is empty")

    # Accept either documents_root or vault_root (prefer documents_root)
    docs_root = raw.get("documents_root") or raw.get("vault_root")
    if not docs_root:
        raise ValueError(
            "Config missing required key: documents_root (or vault_root)"
        )
    # Normalise: both keys resolve to the same value
    raw["documents_root"] = docs_root
    raw["vault_root"] = docs_root

    if "index_root" not in raw:
        raise ValueError("Config missing required key: index_root")

    docs_path = Path(docs_root)
    if not docs_path.exists():
        raise ValueError(f"documents_root does not exist: {docs_path}")

    # --- Validate chunking parameters ---
    chunk_cfg = raw.get("chunking", {})
    max_chars = chunk_cfg.get("max_chars")
    if max_chars is not None:
        if not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError(f"chunking.max_chars must be a positive integer, got {max_chars!r}")
    overlap = chunk_cfg.get("overlap")
    if overlap is not None:
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError(f"chunking.overlap must be a non-negative integer, got {overlap!r}")
        effective_max = max_chars if max_chars is not None else 1800
        if overlap >= effective_max:
            raise ValueError(
                f"chunking.overlap ({overlap}) must be less than max_chars ({effective_max})"
            )

    # --- Validate search parameters ---
    search_cfg = raw.get("search", {})
    for key in ("vector_top_k", "keyword_top_k", "final_top_k"):
        val = search_cfg.get(key)
        if val is not None:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"search.{key} must be a positive integer, got {val!r}")

    return raw
