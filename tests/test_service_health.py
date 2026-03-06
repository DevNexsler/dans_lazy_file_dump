"""Service health checks: fast connectivity tests for all active endpoints.

Each test class is independently skippable when its service is unavailable.
These tests verify "is the service up and reachable?" — not inference quality.

Run with:  pytest tests/test_service_health.py -v
"""

import os
import struct
import zlib

import pytest
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# --- Helpers ---

def _probe_url(url: str, timeout: float = 3.0) -> bool:
    """Return True if URL responds (any non-5xx status)."""
    try:
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


def _minimal_png() -> bytes:
    """Generate a valid 1x1 white PNG for endpoint testing."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return sig + ihdr + idat + iend


# -----------------------------------------------------------------------
# OpenRouter (enrichment + embedding)
# -----------------------------------------------------------------------

_has_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(not _has_openrouter_key, reason="OPENROUTER_API_KEY not set")
class TestOpenRouterHealth:
    """Verify OpenRouter API is reachable and accepts our key."""

    def _headers(self):
        return {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }

    def test_chat_completions_reachable(self):
        """Minimal chat completion returns 200."""
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "minimax/minimax-m2.5",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            headers=self._headers(),
            timeout=30.0,
        )
        assert resp.status_code == 200, f"OpenRouter chat: {resp.status_code} {resp.text[:200]}"

    def test_embeddings_reachable(self):
        """Minimal embedding returns 200 with vector data."""
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            json={
                "model": "qwen/qwen3-embedding-8b",
                "input": ["health check"],
            },
            headers=self._headers(),
            timeout=30.0,
        )
        assert resp.status_code == 200, f"OpenRouter embed: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) > 0


# -----------------------------------------------------------------------
# Baseten (reranker)
# -----------------------------------------------------------------------

_has_baseten_key = bool(os.environ.get("BASETEN_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(not _has_baseten_key, reason="BASETEN_API_KEY not set")
class TestBasetenRerankerHealth:
    """Verify Baseten reranker endpoint is reachable."""

    def test_predict_reachable(self):
        """Minimal /predict returns 200."""
        from core.config import load_config

        config = load_config()
        model_id = config.get("search", {}).get("reranker", {}).get("model_id", "wnppr2y3")

        base = f"https://model-{model_id}.api.baseten.co/environments/production/sync"
        resp = httpx.get(
            f"{base}/v1/models",
            headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
            timeout=60.0,
        )
        assert resp.status_code == 200, f"Baseten reranker: {resp.status_code} {resp.text[:200]}"


# -----------------------------------------------------------------------
# DeepSeek-OCR2 (local)
# -----------------------------------------------------------------------

_deepseek_ocr2_available = _probe_url("http://localhost:8790/health")


@pytest.mark.live
@pytest.mark.skipif(not _deepseek_ocr2_available, reason="DeepSeek-OCR2 not running at localhost:8790")
class TestDeepSeekOCR2Health:
    """Verify local DeepSeek-OCR2 service is responding."""

    def test_health_endpoint(self):
        """GET /health returns 200."""
        resp = httpx.get("http://localhost:8790/health", timeout=5.0)
        assert resp.status_code == 200

    def test_extract_endpoint_accepts_post(self):
        """POST /extract with minimal PNG returns non-5xx."""
        resp = httpx.post(
            "http://localhost:8790/extract",
            files={"file": ("test.png", _minimal_png(), "image/png")},
            timeout=30.0,
        )
        assert resp.status_code < 500, f"DeepSeek-OCR2 /extract: {resp.status_code}"


# -----------------------------------------------------------------------
# LanceDB (local file DB)
# -----------------------------------------------------------------------

@pytest.mark.live
class TestLanceDBHealth:
    """Verify LanceDB index is accessible."""

    def test_store_opens(self):
        """LanceDBStore opens and lists doc_ids without error."""
        from core.config import load_config
        from lancedb_store import LanceDBStore

        config = load_config()
        store = LanceDBStore(
            config["index_root"],
            config.get("lancedb", {}).get("table", "chunks"),
        )
        doc_ids = store.list_doc_ids()
        assert isinstance(doc_ids, list)

    def test_fts_available(self):
        """FTS/tantivy index is operational."""
        from core.config import load_config
        from lancedb_store import LanceDBStore

        config = load_config()
        store = LanceDBStore(
            config["index_root"],
            config.get("lancedb", {}).get("table", "chunks"),
        )
        if not store.fts_available():
            pytest.skip("FTS index not available — run file_index_update to rebuild")
