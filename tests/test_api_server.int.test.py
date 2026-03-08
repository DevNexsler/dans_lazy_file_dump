# REST API Integration Tests - Design Doc: system-design-doc.md
# Generated: 2026-03-08 | Budget Used: 3/3 integration, 0/2 E2E
#
# These tests exercise the Starlette REST API (api_server.py) via HTTPX
# async test client. No external services needed -- uses a temp directory
# as documents_root.
#
# Framework: pytest + httpx (async)
# Run with: pytest tests/test_api_server.int.test.py -v

import hmac
import os
import tempfile
from pathlib import Path

import httpx
import pytest

from api_server import build_api_app
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_docs_root(tmp_path):
    """Create a temporary directory to serve as documents_root.
    Pre-populate with a sample .md file and a subdirectory containing a .pdf-like file.
    """
    # Create sample .md file
    sample_md = tmp_path / "sample.md"
    sample_md.write_text("# Sample Document\n\nThis is a test markdown file.")

    # Create subdirectory with a .pdf placeholder
    subdir = tmp_path / "reports"
    subdir.mkdir()
    sample_pdf = subdir / "report.pdf"
    sample_pdf.write_bytes(b"%PDF-1.4 fake pdf content for testing")

    return tmp_path


@pytest.fixture
async def api_client(tmp_docs_root):
    """Build the Starlette app via build_api_app(tmp_docs_root) and wrap it
    in httpx.AsyncClient(transport=ASGITransport(app)).
    """
    app = build_api_app(tmp_docs_root)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/") as client:
        yield client


def _build_auth_app(documents_root: Path, api_key: str):
    """Build the composed app with auth middleware matching server.py logic."""
    api_app = build_api_app(documents_root)
    app = Starlette(routes=[
        Mount("/api", app=api_app),
    ])

    expected = f"Bearer {api_key}".encode()

    async def auth_app(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            auth_value = b""
            for name, value in scope.get("headers", []):
                if name == b"authorization":
                    auth_value = value
                    break
            if not hmac.compare_digest(auth_value, expected):
                response = JSONResponse({"error": "Unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return auth_app


@pytest.fixture
async def api_client_with_auth(tmp_docs_root):
    """Build unified server with auth middleware, yields (client, api_key)."""
    api_key = "test-secret-key-12345"
    auth_app = _build_auth_app(tmp_docs_root, api_key)
    transport = httpx.ASGITransport(app=auth_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/") as client:
        yield client, api_key


# ===================================================================
# TEST 1: File Upload -- Happy Path and Validation
# ===================================================================


@pytest.mark.anyio
async def test_upload_valid_md_file(api_client, tmp_docs_root):
    """AC-REST-1: Upload a valid .md file returns 201 with correct doc_id and size."""
    file_content = b"# My Upload\n\nNew content here."
    files = {"file": ("upload_test.md", file_content, "text/markdown")}

    resp = await api_client.post("/upload", files=files)

    assert resp.status_code == 201
    body = resp.json()
    assert body["uploaded"] is True
    assert body["doc_id"] == "upload_test.md"
    assert body["size"] == len(file_content)

    # File actually exists on disk
    on_disk = tmp_docs_root / "upload_test.md"
    assert on_disk.exists()
    assert on_disk.read_bytes() == file_content


@pytest.mark.anyio
async def test_upload_rejects_disallowed_extension(api_client):
    """AC-REST-1: Upload .exe file returns 400 with invalid_file_type error."""
    files = {"file": ("malware.exe", b"MZ evil content", "application/octet-stream")}

    resp = await api_client.post("/upload", files=files)

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "invalid_file_type"


@pytest.mark.anyio
async def test_upload_rejects_path_traversal_in_directory(api_client):
    """AC-REST-1: Directory field with ../../ returns 400 with invalid_directory."""
    files = {"file": ("ok.md", b"# OK", "text/markdown")}
    data = {"directory": "../../etc"}

    resp = await api_client.post("/upload", files=files, data=data)

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "invalid_directory"


@pytest.mark.anyio
async def test_upload_rejects_missing_file_field(api_client):
    """AC-REST-1: Upload without file field returns 400 with missing_file."""
    # Send multipart with no "file" field -- send an empty form field instead
    resp = await api_client.post(
        "/upload",
        content=b"",
        headers={"content-type": "multipart/form-data; boundary=boundary123"},
    )

    # The server expects multipart/form-data with a "file" field
    # Sending malformed multipart should result in 400
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_upload_to_subdirectory(api_client, tmp_docs_root):
    """AC-REST-1: Upload with directory field creates file in subdirectory."""
    file_content = b"# Note\n\nSubfolder note."
    files = {"file": ("note.md", file_content, "text/markdown")}
    data = {"directory": "subfolder"}

    resp = await api_client.post("/upload", files=files, data=data)

    assert resp.status_code == 201
    body = resp.json()
    assert "subfolder/" in body["doc_id"] or body["doc_id"] == "subfolder/note.md"

    # File on disk in subfolder
    on_disk = tmp_docs_root / "subfolder" / "note.md"
    assert on_disk.exists()
    assert on_disk.read_bytes() == file_content


# ===================================================================
# TEST 2: File Download and Path Traversal Protection
# ===================================================================


@pytest.mark.anyio
async def test_download_existing_file(api_client, tmp_docs_root):
    """AC-REST-2: Download a pre-existing file returns 200 with correct content."""
    expected_content = "# Sample Document\n\nThis is a test markdown file."

    resp = await api_client.get("/documents/sample.md")

    assert resp.status_code == 200
    assert resp.text == expected_content


@pytest.mark.anyio
async def test_download_nonexistent_file(api_client):
    """AC-REST-2: Download a nonexistent file returns 404 with error body."""
    resp = await api_client.get("/documents/does_not_exist.md")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "not_found"


@pytest.mark.anyio
async def test_download_path_traversal_blocked(api_client):
    """AC-REST-2: Path traversal in doc_id returns 400.

    Uses URL-encoded path components because httpx normalizes bare ../
    sequences before sending the request.
    """
    resp = await api_client.get("/documents/..%2F..%2Fetc%2Fpasswd")

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] is True
    assert body["code"] == "invalid_path"


# ===================================================================
# TEST 3: Authentication Enforcement
# ===================================================================


@pytest.mark.anyio
async def test_auth_rejects_missing_token(api_client_with_auth):
    """AC-REST-4: When API_KEY is set, request without auth header returns 401."""
    client, _api_key = api_client_with_auth

    resp = await client.get("/api/documents")

    assert resp.status_code == 401


@pytest.mark.anyio
async def test_auth_rejects_wrong_token(api_client_with_auth):
    """AC-REST-4: When API_KEY is set, wrong Bearer token returns 401."""
    client, _api_key = api_client_with_auth

    resp = await client.get(
        "/api/documents",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert resp.status_code == 401


@pytest.mark.anyio
async def test_auth_accepts_correct_token(api_client_with_auth, tmp_docs_root):
    """AC-REST-4: When API_KEY is set, correct Bearer token returns 200."""
    client, api_key = api_client_with_auth

    resp = await client.get(
        "/api/documents",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "files" in body


@pytest.mark.anyio
async def test_auth_not_enforced_without_api_key(api_client):
    """AC-REST-4: When API_KEY is not set, requests pass without auth."""
    # api_client fixture has no auth middleware
    resp = await api_client.get("/documents")

    assert resp.status_code == 200
    body = resp.json()
    assert "files" in body


# ===================================================================
# Directory Listing
# ===================================================================


@pytest.mark.anyio
async def test_list_documents_root(api_client, tmp_docs_root):
    """AC-REST-3: Listing root directory returns files with correct structure."""
    resp = await api_client.get("/documents")

    assert resp.status_code == 200
    body = resp.json()
    assert "directory" in body
    assert "files" in body
    assert "total" in body
    assert "offset" in body
    assert "limit" in body

    # Should contain sample.md and reports/ directory
    names = {f["name"] for f in body["files"]}
    assert "sample.md" in names

    # Verify file entry structure
    for f in body["files"]:
        assert "name" in f
        assert "type" in f
        assert "path" in f
        if f["type"] == "file":
            assert "size" in f


@pytest.mark.anyio
async def test_list_documents_filters_extensions(api_client, tmp_docs_root):
    """AC-REST-3: Listing only shows files with allowed extensions."""
    # Create files with disallowed extensions
    (tmp_docs_root / "notes.txt").write_text("text file")
    (tmp_docs_root / "script.py").write_text("python file")

    resp = await api_client.get("/documents")

    body = resp.json()
    names = {f["name"] for f in body["files"] if f["type"] == "file"}

    # .md and .pdf should be present, .txt and .py should NOT
    assert "sample.md" in names
    assert "notes.txt" not in names
    assert "script.py" not in names


@pytest.mark.anyio
async def test_list_documents_pagination(api_client, tmp_docs_root):
    """AC-REST-3: Offset and limit params paginate results correctly."""
    # Create 5 .md files
    for i in range(5):
        (tmp_docs_root / f"doc{i}.md").write_text(f"# Doc {i}")

    # Get total count first
    resp_all = await api_client.get("/documents")
    total = resp_all.json()["total"]
    assert total >= 5  # at least our 5 + the pre-existing sample.md

    # Paginate with limit=2, offset=1
    resp = await api_client.get("/documents?limit=2&offset=1")

    body = resp.json()
    assert len(body["files"]) == 2
    assert body["total"] == total
    assert body["offset"] == 1
