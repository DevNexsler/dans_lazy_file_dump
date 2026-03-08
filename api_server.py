"""REST API for document upload and download.

Separate from MCP server — mounted alongside it in server.py.
Requires the same API_KEY auth when set.
"""

import logging
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Max upload size: 100 MB
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

_ALLOWED_EXTENSIONS = {".md", ".pdf", ".png", ".jpg", ".jpeg"}

_DEFAULT_LIST_LIMIT = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_error(code: str, message: str, status_code: int = 400) -> JSONResponse:
    """Build a structured JSON error response."""
    return JSONResponse({"error": True, "code": code, "message": message}, status_code=status_code)


def _safe_subpath(base: Path, user_input: str) -> Path | None:
    """Resolve a user-provided relative path safely under base.

    Returns the resolved Path if safe, or None if it escapes base.
    """
    candidate = (base / user_input).resolve()
    try:
        candidate.relative_to(base.resolve())
        return candidate
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def upload(request: Request) -> JSONResponse:
    """Upload a file to documents_root.

    Multipart form data:
        file: The file to upload (required).
        directory: Subdirectory within documents_root (optional, default: root).
    """
    docs_root: Path = request.app.state.documents_root

    # Early size rejection via Content-Length header
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_UPLOAD_BYTES:
        return _api_error("file_too_large", f"File exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit", 413)

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return _api_error("invalid_request", "Expected multipart/form-data")

    form = await request.form()
    file = form.get("file")
    if file is None:
        return _api_error("missing_file", "No file field in upload")

    filename = file.filename
    if not filename:
        return _api_error("missing_filename", "File has no filename")

    # Validate extension
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return _api_error("invalid_file_type", f"File type '{ext}' not allowed. Allowed: {sorted(_ALLOWED_EXTENSIONS)}")

    # Sanitize filename — prevent path traversal
    safe_name = Path(filename).name
    if not safe_name or safe_name.startswith("."):
        return _api_error("invalid_filename", "Invalid filename")

    # Resolve target directory safely
    directory = form.get("directory", "")
    if isinstance(directory, str) and directory.strip():
        target_dir = _safe_subpath(docs_root, directory.strip().strip("/"))
        if target_dir is None:
            return _api_error("invalid_directory", "Directory escapes documents root")
    else:
        target_dir = docs_root

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name

    # Read file with size enforcement
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        return _api_error("file_too_large", f"File exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit", 413)

    target_path.write_bytes(content)

    doc_id = str(target_path.relative_to(docs_root)).replace("\\", "/")
    logger.info("Uploaded: %s (%d bytes)", doc_id, len(content))

    return JSONResponse({"uploaded": True, "doc_id": doc_id, "size": len(content)}, status_code=201)


async def download(request: Request) -> FileResponse | JSONResponse:
    """Download a file by doc_id (path relative to documents_root).

    GET /api/documents/{doc_id:path}
    """
    docs_root: Path = request.app.state.documents_root

    doc_id = request.path_params.get("doc_id", "")
    if not doc_id:
        return _api_error("missing_doc_id", "doc_id path parameter required")

    file_path = _safe_subpath(docs_root, doc_id)
    if file_path is None:
        return _api_error("invalid_path", "Path escapes documents root")

    if not file_path.is_file():
        return _api_error("not_found", f"File not found: {doc_id}", 404)

    return FileResponse(file_path, filename=file_path.name)


async def list_documents(request: Request) -> JSONResponse:
    """List files in a directory within documents_root.

    GET /api/documents/?directory=optional/subdir&limit=200&offset=0
    """
    docs_root: Path = request.app.state.documents_root
    directory = request.query_params.get("directory", "")
    limit = min(int(request.query_params.get("limit", _DEFAULT_LIST_LIMIT)), _DEFAULT_LIST_LIMIT)
    offset = int(request.query_params.get("offset", 0))

    if directory:
        target_dir = _safe_subpath(docs_root, directory.strip().strip("/"))
        if target_dir is None:
            return _api_error("invalid_directory", "Directory escapes documents root")
    else:
        target_dir = docs_root

    if not target_dir.is_dir():
        return _api_error("not_found", f"Directory not found: {directory}", 404)

    files = []
    for entry in sorted(target_dir.iterdir(), key=lambda p: p.name):
        rel = str(entry.relative_to(docs_root)).replace("\\", "/")
        if entry.is_dir():
            files.append({"name": entry.name, "type": "directory", "path": rel})
        elif entry.suffix.lower() in _ALLOWED_EXTENSIONS:
            files.append({
                "name": entry.name,
                "type": "file",
                "path": rel,
                "size": entry.stat().st_size,
            })

    total = len(files)
    files = files[offset:offset + limit]

    return JSONResponse({
        "directory": directory or ".",
        "files": files,
        "total": total,
        "offset": offset,
        "limit": limit,
    })


def build_api_app(documents_root: Path) -> Starlette:
    """Build the REST API Starlette app.

    Args:
        documents_root: Path to the documents directory (injected, not re-loaded per request).
    """
    routes = [
        Route("/upload", upload, methods=["POST"]),
        Route("/documents/{doc_id:path}", download, methods=["GET"]),
        Route("/documents/", list_documents, methods=["GET"]),
        Route("/documents", list_documents, methods=["GET"]),
    ]
    app = Starlette(routes=routes)
    app.state.documents_root = Path(documents_root)
    return app
