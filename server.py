"""Unified VPS entrypoint — starts MCP HTTP server on $PORT (default 7788)."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from core.config import load_config

config = load_config()

log_level = config.get("logging", {}).get("level", "WARNING").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))

from mcp_server import run_server

host = config.get("mcp", {}).get("host", "0.0.0.0")
port = int(os.environ.get("PORT", config.get("mcp", {}).get("port", 7788)))

run_server(transport="streamable-http", host=host, port=port)
