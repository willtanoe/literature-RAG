"""Application configuration and defaults."""
from __future__ import annotations

import os
from pathlib import Path

PROJECTS_DIR = Path(
    os.getenv("LITERATURE_RAG_HOME", "./literature_projects")
)
DOWNLOAD_DIR = Path(os.getenv("LITERATURE_RAG_DOWNLOADS", "./downloaded_papers"))
CONFIG_PATH = Path(os.getenv("LITERATURE_RAG_CONFIG", "./.literature_rag_config.json"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "5"))
DEFAULT_RETRIEVAL_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "12"))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", "52428800"))  # 50 MB in bytes
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "500"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "20000"))
NETWORK_TIMEOUT = int(os.getenv("NETWORK_TIMEOUT", "30"))
NETWORK_ATTEMPTS = int(os.getenv("NETWORK_ATTEMPTS", "3"))
SEARCH_CACHE_VERSION = int(os.getenv("SEARCH_CACHE_VERSION", "2"))

# Deprecated: kept for backward compatibility, will be removed in next major release
if "LITERATURE_RAG_PROJECTS" in os.environ:
    PROJECTS_DIR = Path(os.environ["LITERATURE_RAG_PROJECTS"])
if "LITERATURE_RAG_INDEX_DIR" in os.environ:
    DOWNLOAD_DIR = Path(os.environ["LITERATURE_RAG_INDEX_DIR"])
