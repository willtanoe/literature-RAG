"""Application configuration and defaults."""

from __future__ import annotations

import os
from pathlib import Path

PROJECTS_DIR = Path(os.getenv("LITERATURE_RAG_HOME", "./literature_projects"))
DOWNLOAD_DIR = Path(os.getenv("LITERATURE_RAG_DOWNLOADS", "./downloaded_papers"))
CONFIG_PATH = Path(os.getenv("LITERATURE_RAG_CONFIG", "./.literature_rag_config.json"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "5"))
DEFAULT_RETRIEVAL_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "12"))
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", "52428800"))  # 50 MB in bytes
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "500"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "20000"))
NETWORK_TIMEOUT = int(os.getenv("NETWORK_TIMEOUT", "30"))
NETWORK_ATTEMPTS = int(os.getenv("NETWORK_ATTEMPTS", "3"))
SEARCH_CACHE_VERSION = int(os.getenv("SEARCH_CACHE_VERSION", "3"))

# Deprecated: kept for backward compatibility, will be removed in next major release
if "LITERATURE_RAG_PROJECTS" in os.environ:
    PROJECTS_DIR = Path(os.environ["LITERATURE_RAG_PROJECTS"])
if "LITERATURE_RAG_INDEX_DIR" in os.environ:
    DOWNLOAD_DIR = Path(os.environ["LITERATURE_RAG_INDEX_DIR"])

LLM_REQUEST_TIMEOUT_SECONDS = 120
LLM_EXTRACTION_BATCH_PAPERS = 5
LLM_HEARTBEAT_SECONDS = 5
LLM_RETRY_DELAY_SECONDS = 15
LLM_MAX_RETRY_AFTER_SECONDS = 120
PAPER_READ_BUDGET_CHARS = 24_000
PAPER_READ_SLICE_CHARS = 12_000
PAPER_READ_MAX_SLICES = 2
PAPER_READ_PARSE_ATTEMPTS = 2
PAPER_READ_PREVIEW_CHARS = 200
DRAFT_CONTEXT_BUDGET_CHARS = 90_000
PAPER_TIERS = ("core", "related", "peripheral", "excluded")
DOWNLOAD_TIERS = ("core", "related", "peripheral")
GAP_TIERS = ("core", "related")
DEFAULT_TIER = "related"
TIER_DIRECTORIES = {"core": "core", "related": "related", "peripheral": "peripheral"}
TIER_READ_SLICES = {"core": 2, "related": 2, "peripheral": 1, "excluded": 0}
TIER_CONTEXT_CHUNKS = {"core": 2, "related": 1, "peripheral": 1, "excluded": 0}
TIER_SCORES = {"core": 3.0, "related": 2.0, "peripheral": 1.0, "excluded": 0.0}
MAX_TARGET_PAPERS = 80
MAX_PLAN_QUERIES = 8
MAX_PLAN_QUERY_TERMS = 4
CLASSIFY_ABSTRACT_CHARS = 400
CLASSIFY_BATCH_PAPERS = 20
ARXIV_METADATA_BATCH = 50
MAX_GAP_ROWS = 60
MAX_THESIS_CANDIDATES = 12
THESIS_PROPOSAL_TARGET = 4
ARCHIVE_DIR_NAME = "archive"
