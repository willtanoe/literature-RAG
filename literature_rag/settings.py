from pathlib import Path

PROJECTS_DIR = Path("./literature_projects")
DOWNLOAD_DIR = Path("./downloaded_papers")
CONFIG_PATH = Path("./.literature_rag_config.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_N = 5
DEFAULT_RETRIEVAL_K = 12
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_CHUNKS = 20_000
NETWORK_TIMEOUT = 30
NETWORK_ATTEMPTS = 3
SEARCH_CACHE_VERSION = 2
