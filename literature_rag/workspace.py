from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from literature_rag.settings import PROJECTS_DIR


@dataclass(frozen=True)
class ProjectWorkspace:
    root: Path
    papers: Path
    index: Path
    search_cache: Path
    metadata_cache: Path
    output: Path


def get_workspace(topic: str, projects_dir: Path = PROJECTS_DIR) -> ProjectWorkspace:
    normalized = " ".join(topic.lower().split())
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:48] or "review"
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    root = projects_dir / f"{slug}-{suffix}"
    papers = root / "papers"
    index = root / "faiss_index"
    papers.mkdir(parents=True, exist_ok=True)
    return ProjectWorkspace(
        root,
        papers,
        index,
        root / "search_results.json",
        root / "metadata.json",
        root / "output",
    )
