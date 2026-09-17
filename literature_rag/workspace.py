from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from literature_rag.settings import (
    ARCHIVE_DIR_NAME,
    DOWNLOAD_TIERS,
    PROJECTS_DIR,
    TIER_DIRECTORIES,
)

ARCHIVED_NAMES = (
    "report.md",
    "draft.md",
    "critic_attempt.md",
    "review.json",
    "references.bib",
    "thesis_directions.md",
    "thesis_proposals.md",
    "search_plan.json",
    "papers_manifest.json",
    "failure.txt",
)


@dataclass(frozen=True)
class ProjectWorkspace:
    root: Path
    papers: Path
    index: Path
    search_cache: Path
    metadata_cache: Path
    output: Path
    classification_cache: Path
    plan_cache: Path

    def tier_dir(self, tier: str) -> Path:
        return self.papers / TIER_DIRECTORIES.get(tier, tier)

    @property
    def archive(self) -> Path:
        return self.output / ARCHIVE_DIR_NAME


def get_workspace(topic: str, projects_dir: Path = PROJECTS_DIR) -> ProjectWorkspace:
    normalized = " ".join(topic.lower().split())
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:48] or "review"
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    root = projects_dir / f"{slug}-{suffix}"
    papers = root / "papers"
    index = root / "faiss_index"
    papers.mkdir(parents=True, exist_ok=True)
    for tier in DOWNLOAD_TIERS:
        (papers / TIER_DIRECTORIES[tier]).mkdir(parents=True, exist_ok=True)
    return ProjectWorkspace(
        root,
        papers,
        index,
        root / "search_results.json",
        root / "metadata.json",
        root / "output",
        root / "classification.json",
        root / "search_plan.json",
    )


def archive_previous_output(output_dir: Path) -> Path | None:
    existing = [output_dir / name for name in ARCHIVED_NAMES if (output_dir / name).is_file()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir / ARCHIVE_DIR_NAME / stamp
    destination.mkdir(parents=True, exist_ok=True)
    for source in existing:
        shutil.move(str(source), str(destination / source.name))
    print(f"Archive -> moved {len(existing)} previous output file(s) to {destination}")
    return destination
