from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from literature_rag.__log__ import get_logger
from literature_rag.config import LLMConfig
from literature_rag.papers import DownloadedPaper, Paper
from literature_rag.resilience import atomic_write_json, atomic_write_text
from literature_rag.settings import DEFAULT_TIER, PAPER_TIERS

logger = get_logger(__name__)


def export_review(
    output_dir: Path,
    report: str,
    topic: str,
    analysis_question: str,
    downloaded: list[DownloadedPaper],
    llm_config: LLMConfig,
    sota_results: list[dict] | None = None,
    paper_summaries: list[dict] | None = None,
    thesis_proposals: str | None = None,
    gap_register: list[dict] | None = None,
    thesis_candidates: list[dict] | None = None,
    search_plan: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "review.json"
    bibliography_path = output_dir / "references.bib"
    proposals_path = output_dir / "thesis_proposals.md"
    papers_manifest_path = output_dir / "papers_manifest.json"
    plan_path = output_dir / "search_plan.json"
    papers = [paper for _, paper in downloaded]
    tier_counts = _tier_counts(papers)
    print("Export -> writing 1/6: report.md")
    atomic_write_text(report_path, report.rstrip() + "\n")
    manifest = {
        "topic": topic,
        "analysis_question": analysis_question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": llm_config.model_name,
        "base_url": llm_config.base_url,
        "report_file": report_path.name,
        "bibliography_file": bibliography_path.name,
        "papers_manifest_file": papers_manifest_path.name,
        "tier_counts": tier_counts,
        "papers": [_paper_json(path, paper) for path, paper in downloaded],
        "sota_results": sota_results or [],
        "paper_summaries": paper_summaries or [],
        "gap_register": gap_register or [],
        "thesis_candidates": thesis_candidates or [],
    }
    print("Export -> writing 2/6: review.json")
    atomic_write_json(manifest_path, manifest)
    print("Export -> writing 3/6: references.bib")
    atomic_write_text(bibliography_path, "\n\n".join(_bibtex(paper) for paper in papers) + "\n")
    logger.info(f"Markdown: {report_path}")
    logger.info(f"JSON: {manifest_path}")
    logger.info(f"BibTeX: {bibliography_path}")
    print("Export -> writing 4/6: papers_manifest.json")
    atomic_write_json(
        papers_manifest_path,
        {
            "topic": topic,
            "generated_at": manifest["generated_at"],
            "tier_counts": tier_counts,
            "papers": [_paper_json(path, paper) for path, paper in downloaded],
        },
    )
    exports = {
        "markdown": report_path,
        "json": manifest_path,
        "bibtex": bibliography_path,
        "papers_manifest": papers_manifest_path,
    }
    if search_plan:
        print("Export -> writing 5/6: search_plan.json")
        atomic_write_json(plan_path, search_plan)
        manifest["search_plan_file"] = plan_path.name
        exports["search_plan"] = plan_path
    if thesis_proposals and thesis_proposals.strip():
        print("Export -> writing 6/6: thesis_proposals.md")
        atomic_write_text(proposals_path, thesis_proposals.rstrip() + "\n")
        manifest["thesis_proposals_file"] = proposals_path.name
        exports["thesis_proposals"] = proposals_path
    if "search_plan_file" in manifest or "thesis_proposals_file" in manifest:
        atomic_write_json(manifest_path, manifest)
    for label, path in exports.items():
        print(f"Export -> {label}: {path}")
    return exports


def _tier_counts(papers: list[Paper]) -> dict[str, int]:
    counts = {tier: 0 for tier in PAPER_TIERS}
    for paper in papers:
        tier = paper.tier if paper.tier in PAPER_TIERS else DEFAULT_TIER
        counts[tier] += 1
    return {tier: count for tier, count in counts.items() if count}


def _paper_json(path: Path, paper: Paper) -> dict:
    return {
        "title": paper.title,
        "authors": list(paper.authors),
        "arxiv_id": paper.arxiv_id,
        "doi": paper.doi,
        "venue": paper.venue,
        "year": paper.year,
        "citation_count": paper.citation_count,
        "is_open_access": paper.is_open_access,
        "tier": paper.tier or DEFAULT_TIER,
        "url": paper.entry_id,
        "pdf": path.name,
        "pdf_folder": path.parent.name,
    }


def _bibtex(paper: Paper) -> str:
    key = _citation_key(paper)
    fields = [
        ("title", paper.title),
        ("author", " and ".join(paper.authors)),
        ("year", str(paper.year) if paper.year else ""),
        ("journal", paper.venue or ""),
        ("doi", paper.doi or ""),
        ("url", paper.entry_id),
    ]
    body = ",\n".join(f"  {name} = {{{_bibtex_escape(value)}}}" for name, value in fields if value)
    return f"@article{{{key},\n{body}\n}}"


def _citation_key(paper: Paper) -> str:
    family = paper.authors[0].split()[-1] if paper.authors else "unknown"
    title_word = next(iter(re.findall(r"[A-Za-z0-9]+", paper.title)), "paper")
    return re.sub(r"[^A-Za-z0-9]", "", f"{family}{paper.year or 'nd'}{title_word}")


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")
