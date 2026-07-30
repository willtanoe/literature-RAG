from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from literature_rag.config import LLMConfig
from literature_rag.papers import DownloadedPaper, Paper
from literature_rag.resilience import atomic_write_json, atomic_write_text


def export_review(
    output_dir: Path,
    report: str,
    topic: str,
    analysis_question: str,
    downloaded: list[DownloadedPaper],
    llm_config: LLMConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "review.json"
    bibliography_path = output_dir / "references.bib"
    papers = [paper for _, paper in downloaded]
    atomic_write_text(report_path, report.rstrip() + "\n")
    manifest = {
        "topic": topic,
        "analysis_question": analysis_question,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": llm_config.model_name,
        "base_url": llm_config.base_url,
        "report_file": report_path.name,
        "bibliography_file": bibliography_path.name,
        "papers": [_paper_json(path, paper) for path, paper in downloaded],
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(bibliography_path, "\n\n".join(_bibtex(paper) for paper in papers) + "\n")
    print(f"Export -> Markdown: {report_path}")
    print(f"Export -> JSON: {manifest_path}")
    print(f"Export -> BibTeX: {bibliography_path}")
    return {"markdown": report_path, "json": manifest_path, "bibtex": bibliography_path}


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
        "url": paper.entry_id,
        "pdf": path.name,
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
