from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import arxiv

from literature_rag.__log__ import get_logger
from literature_rag.resilience import atomic_write_json, http_retry_after, retry
from literature_rag.settings import (
    DOWNLOAD_DIR,
    MAX_PDF_BYTES,
    NETWORK_ATTEMPTS,
    NETWORK_TIMEOUT,
    SEARCH_CACHE_VERSION,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    entry_id: str
    pdf_url: str
    doi: str | None = None
    venue: str | None = None
    year: int | None = None
    citation_count: int | None = None
    is_open_access: bool | None = None


DownloadedPaper = tuple[Path, Paper]


@dataclass(frozen=True)
class FailedDownload:
    target: Path
    paper: Paper
    error: str


def search_arxiv(topic: str, max_results: int, cache_path: Path | None = None) -> list[Paper]:
    if not topic.strip():
        raise ValueError("The research topic cannot be empty.")
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")
    cached = load_search_cache(cache_path, topic, max_results)
    if cached is not None:
        logger.info(f"Resumed {len(cached)} cached paper(s)")
        return cached

    logger.info(f"Searching arXiv for topic: {topic!r}")
    search = arxiv.Search(
        query=f'all:"{topic.strip()}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        results = [
            Paper(
                arxiv_id=result.get_short_id(),
                title=result.title,
                authors=tuple(author.name for author in result.authors),
                entry_id=result.entry_id,
                pdf_url=result.pdf_url,
                doi=result.doi,
            )
            for result in arxiv.Client().results(search)
        ]
    except Exception as exc:
        raise RuntimeError(f"arXiv search failed: {exc}") from exc
    if not results:
        raise RuntimeError(f"No arXiv papers found for {topic!r}.")
    save_search_cache(cache_path, topic, max_results, results)
    logger.info(f"Found {len(results)} paper(s)")
    return results


def download_papers(
    papers: list[Paper], download_dir: Path = DOWNLOAD_DIR
) -> tuple[list[DownloadedPaper], list[FailedDownload]]:
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[DownloadedPaper] = []
    failed: list[FailedDownload] = []
    for index, paper in enumerate(papers, start=1):
        filename = _paper_filename(paper)
        target = download_dir / filename
        logger.info(f"Downloading [{index}/{len(papers)}] {paper.title}")
        try:
            if not _is_pdf(target):
                _download_pdf(paper.pdf_url, target)
            if not _is_pdf(target):
                raise OSError("downloaded file is not a valid PDF")
            downloaded.append((target, paper))
        except Exception as exc:
            logger.warning(f"Skipping {paper.arxiv_id}: {exc}")
            failed.append(FailedDownload(target, paper, str(exc)))
    logger.info(f"Ready: {len(downloaded)} PDF(s) in {download_dir}")
    return downloaded, failed


def papers_from_dois(dois: list[str]) -> list[Paper]:
    papers = []
    seen = set()
    for value in dois:
        doi = _normalize_doi(value)
        if not doi or doi in seen:
            continue
        seen.add(doi)
        papers.append(
            Paper(
                arxiv_id=f"doi:{doi}",
                title=f"DOI {doi}",
                authors=(),
                entry_id=f"https://doi.org/{doi}",
                pdf_url="",
                doi=doi,
            )
        )
    return papers


def import_local_pdfs(source_dir: Path, destination_dir: Path) -> list[DownloadedPaper]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Local PDF folder does not exist: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    imported: list[DownloadedPaper] = []
    for source in sorted(source_dir.glob("*.pdf")):
        if not _is_pdf(source):
            logger.warning(f"Skipping invalid PDF: {source.name}")
            continue
        digest = _file_hash(source)
        target = destination_dir / f"local_{digest[:12]}_{_safe_filename(source.stem)}.pdf"
        if source.resolve() != target.resolve() and not target.exists():
            shutil.copy2(source, target)
        paper = Paper(
            arxiv_id=f"local:{digest[:16]}",
            title=source.stem.replace("_", " "),
            authors=(),
            entry_id=source.resolve().as_uri(),
            pdf_url="",
        )
        imported.append((target, paper))
        logger.info(f"Imported {source.name}")
    return imported


def deduplicate_downloads(papers: list[DownloadedPaper]) -> list[DownloadedPaper]:
    unique: list[DownloadedPaper] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for path, paper in papers:
        identity = _paper_identity(paper)
        digest = _file_hash(path)
        if identity in seen_ids or digest in seen_hashes:
            logger.debug(f"Skipped duplicate: {paper.title}")
            continue
        seen_ids.add(identity)
        seen_hashes.add(digest)
        unique.append((path, paper))
    logger.info(f"{len(unique)} unique PDF(s)")
    return unique


def recover_manual_downloads(
    downloaded: list[DownloadedPaper], failed: list[FailedDownload]
) -> list[DownloadedPaper]:
    if not failed:
        return downloaded

    logger.warning("Manual download required")
    logger.warning("Some PDFs could not be retrieved automatically. This can be caused by")
    logger.warning("network restrictions, unavailable files, or publisher access controls.")
    for index, item in enumerate(failed, start=1):
        logger.info(f"[{index}] {item.paper.title}")
        logger.info(f"    arXiv: {item.paper.entry_id}")
        if item.paper.doi:
            logger.info(f"    DOI: https://doi.org/{item.paper.doi}")
        logger.info(f"    Save as: {item.target.resolve()}")
        logger.info(f"    Reason: {item.error}")

    pending = failed
    while pending:
        try:
            choice = (
                input(
                    "\nDownload and save the PDF(s), then press Enter to recheck "
                    "or type 's' to skip: "
                )
                .strip()
                .lower()
            )
        except EOFError:
            choice = "s"
        if choice == "s":
            break

        remaining: list[FailedDownload] = []
        for item in pending:
            if _is_pdf(item.target):
                downloaded.append((item.target, item.paper))
                logger.info(f"Found {item.target.name}")
            else:
                remaining.append(item)
                logger.info(f"Still missing/invalid: {item.target.name}")
        pending = remaining

    if not downloaded:
        raise RuntimeError("No valid PDFs are available to index.")
    if pending:
        logger.info(f"Skipped {len(pending)} paper(s)")
    return downloaded


def _paper_filename(paper: Paper) -> str:
    arxiv_id = paper.arxiv_id.replace("/", "_")
    return f"{_safe_filename(arxiv_id)}_{_safe_filename(paper.title)}.pdf"


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:120] or "paper"


def _is_pdf(path: Path) -> bool:
    try:
        if path.stat().st_size < 5 or path.stat().st_size > MAX_PDF_BYTES:
            return False
        with path.open("rb") as file:
            return file.read(5) == b"%PDF-"
    except OSError:
        return False


def _download_pdf(url: str, target: Path) -> None:
    if not url:
        raise OSError("no open-access PDF URL is known")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise OSError("PDF URL must use HTTPS")
    temporary = target.with_suffix(target.suffix + ".part")
    try:

        def transfer() -> None:
            request = Request(url, headers={"User-Agent": "literature-rag/0.1"})
            with (
                urlopen(request, timeout=NETWORK_TIMEOUT) as response,
                temporary.open("wb") as output,
            ):
                final_url = urlparse(response.geturl())
                if final_url.scheme != "https":
                    raise OSError("PDF download redirected to a non-HTTPS URL")
                content_type = response.headers.get_content_type()
                if content_type not in {"application/pdf", "application/octet-stream"}:
                    raise OSError(f"unexpected content type: {content_type}")
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_PDF_BYTES:
                    raise OSError("PDF exceeds the configured size limit")
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_PDF_BYTES:
                        raise OSError("PDF exceeds the configured size limit")
                    output.write(block)

        retry(transfer, attempts=NETWORK_ATTEMPTS, retry_after=http_retry_after)
        if not _is_pdf(temporary):
            raise OSError("response is not a valid PDF")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_search_cache(
    cache_path: Path | None,
    topic: str,
    max_results: int,
    cache_identity: str = "",
) -> list[Paper] | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            data.get("version") != SEARCH_CACHE_VERSION
            or data.get("topic") != topic
            or data.get("max_results") != max_results
            or data.get("cache_identity", "") != cache_identity
        ):
            return None
        return [
            Paper(
                arxiv_id=item["arxiv_id"],
                title=item["title"],
                authors=tuple(item["authors"]),
                entry_id=item["entry_id"],
                pdf_url=item["pdf_url"],
                doi=item.get("doi"),
                venue=item.get("venue"),
                year=item.get("year"),
                citation_count=item.get("citation_count"),
                is_open_access=item.get("is_open_access"),
            )
            for item in data["papers"]
        ]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return None


def save_search_cache(
    cache_path: Path | None,
    topic: str,
    max_results: int,
    papers: list[Paper],
    cache_identity: str = "",
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": SEARCH_CACHE_VERSION,
        "topic": topic,
        "max_results": max_results,
        "cache_identity": cache_identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "papers": [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": list(paper.authors),
                "entry_id": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "venue": paper.venue,
                "year": paper.year,
                "citation_count": paper.citation_count,
                "is_open_access": paper.is_open_access,
            }
            for paper in papers
        ],
    }
    atomic_write_json(cache_path, data)


def _normalize_doi(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value.removeprefix("doi:").strip()


def _paper_identity(paper: Paper) -> str:
    return f"doi:{_normalize_doi(paper.doi)}" if paper.doi else paper.arxiv_id.lower()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
