from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import arxiv

from literature_rag.__log__ import get_logger
from literature_rag.resilience import atomic_write_json, http_retry_after, retry
from literature_rag.settings import (
    ARXIV_METADATA_BATCH,
    DEFAULT_TIER,
    DOWNLOAD_DIR,
    DOWNLOAD_TIERS,
    MAX_PDF_BYTES,
    NETWORK_ATTEMPTS,
    NETWORK_TIMEOUT,
    PAPER_TIERS,
    SEARCH_CACHE_VERSION,
    TIER_DIRECTORIES,
)

logger = get_logger(__name__)

QUERY_STOPWORDS = frozenset(
    [
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "based",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "of",
        "on",
        "onto",
        "or",
        "that",
        "the",
        "their",
        "this",
        "those",
        "to",
        "toward",
        "towards",
        "under",
        "via",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "with",
        "within",
        "without",
        "review",
        "survey",
        "study",
        "approach",
        "approaches",
        "application",
        "applications",
        "method",
        "methods",
        "using",
        "using",
        "toward",
        "new",
        "novel",
        "evaluation",
        "evaluations",
    ]
)
MAX_QUERY_TERMS = 8


def _arxiv_query(topic: str, max_terms: int = MAX_QUERY_TERMS) -> str:
    cleaned = re.sub(r"[\"“”]", " ", topic)
    tokens = re.findall(r"[a-z0-9]+", cleaned.lower())
    significant = [
        token for token in dict.fromkeys(tokens) if len(token) >= 3 and token not in QUERY_STOPWORDS
    ]
    if not significant:
        raise ValueError(f"The research topic has no searchable keywords: {topic!r}")
    if len(significant) > max_terms:
        ranked = sorted(set(significant), key=lambda token: (-len(token), token))
        chosen = set(ranked[:max_terms])
        significant = [token for token in significant if token in chosen]
    return " AND ".join(f"all:{token}" for token in significant)


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
    abstract: str = ""
    tier: str = ""


DownloadedPaper = tuple[Path, Paper]


@dataclass(frozen=True)
class FailedDownload:
    target: Path
    paper: Paper
    error: str


def search_arxiv(
    topic: str,
    max_results: int,
    cache_path: Path | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    max_terms: int = MAX_QUERY_TERMS,
) -> list[Paper]:
    if not topic.strip():
        raise ValueError("The research topic cannot be empty.")
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")
    cached = load_search_cache(cache_path, topic, max_results)
    if cached is not None:
        print(f"Searching -> resumed {len(cached)} cached paper(s)")
        return cached

    print(f"Searching -> arXiv topic: {topic!r}")
    query = _arxiv_query(topic, max_terms)
    if year_min is not None or year_max is not None:
        start = f"{year_min:04d}01010000" if year_min is not None else "199001010000"
        end = f"{year_max:04d}12312359" if year_max is not None else "209912312359"
        query = f"{query} AND submittedDate:[{start} TO {end}]"
        print(f"Searching -> year filter: {year_min or 'any'}..{year_max or 'any'}")
    print(f"Searching -> arXiv query: {query!r}")
    search = arxiv.Search(
        query=query,
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
                abstract=_clean_abstract(result.summary),
            )
            for result in arxiv.Client().results(search)
        ]
    except Exception as exc:
        raise RuntimeError(f"arXiv search failed: {exc}") from exc
    if not results:
        raise RuntimeError(f"No arXiv papers found for {topic!r}.")
    save_search_cache(cache_path, topic, max_results, results)
    print(f"Searching -> found {len(results)} paper(s)")
    return results


def _clean_abstract(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def hydrate_abstracts(papers: list[Paper]) -> list[Paper]:
    missing = [paper for paper in papers if not paper.abstract and _is_arxiv_id(paper.arxiv_id)]
    if not missing:
        return papers
    print(f"Abstracts -> fetching {len(missing)} missing abstract(s) from arXiv")
    resolved: dict[str, str] = {}
    client = arxiv.Client()
    for start in range(0, len(missing), ARXIV_METADATA_BATCH):
        batch = missing[start : start + ARXIV_METADATA_BATCH]
        identifiers = [paper.arxiv_id for paper in batch]
        try:
            for result in client.results(arxiv.Search(id_list=identifiers)):
                short_id = result.get_short_id().lower()
                abstract = _clean_abstract(result.summary)
                resolved[short_id] = abstract
                resolved.setdefault(_bare_arxiv_id(short_id), abstract)
        except Exception as exc:
            print(f"Abstracts -> batch unavailable: {exc}")
            continue
    hydrated = []
    for paper in papers:
        if paper.abstract:
            hydrated.append(paper)
            continue
        abstract = resolved.get(paper.arxiv_id.lower()) or resolved.get(
            _bare_arxiv_id(paper.arxiv_id), ""
        )
        hydrated.append(replace(paper, abstract=abstract) if abstract else paper)
    filled = sum(1 for paper in hydrated if paper.abstract)
    print(f"Abstracts -> available for {filled}/{len(hydrated)} paper(s)")
    return hydrated


def _is_arxiv_id(value: str) -> bool:
    return not value.startswith(("doi:", "local:"))


def _bare_arxiv_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.strip().lower())


def download_papers(
    papers: list[Paper], download_dir: Path = DOWNLOAD_DIR
) -> tuple[list[DownloadedPaper], list[FailedDownload]]:
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[DownloadedPaper] = []
    failed: list[FailedDownload] = []
    reused = 0
    for index, paper in enumerate(papers, start=1):
        filename = _paper_filename(paper)
        target = _tier_target(download_dir, paper, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading -> [{index}/{len(papers)}] {paper.title}")
        try:
            existing = find_existing_pdf(download_dir, filename)
            if existing is not None:
                downloaded.append((existing, paper))
                reused += 1
                continue
            _download_pdf(paper.pdf_url, target)
            if not _is_pdf(target):
                raise OSError("downloaded file is not a valid PDF")
            downloaded.append((target, paper))
        except Exception as exc:
            print(f"Downloading -> skipped {paper.arxiv_id}: {exc}")
            failed.append(FailedDownload(target, paper, str(exc)))
    if reused:
        print(f"Downloading -> reused {reused} PDF(s) already present in {download_dir}")
    print(f"Downloading -> ready: {len(downloaded)} PDF(s) in {download_dir}")
    return downloaded, failed


def _tier_target(download_dir: Path, paper: Paper, filename: str) -> Path:
    directory = TIER_DIRECTORIES.get(paper.tier)
    return (download_dir / directory / filename) if directory else download_dir / filename


def find_existing_pdf(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if _is_pdf(direct):
        return direct
    if not root.is_dir():
        return None
    for candidate in sorted(root.rglob(filename)):
        if _is_pdf(candidate):
            return candidate
    return None


def organize_papers_by_tier(
    downloaded: list[DownloadedPaper], papers_root: Path
) -> list[DownloadedPaper]:
    organized: list[DownloadedPaper] = []
    moved = 0
    counts: dict[str, int] = {}
    for path, paper in downloaded:
        tier = paper.tier if paper.tier in PAPER_TIERS else DEFAULT_TIER
        counts[tier] = counts.get(tier, 0) + 1
        directory = TIER_DIRECTORIES.get(tier)
        if directory is None:
            organized.append((path, paper))
            continue
        target = papers_root / directory / path.name
        if path.resolve() == target.resolve():
            organized.append((path, paper))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(target))
            moved += 1
        except OSError as exc:
            print(f"Tiering -> could not move {path.name}: {exc}")
            organized.append((path, paper))
            continue
        organized.append((target, paper))
    layout = ", ".join(f"{tier}: {counts[tier]}" for tier in PAPER_TIERS if tier in counts)
    print(f"Tiering -> moved {moved} PDF(s) into tier folders ({layout or 'no papers'})")
    return organized


def import_workspace_pdfs(
    papers_root: Path, known: list[DownloadedPaper] | None = None
) -> list[DownloadedPaper]:
    if not papers_root.is_dir():
        return []
    claimed = {path.resolve() for path, _paper in known or []}
    extras = [
        candidate
        for candidate in sorted(papers_root.rglob("*.pdf"))
        if candidate.resolve() not in claimed
    ]
    if not extras:
        return []
    print(f"Workspace import -> found {len(extras)} unclaimed PDF(s) under {papers_root}")
    imported: list[DownloadedPaper] = []
    for source in extras:
        if not _is_pdf(source):
            print(f"Workspace import -> skipped invalid PDF: {source.name}")
            continue
        imported.append((source, _pdf_paper(source, _workspace_tier(source, papers_root))))
    print(f"Workspace import -> ready {len(imported)} PDF(s)")
    return imported


def _workspace_tier(path: Path, papers_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(papers_root.resolve())
    except ValueError:
        return DEFAULT_TIER
    head = relative.parts[0] if len(relative.parts) > 1 else ""
    for tier in DOWNLOAD_TIERS:
        if head == TIER_DIRECTORIES[tier]:
            return tier
    return DEFAULT_TIER


def _pdf_paper(source: Path, tier: str = "") -> Paper:
    digest = _file_hash(source)
    return Paper(
        arxiv_id=f"local:{digest[:16]}",
        title=source.stem.replace("_", " "),
        authors=(),
        entry_id=source.resolve().as_uri(),
        pdf_url="",
        tier=tier,
    )


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
                tier="core",
            )
        )
    return papers


def import_local_pdfs(
    source_dir: Path, destination_dir: Path, tier: str = "core"
) -> list[DownloadedPaper]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Local PDF folder does not exist: {source_dir}")
    directory = TIER_DIRECTORIES.get(tier, "")
    destination_root = destination_dir / directory if directory else destination_dir
    destination_root.mkdir(parents=True, exist_ok=True)
    imported: list[DownloadedPaper] = []
    for source in sorted(source_dir.rglob("*.pdf")):
        if not _is_pdf(source):
            print(f"Import -> skipped invalid PDF: {source.name}")
            continue
        digest = _file_hash(source)
        target = destination_root / f"local_{digest[:12]}_{_safe_filename(source.stem)}.pdf"
        if source.resolve() != target.resolve() and not target.exists():
            shutil.copy2(source, target)
        paper = Paper(
            arxiv_id=f"local:{digest[:16]}",
            title=source.stem.replace("_", " "),
            authors=(),
            entry_id=source.resolve().as_uri(),
            pdf_url="",
            tier=tier,
        )
        imported.append((target, paper))
        print(f"Import -> ready {source.name}")
    return imported


def deduplicate_downloads(papers: list[DownloadedPaper]) -> list[DownloadedPaper]:
    unique: list[DownloadedPaper] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for path, paper in papers:
        identity = _paper_identity(paper)
        digest = _file_hash(path)
        if identity in seen_ids or digest in seen_hashes:
            print(f"Deduplication -> skipped duplicate: {paper.title}")
            continue
        seen_ids.add(identity)
        seen_hashes.add(digest)
        unique.append((path, paper))
    print(f"Deduplication -> {len(unique)} unique PDF(s)")
    return unique


def recover_manual_downloads(
    downloaded: list[DownloadedPaper], failed: list[FailedDownload]
) -> list[DownloadedPaper]:
    if not failed:
        return downloaded

    print("\nManual download required")
    print("Some PDFs could not be retrieved automatically. This can be caused by")
    print("network restrictions, unavailable files, or publisher access controls.")
    print("Pressing Enter retries the automatic download and rechecks. If it still")
    print("fails, open the arXiv/DOI URL below in a browser, save the PDF to the")
    print("exact 'Save as' path, then press Enter again. Type 's' to skip.")
    for index, item in enumerate(failed, start=1):
        print(f"\n[{index}] {item.paper.title}")
        print(f"    arXiv: {item.paper.entry_id}")
        if item.paper.doi:
            print(f"    DOI: https://doi.org/{item.paper.doi}")
        print(f"    Save as: {item.target.resolve()}")
        print(f"    Reason: {item.error}")

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
            if not _is_pdf(item.target) and item.paper.pdf_url:
                with contextlib.suppress(Exception):
                    _download_pdf(item.paper.pdf_url, item.target)
            if _is_pdf(item.target):
                downloaded.append((item.target, item.paper))
                print(f"Manual download -> recovered {item.target.name}")
            else:
                remaining.append(item)
                print(f"Manual download -> still missing/invalid: {item.target.name}")
        pending = remaining

    if not downloaded:
        raise RuntimeError("No valid PDFs are available to index.")
    if pending:
        print(f"Manual download -> skipped {len(pending)} paper(s)")
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
                abstract=item.get("abstract") or "",
                tier=item.get("tier") or "",
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
                "abstract": paper.abstract,
                "tier": paper.tier,
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
