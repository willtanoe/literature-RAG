from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from literature_rag.papers import Paper
from literature_rag.resilience import atomic_write_json, http_retry_after, retry
from literature_rag.settings import NETWORK_ATTEMPTS, NETWORK_TIMEOUT

LOOKUP_THROTTLE_SECONDS = 1.1
ARXIV_YEAR_PATTERN = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
ARXIV_YEAR_MIN = 2007


def enrich_papers(
    papers: list[Paper],
    cache_path: Path | None = None,
    unpaywall_email: str = "",
    s2_api_key: str = "",
) -> list[Paper]:
    cache = _load_cache(cache_path)
    enriched = []
    changed = False
    lookup_pending = False
    stats: Counter[str] = Counter()
    for index, paper in enumerate(papers, start=1):
        print(f"Metadata -> paper {index}/{len(papers)}: {paper.title}")
        key = paper.doi or paper.arxiv_id
        data = cache.get(key)
        if data is None:
            if lookup_pending:
                time.sleep(LOOKUP_THROTTLE_SECONDS)
            lookup_pending = True
            values = _lookup_metadata(paper, unpaywall_email, s2_api_key, stats)
            cache[key] = {
                "values": values,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sources": ["Crossref", "Semantic Scholar"]
                + (["Unpaywall"] if unpaywall_email and paper.doi else []),
            }
            data = values
            changed = True
        elif "values" in data:
            data = data["values"]
        updates = {k: v for k, v in data.items() if v is not None}
        if "authors" in updates:
            updates["authors"] = tuple(updates["authors"])
        if not updates.get("year"):
            inferred = _year_from_arxiv_id(paper.arxiv_id)
            if inferred:
                updates["year"] = inferred
                stats["year_inferred"] += 1
        enriched.append(replace(paper, **updates))
    if stats:
        _print_summary(stats)
    if changed and cache_path is not None:
        atomic_write_json(cache_path, cache)
    return enriched


def _lookup_metadata(paper: Paper, email: str, s2_api_key: str, stats: Counter[str]) -> dict:
    data: dict = {}
    if paper.doi:
        try:
            message = _get_json(f"https://api.crossref.org/works/{quote(paper.doi)}")["message"]
            data.update(
                title=(message.get("title") or [paper.title])[0],
                authors=tuple(
                    " ".join(filter(None, (author.get("given"), author.get("family"))))
                    for author in message.get("author", [])
                )
                or paper.authors,
                venue=(message.get("container-title") or [None])[0],
                year=_crossref_year(message),
            )
            stats["crossref_ok"] += 1
        except Exception as exc:
            stats[f"crossref_{_categorize(exc)}"] += 1
        if email:
            try:
                oa = _get_json(
                    f"https://api.unpaywall.org/v2/{quote(paper.doi)}?email={quote(email)}"
                )
                location = oa.get("best_oa_location") or {}
                data["pdf_url"] = location.get("url_for_pdf") or paper.pdf_url
                data["is_open_access"] = bool(oa.get("is_oa"))
                stats["unpaywall_ok"] += 1
            except Exception as exc:
                stats[f"unpaywall_{_categorize(exc)}"] += 1
    try:
        identity = f"DOI:{paper.doi}" if paper.doi else f"ARXIV:{paper.arxiv_id}"
        fields = "citationCount,year,venue,openAccessPdf"
        endpoint = "https://api.semanticscholar.org/graph/v1/paper"
        headers = {"x-api-key": s2_api_key} if s2_api_key else None
        semantic = _get_json(f"{endpoint}/{quote(identity, safe='')}?fields={fields}", headers)
        data.update(
            citation_count=semantic.get("citationCount"),
            year=semantic.get("year") or data.get("year"),
            venue=semantic.get("venue") or data.get("venue"),
        )
        oa_pdf = semantic.get("openAccessPdf") or {}
        data["pdf_url"] = data.get("pdf_url") or oa_pdf.get("url") or paper.pdf_url
        stats["semanticscholar_ok"] += 1
    except Exception as exc:
        stats[f"semanticscholar_{_categorize(exc)}"] += 1
    return data


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    def fetch() -> dict:
        request_headers = {"User-Agent": "literature-rag/0.1"}
        if headers:
            request_headers.update(headers)
        request = Request(url, headers=request_headers)
        with urlopen(request, timeout=NETWORK_TIMEOUT) as response:
            return json.load(response)

    return retry(fetch, attempts=NETWORK_ATTEMPTS, retry_after=http_retry_after)


def _categorize(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        if exc.code == 404:
            return "missing"
        if exc.code == 429:
            return "limited"
    return "error"


def _year_from_arxiv_id(arxiv_id: str) -> int | None:
    match = ARXIV_YEAR_PATTERN.match(arxiv_id.strip().lower())
    if not match:
        return None
    year = 2000 + int(match.group(1))
    if ARXIV_YEAR_MIN <= year <= datetime.now().year + 1:
        return year
    return None


def _print_summary(stats: Counter[str]) -> None:
    def service(name: str) -> str:
        buckets = {
            label: stats.get(f"{name}_{label}", 0)
            for label in ("ok", "missing", "limited", "error")
        }
        if not any(buckets.values()):
            return ""
        parts = [f"{buckets['ok']} ok"]
        if buckets["missing"]:
            parts.append(f"{buckets['missing']} not found")
        if buckets["limited"]:
            parts.append(f"{buckets['limited']} rate-limited")
        if buckets["error"]:
            parts.append(f"{buckets['error']} errors")
        return f"{name}: {', '.join(parts)}"

    lines = [
        line
        for line in (
            service("Crossref"),
            service("Unpaywall"),
            service("Semantic Scholar"),
        )
        if line
    ]
    if stats.get("year_inferred"):
        lines.append(f"year inferred from arXiv ID: {stats['year_inferred']}")
    print("Metadata -> " + "; ".join(lines))


def _crossref_year(message: dict) -> int | None:
    parts = (message.get("published") or {}).get("date-parts") or []
    return parts[0][0] if parts and parts[0] else None


def _load_cache(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def key_label(paper: Paper) -> str:
    return paper.doi or paper.arxiv_id
