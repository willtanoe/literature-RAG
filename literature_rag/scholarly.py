from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from literature_rag.__log__ import get_logger
from literature_rag.papers import Paper
from literature_rag.resilience import atomic_write_json, http_retry_after, retry
from literature_rag.settings import NETWORK_ATTEMPTS, NETWORK_TIMEOUT

logger = get_logger(__name__)


def enrich_papers(
    papers: list[Paper], cache_path: Path | None = None, unpaywall_email: str = ""
) -> list[Paper]:
    cache = _load_cache(cache_path)
    enriched = []
    changed = False
    for paper in papers:
        key = paper.doi or paper.arxiv_id
        data = cache.get(key)
        if data is None:
            values = _lookup_metadata(paper, unpaywall_email)
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
        enriched.append(replace(paper, **updates))
    if changed and cache_path is not None:
        atomic_write_json(cache_path, cache)
    return enriched


def _lookup_metadata(paper: Paper, email: str) -> dict:
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
        except Exception as exc:
            logger.debug(f"Crossref metadata unavailable for {paper.doi}: {exc}")
        if email:
            try:
                oa = _get_json(
                    f"https://api.unpaywall.org/v2/{quote(paper.doi)}?email={quote(email)}"
                )
                location = oa.get("best_oa_location") or {}
                data["pdf_url"] = location.get("url_for_pdf") or paper.pdf_url
                data["is_open_access"] = bool(oa.get("is_oa"))
            except Exception as exc:
                logger.debug(f"Unpaywall OA lookup unavailable for {paper.doi}: {exc}")
    try:
        identity = f"DOI:{paper.doi}" if paper.doi else f"ARXIV:{paper.arxiv_id}"
        fields = "citationCount,year,venue,openAccessPdf"
        endpoint = "https://api.semanticscholar.org/graph/v1/paper"
        semantic = _get_json(f"{endpoint}/{quote(identity, safe='')}?fields={fields}")
        data.update(
            citation_count=semantic.get("citationCount"),
            year=semantic.get("year") or data.get("year"),
            venue=semantic.get("venue") or data.get("venue"),
        )
        oa_pdf = semantic.get("openAccessPdf") or {}
        data["pdf_url"] = data.get("pdf_url") or oa_pdf.get("url") or paper.pdf_url
    except Exception as exc:
        logger.debug(f"Semantic Scholar metadata unavailable for {key_label(paper)}: {exc}")
    return data


def _get_json(url: str) -> dict:
    def fetch() -> dict:
        request = Request(url, headers={"User-Agent": "literature-rag/0.1"})
        with urlopen(request, timeout=NETWORK_TIMEOUT) as response:
            return json.load(response)

    return retry(fetch, attempts=NETWORK_ATTEMPTS, retry_after=http_retry_after)


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
