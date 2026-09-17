import json
from urllib.error import HTTPError

from literature_rag.papers import Paper
from literature_rag.scholarly import (
    _categorize,
    _year_from_arxiv_id,
    enrich_papers,
)


def test_enrichment_writes_provenance_and_reuses_cache(tmp_path, monkeypatch):
    responses = [
        {
            "title": "Enriched",
            "authors": ("Ada Researcher",),
            "venue": "Journal",
            "year": 2025,
            "citation_count": 8,
            "pdf_url": "https://example.org/paper.pdf",
        }
    ]
    monkeypatch.setattr(
        "literature_rag.scholarly._lookup_metadata",
        lambda paper, email, key, stats: responses.pop(),
    )
    cache = tmp_path / "metadata.json"
    paper = Paper("1", "Original", (), "https://arxiv.org/abs/1", "")

    enriched = enrich_papers([paper], cache)
    reused = enrich_papers([paper], cache)

    assert enriched == reused
    assert enriched[0].title == "Enriched"
    data = json.loads(cache.read_text())["1"]
    assert data["retrieved_at"]
    assert "Semantic Scholar" in data["sources"]
    assert responses == []


def test_year_is_inferred_from_arxiv_id_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "literature_rag.scholarly._lookup_metadata",
        lambda paper, email, key, stats: {},
    )
    paper = Paper("2512.14242v1", "Recent", (), "https://arxiv.org/abs/2512.14242", "")

    enriched = enrich_papers([paper], tmp_path / "metadata.json")

    assert enriched[0].year == 2025


def test_year_fallback_patterns():
    assert _year_from_arxiv_id("2101.09878v1") == 2021
    assert _year_from_arxiv_id("2312.04432v2") == 2023
    assert _year_from_arxiv_id("2512.15759v1") == 2025
    assert _year_from_arxiv_id("10.5220/0012322100003648") is None
    assert _year_from_arxiv_id("local:abcdef1234567890") is None
    assert _year_from_arxiv_id("9901.12345v1") is None


def test_s2_api_key_is_sent_as_header(tmp_path, monkeypatch):
    captured = {}

    def fake_get_json(url, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return {"citationCount": 3, "year": 2024, "venue": "Conf"}

    monkeypatch.setattr("literature_rag.scholarly._get_json", fake_get_json)
    paper = Paper("2101.09878v1", "Paper", (), "https://arxiv.org/abs/2101.09878", "")

    enriched = enrich_papers([paper], tmp_path / "metadata.json", s2_api_key="s2-secret")

    assert "semanticscholar.org" in captured["url"]
    assert captured["headers"] == {"x-api-key": "s2-secret"}
    assert enriched[0].citation_count == 3


def test_error_categorization():
    assert _categorize(HTTPError("u", 404, "Not Found", None, None)) == "missing"
    assert _categorize(HTTPError("u", 429, "Too Many", None, None)) == "limited"
    assert _categorize(HTTPError("u", 500, "Server", None, None)) == "error"
    assert _categorize(RuntimeError("boom")) == "error"
