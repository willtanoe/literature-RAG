import json

from literature_rag.papers import Paper
from literature_rag.scholarly import enrich_papers


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
    monkeypatch.setattr("literature_rag.scholarly._lookup_metadata", lambda *_: responses.pop())
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
