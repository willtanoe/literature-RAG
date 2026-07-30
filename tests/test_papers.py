from literature_rag.papers import (
    Paper,
    deduplicate_downloads,
    load_search_cache,
    papers_from_dois,
    save_search_cache,
)


def test_doi_normalization_and_deduplication(tmp_path):
    papers = papers_from_dois(["https://doi.org/10.X/A", "doi:10.x/a"])
    assert len(papers) == 1
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-1.7 same")
    second.write_bytes(b"%PDF-1.7 same")
    local = Paper("local:1", "Local", (), "file:///a", "")
    assert len(deduplicate_downloads([(first, papers[0]), (second, local)])) == 1


def test_search_cache_identity_prevents_stale_reuse(tmp_path):
    path = tmp_path / "search.json"
    paper = Paper("1", "Title", ("Author",), "https://arxiv.org/abs/1", "https://x/p.pdf")
    save_search_cache(path, "topic", 5, [paper], "objective-a")
    assert load_search_cache(path, "topic", 5, "objective-a") == [paper]
    assert load_search_cache(path, "topic", 5, "objective-b") is None
