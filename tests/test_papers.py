import pytest

from literature_rag.papers import (
    FailedDownload,
    Paper,
    _arxiv_query,
    _paper_filename,
    deduplicate_downloads,
    download_papers,
    find_existing_pdf,
    hydrate_abstracts,
    import_workspace_pdfs,
    load_search_cache,
    organize_papers_by_tier,
    papers_from_dois,
    recover_manual_downloads,
    save_search_cache,
)


def test_manual_recovery_retries_download_on_enter(tmp_path, monkeypatch):
    paper = Paper(
        "2209.14086v2",
        "Momentum Gradient Descent FL",
        ("Author",),
        "https://arxiv.org/abs/2209.14086",
        "https://arxiv.org/pdf/2209.14086",
    )
    target = tmp_path / "paper.pdf"
    failed = [FailedDownload(target, paper, "network hiccup")]

    def fake_download(url, destination):
        destination.write_bytes(b"%PDF-1.7 recovered")

    monkeypatch.setattr("literature_rag.papers._download_pdf", fake_download)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    downloaded = recover_manual_downloads([], failed)

    assert downloaded == [(target, paper)]
    assert target.read_bytes().startswith(b"%PDF-")


def test_manual_recovery_keeps_missing_after_failed_retry(tmp_path, monkeypatch):
    paper = Paper("1", "Missing", (), "e1", "https://x/p.pdf")
    target = tmp_path / "missing.pdf"
    failed = [FailedDownload(target, paper, "unavailable")]
    existing = tmp_path / "existing.pdf"
    existing.write_bytes(b"%PDF-1.7 existing")
    kept = (existing, Paper("2", "Existing", (), "e2", ""))

    def fake_download(url, destination):
        raise OSError("still down")

    monkeypatch.setattr("literature_rag.papers._download_pdf", fake_download)
    monkeypatch.setattr("builtins.input", lambda _prompt: "s")

    downloaded = recover_manual_downloads([kept], failed)

    assert downloaded == [kept]
    assert not target.exists()


def test_arxiv_query_and_terms_without_stopwords():
    query = _arxiv_query('Federated Learning for "Intrusion Detection"')
    assert query == "all:federated AND all:learning AND all:intrusion AND all:detection"


def test_arxiv_query_deduplicates_and_ranks_long_terms():
    query = _arxiv_query(
        "Robust Federated Learning for Network Intrusion Detection "
        "under Non-IID and Adversarial Conditions"
    )
    terms = query.split(" AND ")
    assert len(terms) == 8
    assert "all:adversarial" in terms
    assert "all:detection" in terms
    assert len(set(terms)) == 8


def test_arxiv_query_rejects_keyword_free_topic():
    with pytest.raises(ValueError):
        _arxiv_query('"the and of"')


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


def test_search_cache_round_trips_abstract_and_tier(tmp_path):
    path = tmp_path / "search.json"
    paper = Paper(
        "2101.00001",
        "Title",
        ("Author",),
        "https://arxiv.org/abs/2101.00001",
        "https://x/p.pdf",
        abstract="We study differential privacy for federated intrusion detection.",
        tier="core",
    )
    save_search_cache(path, "topic", 5, [paper])
    restored = load_search_cache(path, "topic", 5)
    assert restored == [paper]
    assert restored[0].abstract.startswith("We study")
    assert restored[0].tier == "core"


def test_download_reuses_pdf_already_filed_under_a_tier(tmp_path, monkeypatch):
    paper = Paper(
        "2209.14086v2",
        "Momentum Gradient Descent FL",
        ("Author",),
        "https://arxiv.org/abs/2209.14086",
        "https://arxiv.org/pdf/2209.14086",
        tier="core",
    )
    filed = tmp_path / "core" / _paper_filename(paper)
    filed.parent.mkdir(parents=True)
    filed.write_bytes(b"%PDF-1.7 already downloaded")

    def fail_download(url, destination):
        raise AssertionError("download must not run when the PDF already exists")

    monkeypatch.setattr("literature_rag.papers._download_pdf", fail_download)

    downloaded, failed = download_papers([paper], tmp_path)

    assert downloaded == [(filed, paper)]
    assert failed == []
    assert find_existing_pdf(tmp_path, _paper_filename(paper)) == filed


def test_download_places_new_pdf_in_its_tier_folder(tmp_path, monkeypatch):
    paper = Paper("1", "Edge Case", (), "e1", "https://x/p.pdf", tier="peripheral")

    def fake_download(url, destination):
        destination.write_bytes(b"%PDF-1.7 fetched")

    monkeypatch.setattr("literature_rag.papers._download_pdf", fake_download)

    downloaded, failed = download_papers([paper], tmp_path)

    assert failed == []
    assert downloaded[0][0].parent.name == "peripheral"


def test_organize_papers_by_tier_moves_pdfs_into_tier_folders(tmp_path):
    core = tmp_path / "core.pdf"
    core.write_bytes(b"%PDF-1.7 core")
    stray = tmp_path / "stray.pdf"
    stray.write_bytes(b"%PDF-1.7 stray")
    downloaded = [
        (core, Paper("1", "Core", (), "e1", "", tier="core")),
        (stray, Paper("2", "Stray", (), "e2", "")),
    ]

    organized = organize_papers_by_tier(downloaded, tmp_path)

    assert organized[0][0] == tmp_path / "core" / "core.pdf"
    assert organized[1][0] == tmp_path / "related" / "stray.pdf"
    assert not core.exists()
    assert organize_papers_by_tier(organized, tmp_path) == organized


def test_import_workspace_pdfs_infers_tier_and_skips_claimed(tmp_path):
    claimed = tmp_path / "core" / "claimed.pdf"
    claimed.parent.mkdir(parents=True)
    claimed.write_bytes(b"%PDF-1.7 claimed")
    orphan = tmp_path / "peripheral" / "orphan.pdf"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"%PDF-1.7 orphan")
    legacy = tmp_path / "legacy.pdf"
    legacy.write_bytes(b"%PDF-1.7 legacy")
    known = [(claimed, Paper("1", "Claimed", (), "e1", "", tier="core"))]

    imported = import_workspace_pdfs(tmp_path, known)

    tiers = {path.name: paper.tier for path, paper in imported}
    assert tiers == {"orphan.pdf": "peripheral", "legacy.pdf": "related"}
    assert all(paper.arxiv_id.startswith("local:") for _path, paper in imported)


def test_hydrate_abstracts_fills_missing_summaries(monkeypatch):
    papers = [
        Paper("2101.00001v1", "Missing", (), "e1", "p1"),
        Paper("2101.00002", "Present", (), "e2", "p2", abstract="kept"),
        Paper("local:abc", "Local", (), "e3", ""),
    ]

    class FakeResult:
        def __init__(self, identifier, summary):
            self.identifier = identifier
            self.summary = summary

        def get_short_id(self):
            return self.identifier

    class FakeClient:
        def results(self, search):
            assert search.id_list == ["2101.00001v1"]
            return [FakeResult("2101.00001v1", "fetched  abstract\ntext")]

    monkeypatch.setattr("literature_rag.papers.arxiv.Client", FakeClient)
    monkeypatch.setattr(
        "literature_rag.papers.arxiv.Search",
        lambda id_list: type("Search", (), {"id_list": id_list})(),
    )

    hydrated = hydrate_abstracts(papers)

    assert hydrated[0].abstract == "fetched abstract text"
    assert hydrated[1].abstract == "kept"
    assert hydrated[2].abstract == ""
