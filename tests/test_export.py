import json

from literature_rag.config import LLMConfig
from literature_rag.export import export_review
from literature_rag.papers import Paper


def test_exports_do_not_leak_secret_or_absolute_pdf_path(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    paper = Paper("1", "Study", ("Jane Doe",), "https://example.org", "", year=2025)
    paths = export_review(
        tmp_path / "output",
        "# Report",
        "topic",
        "objective",
        [(pdf, paper)],
        LLMConfig("https://llm.example/v1", "secret", "model"),
        sota_results=[{"paper": "Study"}],
        paper_summaries=[{"paper": "Study", "stated_future_work": "More tests"}],
        thesis_proposals="# Thesis Proposals\n\nProposal one.\n",
    )
    manifest_text = paths["json"].read_text()
    manifest = json.loads(manifest_text)
    assert "secret" not in manifest_text
    assert manifest["papers"][0]["pdf"] == "paper.pdf"
    assert manifest["sota_results"] == [{"paper": "Study"}]
    assert manifest["paper_summaries"][0]["stated_future_work"] == "More tests"
    assert manifest["thesis_proposals_file"] == "thesis_proposals.md"
    assert paths["thesis_proposals"].exists()
    assert "@article{Doe2025Study" in paths["bibtex"].read_text()


def test_exports_record_tiers_plan_and_gap_register(tmp_path):
    core_pdf = tmp_path / "core" / "core.pdf"
    core_pdf.parent.mkdir()
    core_pdf.write_bytes(b"%PDF-1.7")
    peripheral_pdf = tmp_path / "peripheral" / "edge.pdf"
    peripheral_pdf.parent.mkdir()
    peripheral_pdf.write_bytes(b"%PDF-1.7")
    downloaded = [
        (core_pdf, Paper("1", "Core", (), "https://example.org/1", "", tier="core")),
        (
            peripheral_pdf,
            Paper("2", "Edge", (), "https://example.org/2", "", tier="peripheral"),
        ),
    ]
    plan = {"version": 1, "queries": [{"query": "federated privacy", "tier": "core"}]}
    gap_register = [{"paper": "Core", "tier": "core", "kind": "limitation", "statement": "x"}]

    paths = export_review(
        tmp_path / "output",
        "# Report",
        "topic",
        "objective",
        downloaded,
        LLMConfig("https://llm.example/v1", "secret", "model"),
        gap_register=gap_register,
        thesis_candidates=[{"score": 7.5, "papers": ["Core", "Edge"]}],
        search_plan=plan,
    )

    manifest = json.loads(paths["json"].read_text())
    papers_manifest = json.loads(paths["papers_manifest"].read_text())
    assert manifest["tier_counts"] == {"core": 1, "peripheral": 1}
    assert manifest["gap_register"] == gap_register
    assert manifest["thesis_candidates"][0]["score"] == 7.5
    assert manifest["search_plan_file"] == "search_plan.json"
    assert json.loads(paths["search_plan"].read_text()) == plan
    folders = {item["tier"]: item["pdf_folder"] for item in papers_manifest["papers"]}
    assert folders == {"core": "core", "peripheral": "peripheral"}
