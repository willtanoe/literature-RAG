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
    )
    manifest_text = paths["json"].read_text()
    manifest = json.loads(manifest_text)
    assert "secret" not in manifest_text
    assert manifest["papers"][0]["pdf"] == "paper.pdf"
    assert "@article{Doe2025Study" in paths["bibtex"].read_text()
