from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from literature_rag.ingestion import (
    _clean_page_text,
    _detect_section,
    _drop_repeated_lines,
    _is_bibliography_page,
    _load_safe_index,
    _save_safe_index,
    load_and_split_papers,
)
from literature_rag.papers import Paper
from literature_rag.settings import MAX_CHUNKS


def test_max_chunks_enforced_strictly(tmp_path, monkeypatch):
    """Verify that MAX_CHUNKS is enforced and cannot be exceeded."""
    fake_doc = Document(
        page_content="This is sample text that will be split into many chunks for testing.",
        metadata={"page": 1},
    )

    def mock_load(self):
        return [fake_doc]

    class ManyChunkSplitter:
        CHUNK_COUNT = MAX_CHUNKS + 50

        def split_documents(self, docs):
            return [
                Document(page_content=f"chunk content {i}", metadata={"page": 1})
                for _ in docs
                for i in range(self.CHUNK_COUNT)
            ]

    monkeypatch.setattr(
        "literature_rag.ingestion.RecursiveCharacterTextSplitter",
        lambda *_, **__: ManyChunkSplitter(),
    )
    monkeypatch.setattr("langchain_community.document_loaders.PyPDFLoader.load", mock_load)

    path = tmp_path / "doc_0.pdf"
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n%%EOF")
    papers = [(path, Paper("id0", "Title", (), "http://ex/i", ""))]

    result = load_and_split_papers(papers)
    assert len(result) <= MAX_CHUNKS


def test_safe_faiss_roundtrip_uses_json_not_pickle(tmp_path):
    embeddings = FakeEmbeddings(size=8)
    source = FAISS.from_documents(
        [Document(page_content="evidence", metadata={"evidence_id": "P1-p1-c1"})],
        embeddings,
    )

    _save_safe_index(source, tmp_path)
    restored = _load_safe_index(tmp_path, embeddings)

    assert not (tmp_path / "index.pkl").exists()
    assert (tmp_path / "documents.json").exists()
    assert restored.similarity_search("evidence", k=1)[0].page_content == "evidence"


def test_bibliography_page_with_heading_is_detected():
    text = (
        "References\n\n"
        "[1] A. Author, B. Writer, and C. Researcher. On federated learning "
        "aggregation methods. Journal of Things, 2021.\n"
        "[2] D. Scientist et al. Another study about intrusion detection systems "
        "with several authors and a long title. arXiv:2103.12345.\n"
    )
    assert _is_bibliography_page(text)


def test_dense_citation_page_without_heading_is_detected():
    entries = " ".join(
        f"[{i}] Author {i} et al. published a study about topic number {i} "
        f"with DOI 10.1000/example{i} in 2020."
        for i in range(1, 26)
    )
    assert _is_bibliography_page(entries)


def test_normal_content_page_is_kept():
    text = (
        "We evaluate our federated intrusion detection framework on three public "
        "datasets. Prior work [1] established the baseline, and follow-up studies "
        "[2,3] improved robustness under non-IID data distributions. Our method "
        "differs by combining frequency analysis with client selection, and we "
        "report accuracy, false positive rate, and response time for every "
        "experiment configuration in the evaluation section below."
    )
    assert not _is_bibliography_page(text)


def test_short_or_empty_pages_are_kept():
    assert not _is_bibliography_page("")
    assert not _is_bibliography_page("References")


def test_clean_page_text_fixes_hyphenation_and_junk():
    raw = (
        "Federated learn-\ning improves distribu-\nted training.\n12\nWe  evaluate   robustness.\n"
    )
    cleaned = _clean_page_text(raw)
    assert "learning improves" in cleaned
    assert "distributed training" in cleaned
    assert "12" not in cleaned.split()
    assert "  " not in cleaned
    assert "We evaluate robustness." in cleaned


def test_clean_page_text_removes_lone_surrogate_but_keeps_valid_math_unicode():
    cleaned = _clean_page_text("valid 𝜖 malformed \ud835 symbol")
    assert "𝜖" in cleaned
    assert not any(0xD800 <= ord(character) <= 0xDFFF for character in cleaned)


def test_drop_repeated_lines_removes_running_headers():
    pages = [
        "Journal of Federated Systems\n\nSome unique content one.\n3",
        "Journal of Federated Systems\n\nDifferent content two.\n4",
        "Journal of Federated Systems\n\nMore content three.\n5",
        "Journal of Federated Systems\n\nFinal content four.\n6",
    ]
    cleaned = _drop_repeated_lines(pages)
    joined = "\n".join(cleaned)
    assert "Journal of Federated Systems" not in joined
    assert "unique content one" in joined
    assert "Final content four" in joined


def test_drop_repeated_lines_keeps_short_documents():
    pages = ["Header text\ncontent A", "Header text\ncontent B"]
    assert _drop_repeated_lines(pages) == pages


def test_detect_section_reads_headings():
    assert _detect_section("6 Conclusion and Future Work\nWe plan to extend") == "future_work"
    assert _detect_section("5. Conclusion\nWe summarized") == "conclusion"
    assert _detect_section("4 Experiments\nWe evaluate on NSL-KDD") == "results"
    assert _detect_section("3 Proposed Method\nOur framework") == "method"
    assert _detect_section("2 Related Work\nPrior studies") == "related_work"


def test_detect_section_defaults_to_body_for_plain_prose():
    text = (
        "The federated client trains locally and uploads gradients to the aggregation "
        "server, which then averages them before broadcasting the global model again."
    )
    assert _detect_section(text) == "body"
