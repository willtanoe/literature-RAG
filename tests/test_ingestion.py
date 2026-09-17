from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from literature_rag.ingestion import _load_safe_index, _save_safe_index
from literature_rag.papers import Paper
from literature_rag.settings import MAX_CHUNKS


def test_max_chunks_enforced_strictly(tmp_path, monkeypatch):
    """BUG FIX: Verify that MAX_CHUNKS is enforced and cannot be exceeded."""
    from langchain_core.documents import Document
    
    fake_doc = Document(
        page_content="This is sample text that will be split into many chunks for testing.",
        metadata={"page": 1},
    )
    
    def mock_load(self):
        return [fake_doc]
    
    class ManyChunkSplitter:
        CHUNK_COUNT = MAX_CHUNKS + 50
        def split_documents(self, docs):
            result = []
            for _ in docs:
                for i in range(self.CHUNK_COUNT):
                    result.append(Document(page_content=f"chunk content {i}", metadata={"page": 1}))
            return result
    
    StubSplitter = ManyChunkSplitter()
    monkeypatch.setattr(
        "literature_rag.ingestion.RecursiveCharacterTextSplitter", lambda *_, **__: StubSplitter
    )
    monkeypatch.setattr("langchain_community.document_loaders.PyPDFLoader.load", mock_load)
    
    paths = [tmp_path / "doc_0.pdf"]
    paths[0].write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n%%EOF")
    papers = [(paths[0], Paper("id0", "Title", (), "http://ex/i", ""))]
    
    from literature_rag.ingestion import load_and_split_papers
    result = load_and_split_papers(papers)
    assert len(result) <= MAX_CHUNKS, f"MAX_CHUNKS not enforced: got {len(result)} chunks"


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
