from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from literature_rag.ingestion import _load_safe_index, _save_safe_index


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
