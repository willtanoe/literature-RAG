from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from literature_rag.papers import DownloadedPaper
from literature_rag.resilience import atomic_write_json
from literature_rag.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_CHUNKS,
    MAX_PDF_PAGES,
)


def load_and_split_papers(downloaded: list[DownloadedPaper]) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Any] = []
    for path, paper in downloaded:
        print(f"Embedding -> loading {path.name}")
        try:
            pages = PyPDFLoader(str(path)).load()
            if len(pages) > MAX_PDF_PAGES:
                raise ValueError(f"PDF exceeds {MAX_PDF_PAGES} pages")
            for page in pages:
                page.metadata.update(
                    {
                        "title": paper.title,
                        "authors": ", ".join(paper.authors),
                        "arxiv_id": paper.arxiv_id,
                        "url": paper.entry_id,
                        "doi": paper.doi or "",
                        "venue": paper.venue or "",
                        "year": paper.year or "",
                        "citation_count": paper.citation_count or 0,
                        "is_open_access": paper.is_open_access,
                        "paper_id": _paper_id(paper.arxiv_id),
                    }
                )
            paper_chunks = splitter.split_documents(pages)
            for index, chunk in enumerate(paper_chunks, start=1):
                page_index = chunk.metadata.get("page")
                page_number = page_index + 1 if isinstance(page_index, int) else 0
                chunk.metadata["evidence_id"] = (
                    f"{_paper_id(paper.arxiv_id)}-p{page_number}-c{index}"
                )
            # Enforce budget BEFORE extending to prevent overflow
            projected = len(chunks) + len(paper_chunks)
            if projected > MAX_CHUNKS:
                excess = projected - MAX_CHUNKS
                dropped = paper_chunks[-excess:] if excess <= len(paper_chunks) else paper_chunks
                paper_chunks = paper_chunks[:-excess] if excess <= len(paper_chunks) else []
                print(f"Embedding -> truncated {len(dropped)} chunk(s) to stay within budget")
            chunks.extend(paper_chunks)
        except Exception as exc:
            print(f"Embedding -> skipped {path.name}: {exc}")
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise RuntimeError("No readable text was extracted from the downloaded PDFs.")
    if len(chunks) > MAX_CHUNKS:
        raise RuntimeError(f"Corpus exceeds {MAX_CHUNKS} chunks after truncation")
    print(f"Embedding -> created {len(chunks)} chunk(s)")
    return chunks


def build_vector_store(chunks: list[Any]) -> FAISS:
    print(f"Embedding -> encoding locally with {EMBEDDING_MODEL}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError(f"Embedding or FAISS indexing failed: {exc}") from exc
    print("Embedding -> FAISS index ready")
    return vector_store


def load_or_build_vector_store(downloaded: list[DownloadedPaper], index_dir: Path) -> FAISS:
    fingerprint = _corpus_fingerprint(downloaded)
    manifest_path = index_dir / "manifest.json"
    embeddings = _embeddings()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and _index_integrity(index_dir, manifest):
            vector_store = _load_safe_index(index_dir, embeddings)
            print(f"Embedding -> resumed persisted FAISS index in {index_dir}")
            return vector_store
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass

    chunks = load_and_split_papers(downloaded)
    print(f"Embedding -> encoding locally with {EMBEDDING_MODEL}")
    temporary: Path | None = None
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".faiss-", dir=index_dir.parent))
        _save_safe_index(vector_store, temporary)
        files = {name: _hash_file(temporary / name) for name in ("index.faiss", "documents.json")}
        atomic_write_json(temporary / "manifest.json", {"fingerprint": fingerprint, "files": files})
        backup = index_dir.with_name(f".{index_dir.name}.old")
        if backup.exists():
            shutil.rmtree(backup)
        if index_dir.exists():
            index_dir.replace(backup)
        temporary.replace(index_dir)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception as exc:
        raise RuntimeError(f"Embedding or FAISS indexing failed: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    print(f"Embedding -> persisted FAISS index in {index_dir}")
    return vector_store


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _corpus_fingerprint(downloaded: list[DownloadedPaper]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{EMBEDDING_MODEL}:{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode())
    for path, paper in sorted(downloaded, key=lambda item: item[1].arxiv_id):
        digest.update(paper.arxiv_id.encode("utf-8"))
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _paper_id(identity: str) -> str:
    return f"P{hashlib.sha256(identity.lower().encode()).hexdigest()[:10]}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_integrity(index_dir: Path, manifest: dict) -> bool:
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    try:
        return all(_hash_file(index_dir / name) == expected for name, expected in files.items())
    except OSError:
        return False


def _save_safe_index(vector_store: FAISS, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(vector_store.index, str(directory / "index.faiss"))
    documents = []
    for position, document_id in sorted(vector_store.index_to_docstore_id.items()):
        document = vector_store.docstore.search(document_id)
        if not isinstance(document, Document):
            raise RuntimeError(f"FAISS document {document_id} is missing")
        documents.append(
            {
                "position": position,
                "id": document_id,
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
        )
    atomic_write_json(directory / "documents.json", documents)


def _load_safe_index(index_dir: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
    index = faiss.read_index(str(index_dir / "index.faiss"))
    records = json.loads((index_dir / "documents.json").read_text(encoding="utf-8"))
    documents = {
        record["id"]: Document(page_content=record["page_content"], metadata=record["metadata"])
        for record in records
    }
    mapping = {int(record["position"]): record["id"] for record in records}
    if index.ntotal != len(mapping):
        raise ValueError("FAISS index and document mapping sizes differ")
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(documents),
        index_to_docstore_id=mapping,
    )
