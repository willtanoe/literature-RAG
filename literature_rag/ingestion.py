from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from literature_rag.__log__ import get_logger
from literature_rag.papers import DownloadedPaper
from literature_rag.resilience import atomic_write_json
from literature_rag.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TIER,
    EMBEDDING_MODEL,
    MAX_CHUNKS,
    MAX_PDF_PAGES,
)

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)

REFERENCE_HEADING_PATTERN = re.compile(
    r"^\s*(references|bibliography|literature\s+cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
BRACKET_CITATION_PATTERN = re.compile(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/\S+")
ARXIV_ID_PATTERN = re.compile(r"arxiv:\d{4}\.\d{4,5}", re.IGNORECASE)
MIN_BRACKET_CITATIONS = 6
BIBLIOGRAPHY_DENSITY_THRESHOLD = 0.02
HYPHENATION_PATTERN = re.compile(r"([A-Za-z])-\s*\n\s*([a-z])")
PAGE_NUMBER_LINE_PATTERN = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
EXCESS_WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")
SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")
REPEATED_LINE_MAX_LENGTH = 80
REPEATED_LINE_PAGE_SHARE = 0.5

SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("future_work", re.compile(r"\b(future\s+work|future\s+research|future\s+direction)", re.I)),
    ("conclusion", re.compile(r"\bconclusion(s)?\b", re.I)),
    ("limitations", re.compile(r"\b(limitation(s)?|threats?\s+to\s+validity)\b", re.I)),
    ("results", re.compile(r"\b(results?|evaluation|experiment(s|al)?)\b", re.I)),
    ("method", re.compile(r"\b(method(s|ology)?|approach|proposed|design|framework)\b", re.I)),
    ("related_work", re.compile(r"\b(related\s+work|background)\b", re.I)),
    ("abstract", re.compile(r"\babstract\b", re.I)),
    ("introduction", re.compile(r"\bintroduction\b", re.I)),
)
FUTURE_WORK_TEXT_PATTERN = re.compile(
    r"(future\s+work|future\s+research|future\s+direction|left\s+for\s+future|"
    r"in\s+future\s+work|we\s+plan\s+to|remains?\s+open|open\s+problem|"
    r"further\s+(work|study|investigation)|limitation)",
    re.IGNORECASE,
)
METRIC_TEXT_PATTERN = re.compile(
    r"(\b\d{1,3}\.\d+\s*%|\b\d{1,3}\s*%|\bF1\b|\bAUC\b|\bAUPRC\b|\baccuracy\b|"
    r"\bprecision\b|\brecall\b|\bepsilon\b|\bε\b)",
    re.IGNORECASE,
)
HEADING_MAX_WORDS = 12


def _detect_section(text: str) -> str:
    for line in text.strip().splitlines()[:6]:
        candidate = line.strip().strip("#").strip()
        if not candidate or len(candidate.split()) > HEADING_MAX_WORDS:
            continue
        for name, pattern in SECTION_PATTERNS:
            if pattern.search(candidate):
                return name
    for name, pattern in SECTION_PATTERNS[:3]:
        if pattern.search(text[:400]):
            return name
    return "body"


def _clean_page_text(text: str) -> str:
    text = SURROGATE_PATTERN.sub(" ", text)
    text = HYPHENATION_PATTERN.sub(r"\1\2", text)
    text = PAGE_NUMBER_LINE_PATTERN.sub("", text)
    text = EXCESS_WHITESPACE_PATTERN.sub(" ", text)
    text = EXCESS_NEWLINES_PATTERN.sub("\n\n", text)
    return text.strip()


def _drop_repeated_lines(pages: list[str]) -> list[str]:
    if len(pages) < 3:
        return pages
    line_counts: Counter[str] = Counter()
    for text in pages:
        seen = {
            line.strip()
            for line in text.splitlines()
            if line.strip() and len(line.strip()) <= REPEATED_LINE_MAX_LENGTH
        }
        line_counts.update(seen)
    threshold = max(2, int(len(pages) * REPEATED_LINE_PAGE_SHARE))
    repeated = {line for line, count in line_counts.items() if count >= threshold}
    if not repeated:
        return pages
    cleaned = []
    for text in pages:
        kept = [line for line in text.splitlines() if line.strip() not in repeated]
        cleaned.append(EXCESS_NEWLINES_PATTERN.sub("\n\n", "\n".join(kept)).strip())
    return cleaned


def _is_bibliography_page(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 100:
        return False
    if REFERENCE_HEADING_PATTERN.search("\n".join(stripped.splitlines()[:3])):
        return True
    brackets = len(BRACKET_CITATION_PATTERN.findall(stripped))
    markers = (
        brackets + len(DOI_PATTERN.findall(stripped)) + len(ARXIV_ID_PATTERN.findall(stripped))
    )
    words = len(stripped.split())
    if words == 0:
        return False
    return brackets >= MIN_BRACKET_CITATIONS and markers / words >= BIBLIOGRAPHY_DENSITY_THRESHOLD


def load_and_split_papers(downloaded: list[DownloadedPaper]) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Any] = []
    for path, paper in downloaded:
        logger.debug(f"Loading PDF: {path.name}")
        try:
            pages = PyPDFLoader(str(path)).load()
            if len(pages) > MAX_PDF_PAGES:
                raise ValueError(f"PDF exceeds {MAX_PDF_PAGES} pages")
            cleaned_texts = _drop_repeated_lines(
                [_clean_page_text(page.page_content) for page in pages]
            )
            kept_pages = []
            skipped_reference_pages = 0
            for page, cleaned in zip(pages, cleaned_texts, strict=True):
                page.page_content = cleaned
                if _is_bibliography_page(page.page_content):
                    skipped_reference_pages += 1
                    continue
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
                        "tier": paper.tier or DEFAULT_TIER,
                        "paper_id": _paper_id(paper.arxiv_id),
                    }
                )
                kept_pages.append(page)
            if skipped_reference_pages:
                print(
                    f"Embedding -> skipped {skipped_reference_pages} "
                    f"reference page(s) in {path.name}"
                )
            paper_chunks = splitter.split_documents(kept_pages)
            current_section = "body"
            for index, chunk in enumerate(paper_chunks, start=1):
                page_index = chunk.metadata.get("page")
                page_number = page_index + 1 if isinstance(page_index, int) else 0
                chunk.metadata["evidence_id"] = (
                    f"{_paper_id(paper.arxiv_id)}-p{page_number}-c{index}"
                )
                detected = _detect_section(chunk.page_content)
                if detected != "body":
                    current_section = detected
                chunk.metadata["section"] = current_section
                chunk.metadata["chunk_index"] = index
                chunk.metadata["has_future_work"] = bool(
                    FUTURE_WORK_TEXT_PATTERN.search(chunk.page_content)
                )
                chunk.metadata["has_metrics"] = bool(METRIC_TEXT_PATTERN.search(chunk.page_content))
            projected = len(chunks) + len(paper_chunks)
            if projected > MAX_CHUNKS:
                excess = projected - MAX_CHUNKS
                paper_chunks = paper_chunks[:-excess] if excess < len(paper_chunks) else []
                logger.warning(f"Truncated {excess} chunk(s) to stay within budget")
            chunks.extend(paper_chunks)
        except Exception as exc:
            logger.warning(f"Skipping {path.name}: {exc}")
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise RuntimeError(
            "No readable text was extracted from the downloaded PDFs "
            "(all pages may be empty, scanned, or reference lists)."
        )
    if len(chunks) > MAX_CHUNKS:
        raise RuntimeError(f"Corpus exceeds {MAX_CHUNKS} chunks after truncation")
    logger.info(f"Created {len(chunks)} chunk(s)")
    return chunks


def build_vector_store(chunks: list[Any]) -> FAISS:
    logger.info(f"Encoding locally with {EMBEDDING_MODEL}")
    try:
        vector_store = FAISS.from_documents(chunks, _embeddings())
    except Exception as exc:
        raise RuntimeError(f"Embedding or FAISS indexing failed: {exc}") from exc
    logger.info("FAISS index ready")
    return vector_store


def load_or_build_vector_store(downloaded: list[DownloadedPaper], index_dir: Path) -> FAISS:
    fingerprint = _corpus_fingerprint(downloaded)
    manifest_path = index_dir / "manifest.json"
    embeddings = _embeddings()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and _index_integrity(index_dir, manifest):
            vector_store = _load_safe_index(index_dir, embeddings)
            logger.info(f"Resuming persisted FAISS index in {index_dir}")
            return vector_store
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass

    chunks = load_and_split_papers(downloaded)
    logger.info(f"Encoding locally with {EMBEDDING_MODEL}")
    temporary: Path | None = None
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".faiss-", dir=index_dir.parent))
        _save_safe_index(vector_store, temporary)
        logger.info("Saving FAISS index and document manifest")
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
    logger.info(f"Persisted FAISS index in {index_dir}")
    return vector_store


def _embeddings() -> HuggingFaceEmbeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        show_progress=True,
    )


def _corpus_fingerprint(downloaded: list[DownloadedPaper]) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{EMBEDDING_MODEL}:{CHUNK_SIZE}:{CHUNK_OVERLAP}:refs-filter-v1:text-clean-v2:sections-v1:tiers-v1".encode()
    )
    for path, paper in sorted(downloaded, key=lambda item: item[1].arxiv_id):
        digest.update(paper.arxiv_id.encode("utf-8"))
        digest.update((paper.tier or DEFAULT_TIER).encode("utf-8"))
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
