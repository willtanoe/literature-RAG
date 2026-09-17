from __future__ import annotations

from pathlib import Path

from literature_rag.__log__ import get_logger
from literature_rag.analysis import generate_analysis
from literature_rag.config import LLMConfig
from literature_rag.export import export_review
from literature_rag.ingestion import load_or_build_vector_store
from literature_rag.papers import (
    deduplicate_downloads,
    download_papers,
    import_local_pdfs,
    papers_from_dois,
    recover_manual_downloads,
)
from literature_rag.resilience import atomic_write_text, redact_secrets, workspace_lock
from literature_rag.scholarly import enrich_papers
from literature_rag.search_agent import iterative_search
from literature_rag.settings import DOWNLOAD_DIR
from literature_rag.workspace import ProjectWorkspace, get_workspace

logger = get_logger(__name__)


def run_pipeline(
    topic: str,
    analysis_question: str,
    llm_config: LLMConfig,
    top_n: int = 5,
    retrieval_k: int = 12,
    download_dir: Path = DOWNLOAD_DIR,
    local_pdf_dir: Path | None = None,
    dois: list[str] | None = None,
    unpaywall_email: str = "",
) -> str:
    workspace = get_workspace(topic) if download_dir == DOWNLOAD_DIR else None
    paper_dir = workspace.papers if workspace else download_dir
    logger.info(f"Project -> {workspace.root}" if workspace else f"Project -> {paper_dir}")
    lock_root = workspace.root if workspace else paper_dir
    output_dir = workspace.output if workspace else paper_dir / "output"
    with workspace_lock(lock_root):
        try:
            report = _run_locked_pipeline(
                topic,
                analysis_question,
                llm_config,
                top_n,
                retrieval_k,
                paper_dir,
                local_pdf_dir,
                dois or [],
                unpaywall_email,
                workspace,
            )
        except Exception as exc:
            atomic_write_text(
                output_dir / "failure.txt",
                redact_secrets(exc, (llm_config.api_key,)) + "\n",
            )
            raise
        (output_dir / "failure.txt").unlink(missing_ok=True)
        return report


def _run_locked_pipeline(
    topic: str,
    analysis_question: str,
    llm_config: LLMConfig,
    top_n: int,
    retrieval_k: int,
    paper_dir: Path,
    local_pdf_dir: Path | None,
    dois: list[str],
    unpaywall_email: str,
    workspace: ProjectWorkspace | None,
) -> str:
    cache_path = workspace.search_cache if workspace else None
    papers = iterative_search(
        topic,
        analysis_question,
        llm_config,
        top_n,
        cache_path,
    )
    doi_papers = papers_from_dois(dois)
    metadata_cache = workspace.metadata_cache if workspace else None
    papers = enrich_papers(papers + doi_papers, metadata_cache, unpaywall_email)
    downloaded, failed = download_papers(papers, paper_dir)
    if local_pdf_dir is not None:
        downloaded.extend(import_local_pdfs(local_pdf_dir, paper_dir))
    downloaded = recover_manual_downloads(downloaded, failed)
    downloaded = deduplicate_downloads(downloaded)
    index_dir = workspace.index if workspace else paper_dir / ".faiss_index"
    vector_store = load_or_build_vector_store(downloaded, index_dir)
    output_dir = workspace.output if workspace else paper_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = generate_analysis(
        vector_store=vector_store,
        topic=topic,
        analysis_question=analysis_question,
        llm_config=llm_config,
        retrieval_k=retrieval_k,
        draft_path=output_dir / "draft.md",
    )
    export_review(
        output_dir,
        report,
        topic,
        analysis_question,
        downloaded,
        llm_config,
    )
    return report
