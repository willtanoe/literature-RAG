from collections.abc import Callable
from pathlib import Path
from typing import Any

from literature_rag.analysis import AnalysisResult, classify_relevance, generate_analysis
from literature_rag.config import LLMConfig
from literature_rag.export import export_review
from literature_rag.ingestion import load_or_build_vector_store
from literature_rag.papers import (
    Paper,
    deduplicate_downloads,
    download_papers,
    hydrate_abstracts,
    import_local_pdfs,
    import_workspace_pdfs,
    organize_papers_by_tier,
    papers_from_dois,
    recover_manual_downloads,
)
from literature_rag.resilience import (
    atomic_write_text,
    redact_secrets,
    tee_stdout,
    workspace_lock,
)
from literature_rag.scholarly import enrich_papers
from literature_rag.search_agent import SearchPlan, iterative_search, plan_as_json, plan_searches
from literature_rag.settings import DOWNLOAD_DIR
from literature_rag.workspace import ProjectWorkspace, archive_previous_output, get_workspace

PlanConfirm = Callable[[SearchPlan], bool]


class PlanRejected(RuntimeError):
    pass


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
    year_min: int | None = None,
    year_max: int | None = None,
    s2_api_key: str = "",
    plan_confirm: PlanConfirm | None = None,
) -> tuple[str, Any]:
    workspace = get_workspace(topic) if download_dir == DOWNLOAD_DIR else None
    paper_dir = workspace.papers if workspace else download_dir
    print(f"Project -> {workspace.root}" if workspace else f"Project -> {paper_dir}")
    lock_root = workspace.root if workspace else paper_dir
    output_dir = workspace.output if workspace else paper_dir / "output"
    with workspace_lock(lock_root), tee_stdout(output_dir / "run.log"):
        try:
            report, vector_store = _run_locked_pipeline(
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
                year_min,
                year_max,
                s2_api_key,
                plan_confirm,
            )
        except PlanRejected:
            raise
        except Exception as exc:
            atomic_write_text(
                output_dir / "failure.txt",
                redact_secrets(exc, (llm_config.api_key,)) + "\n",
            )
            raise
        (output_dir / "failure.txt").unlink(missing_ok=True)
        return report, vector_store


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
    year_min: int | None = None,
    year_max: int | None = None,
    s2_api_key: str = "",
    plan_confirm: PlanConfirm | None = None,
) -> tuple[str, Any]:
    cache_path = workspace.search_cache if workspace else None
    plan_cache = workspace.plan_cache if workspace else None
    plan = plan_searches(topic, analysis_question, llm_config, top_n, plan_cache)
    if plan_confirm is not None and not plan_confirm(plan):
        raise PlanRejected("Search plan rejected before any paper was downloaded.")
    output_dir = workspace.output if workspace else paper_dir / "output"
    archive_previous_output(output_dir)
    papers = iterative_search(
        topic,
        analysis_question,
        llm_config,
        top_n,
        cache_path,
        plan=plan,
        year_min=year_min,
        year_max=year_max,
    )
    doi_papers = papers_from_dois(dois)
    metadata_cache = workspace.metadata_cache if workspace else None
    papers = enrich_papers(papers + doi_papers, metadata_cache, unpaywall_email, s2_api_key)
    papers = hydrate_abstracts(papers)
    papers = classify_relevance(
        papers,
        topic,
        analysis_question,
        llm_config,
        workspace.classification_cache if workspace else None,
    )
    papers = _drop_excluded(papers)
    downloaded, failed = download_papers(papers, paper_dir)
    if local_pdf_dir is not None:
        downloaded.extend(import_local_pdfs(local_pdf_dir, paper_dir))
    downloaded = recover_manual_downloads(downloaded, failed)
    downloaded = deduplicate_downloads(downloaded)
    downloaded = organize_papers_by_tier(downloaded, paper_dir)
    downloaded.extend(import_workspace_pdfs(paper_dir, downloaded))
    downloaded = deduplicate_downloads(downloaded)
    index_dir = workspace.index if workspace else paper_dir / ".faiss_index"
    vector_store = load_or_build_vector_store(downloaded, index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis: AnalysisResult = generate_analysis(
        vector_store=vector_store,
        topic=topic,
        analysis_question=analysis_question,
        llm_config=llm_config,
        retrieval_k=retrieval_k,
        draft_path=output_dir / "draft.md",
    )
    export_review(
        output_dir,
        analysis.report,
        topic,
        analysis_question,
        downloaded,
        llm_config,
        sota_results=analysis.sota_results,
        paper_summaries=analysis.paper_summaries,
        thesis_proposals=analysis.thesis_proposals,
        gap_register=analysis.gap_register,
        thesis_candidates=analysis.thesis_candidates,
        search_plan=plan_as_json(plan),
    )
    return analysis.report, vector_store


def _drop_excluded(papers: list[Paper]) -> list[Paper]:
    kept = [paper for paper in papers if paper.tier != "excluded"]
    dropped = len(papers) - len(kept)
    if dropped:
        print(f"Triage -> dropped {dropped} excluded paper(s) before downloading")
    if not kept:
        raise RuntimeError(
            "Every candidate paper was classified as excluded; broaden the topic or objective."
        )
    return kept
