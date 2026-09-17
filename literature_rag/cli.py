from pathlib import Path

from literature_rag.__log__ import get_logger
from literature_rag.config import choose_llm
from literature_rag.pipeline import run_pipeline
from literature_rag.settings import DEFAULT_RETRIEVAL_K, DEFAULT_TOP_N

logger = get_logger(__name__)


def main() -> None:
    print("\nAgentic Literature Review & RAG")
    print("=" * 34)
    try:
        llm_config = choose_llm()
        topic = input("\nResearch topic: ").strip()
        analysis_question = input("Specific analysis objective: ").strip()
        local_folder = input("Local PDF folder [optional]: ").strip()
        doi_values = input("DOIs, comma-separated [optional]: ").strip()
        unpaywall_email = input("Email for Unpaywall OA lookup [optional]: ").strip()
        endpoint_host = llm_config.base_url.split("//", 1)[-1].split("/", 1)[0]
        consent = (
            input(f"PDF evidence will be sent to {endpoint_host}. Continue? [y/N]: ")
            .strip()
            .lower()
        )
        if consent not in {"y", "yes"}:
            logger.warning("Pipeline cancelled")
            return
        report = run_pipeline(
            topic=topic,
            analysis_question=analysis_question,
            llm_config=llm_config,
            top_n=DEFAULT_TOP_N,
            retrieval_k=DEFAULT_RETRIEVAL_K,
            local_pdf_dir=Path(local_folder).expanduser() if local_folder else None,
            dois=[value.strip() for value in doi_values.split(",") if value.strip()],
            unpaywall_email=unpaywall_email,
        )
        print("\n" + report)
    except (ValueError, RuntimeError) as exc:
        logger.error(f"Pipeline failed: {exc}")
