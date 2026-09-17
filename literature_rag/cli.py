"""CLI entry point for agentic literature review."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from literature_rag.__log__ import get_logger, setup_logging
from literature_rag.config import choose_llm
from literature_rag.pipeline import run_pipeline
from literature_rag.settings import DEFAULT_RETRIEVAL_K, DEFAULT_TOP_N

logger = get_logger(__name__)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:] if None).
    
    Returns:
        Parsed namespace with all CLI options.
    """
    parser = argparse.ArgumentParser(
        prog="literature-rag",
        description=(
            "Agentic Literature Review and RAG - Build evidence-grounded literature "
            "reviews from arXiv papers"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default when no arguments provided)
  literature-rag

  # Non-interactive mode
  literature-rag --topic "retrieval augmented generation" \\
                 --objective "compare evaluation metrics and identify gaps" \\
                 --dois 10.1145/example,10.1000/sample \\
                 --local-pdf ./my_papers \\
                 --top-n 10 \\
                 --yes
  
  # Verbose logging
  literature-rag --topic "graph neural networks" --verbose

  # Quiet mode (minimal output)
  literature-rag --topic "transformers" --quiet
""",
    )
    
    # Mode selection
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        default=None,
        help="Run in interactive mode (prompts for input)"
    )
    
    # Required arguments for non-interactive mode
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic and primary arXiv query (required for non-interactive mode)"
    )
    parser.add_argument(
        "--objective",
        type=str,
        help=(
            "Specific analysis objective for follow-up searches; "
            "required in non-interactive mode"
        )
    )
    
    # Optional inputs
    parser.add_argument(
        "--local-pdf",
        type=str,
        metavar="PATH",
        help="Directory containing additional PDF documents to include"
    )
    parser.add_argument(
        "--dois",
        type=str,
        metavar="DOIS",
        help="Comma-separated DOIs or DOI URLs to process"
    )
    parser.add_argument(
        "--unpaywall-email",
        type=str,
        metavar="EMAIL",
        help="Email address for Unpaywall OA lookup (optional but recommended)"
    )
    
    # Pipeline parameters
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        metavar="N",
        help=f"Target number of papers from arXiv search (default: {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        metavar="K",
        help=f"Number of chunks to retrieve for analysis (default: {DEFAULT_RETRIEVAL_K})"
    )
    
    # Behavior flags
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm prompts (non-interactive mode only)"
    )
    parser.add_argument(
        "--no-manual-recovery",
        action="store_true",
        help="Skip manual PDF recovery prompt if downloads fail"
    )
    
    # Logging configuration
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times, e.g., -vv)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-critical output"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="FILE",
        help="Append logs to specified file path"
    )
    
    return parser.parse_args(args)


def configure_logging(args: argparse.Namespace) -> None:
    """Configure logging based on CLI arguments."""
    level = "INFO"
    log_file = None
    
    if args.verbose >= 3 or args.verbose == 2:
        level = "DEBUG"
    elif args.verbose == 1:
        level = "INFO"
    
    if args.quiet and not args.log_file:
        level = "ERROR"
    elif args.log_file:
        log_file = Path(args.log_file)
    
    setup_logging(level, log_file)


def main(args: list[str] | None = None) -> None:
    """Main entry point for CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:] if None).
    """
    parsed = parse_args(args)
    configure_logging(parsed)
    
    # Determine mode
    is_interactive = (
        parsed.interactive is None and 
        (parsed.topic is None or parsed.objective is None)
    )
    
    if is_interactive:
        run_interactive_mode(parsed)
    else:
        run_non_interactive_mode(parsed)


def run_interactive_mode(parsed: argparse.Namespace) -> None:
    """Run in interactive mode with user prompts."""
    print("\nAgentic Literature Review & RAG")
    print("=" * 34)
    try:
        llm_config = choose_llm()
        topic = input("\nResearch topic: ").strip()
        analysis_question = input("Specific analysis objective: ").strip()
        input("Local PDF folder [optional]: ").strip()
        input("DOIs, comma-separated [optional]: ").strip()
        input("Email for Unpaywall OA lookup [optional]: ").strip()
        
        endpoint_host = llm_config.base_url.split("//", 1)[-1].split("/", 1)[0]
        consent = (
            input(f"\nPDF evidence will be sent to {endpoint_host}. Continue? [y/N]: ")
            .strip()
            .lower()
        )
        if consent not in {"y", "yes"}:
            logger.warning("Pipeline cancelled by user")
            return
        
        report = _run_pipeline_with_args(topic, analysis_question, llm_config, parsed)
        print("\n" + report)
    except (ValueError, RuntimeError) as exc:
        logger.error(f"Pipeline failed: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)


def run_non_interactive_mode(parsed: argparse.Namespace) -> None:
    """Run in non-interactive mode with pre-configured arguments."""
    if not parsed.topic or not parsed.objective:
        logger.error("--topic and --objective are required for non-interactive mode")
        sys.exit(1)
    
    try:
        # Use saved LLM config or allow user to add one interactively
        llm_config = choose_llm()
        
        # Auto-confirm consent in non-interactive mode
        if not parsed.yes:
            endpoint_host = (
                llm_config.base_url.split("//", 1)[-1].split("/", 1)[0]
            )
            logger.warning(
                f"\nPDF evidence will be sent to {endpoint_host}. "
                "Run again with --yes to skip this confirmation."
            )
            response = input("Continue? [y/N]: ").strip().lower()
            if response not in {"y", "yes"}:
                logger.warning("Pipeline cancelled by user")
                return
        
        report = _run_pipeline_with_args(
            parsed.topic,
            parsed.objective,
            llm_config,
            parsed
        )
        print("\n" + report)
    except (ValueError, RuntimeError) as exc:
        logger.error(f"Pipeline failed: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)


def _run_pipeline_with_args(
    topic: str,
    analysis_question: str,
    llm_config: Any,
    args: argparse.Namespace,
) -> str:
    """Execute the pipeline with given parameters."""
    local_pdf_dir = (
        Path(args.local_pdf).expanduser() if args.local_pdf else None
    )
    dois_list = (
        [value.strip() for value in args.dois.split(",") if value.strip()]
        if args.dois
        else []
    )
    
    return run_pipeline(
        topic=topic,
        analysis_question=analysis_question,
        llm_config=llm_config,
        top_n=args.top_n,
        retrieval_k=args.retrieval_k,
        local_pdf_dir=local_pdf_dir,
        dois=dois_list,
        unpaywall_email=args.unpaywall_email or "",
    )


if __name__ == "__main__":
    main()
