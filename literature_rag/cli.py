"""CLI entry point for agentic literature review."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from literature_rag.__log__ import get_logger, setup_logging
from literature_rag.config import check_endpoint, choose_llm, get_semantic_scholar_key
from literature_rag.settings import DEFAULT_RETRIEVAL_K, DEFAULT_TOP_N, MAX_TARGET_PAPERS

logger = get_logger(__name__)


def run_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Load and invoke the pipeline lazily so CLI parsing remains usable standalone."""
    from literature_rag.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


YEAR_RANGE_PATTERN = re.compile(r"(\d{4})(?:\s*-\s*(\d{4})?)?")
MAX_TOP_N = MAX_TARGET_PAPERS


def _split_topics(raw: str) -> list[str]:
    topics: list[str] = []
    current: list[str] = []
    in_quotes = False
    for char in raw.replace(";", ","):
        if char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            topics.append("".join(current))
            current = []
        else:
            current.append(char)
    topics.append("".join(current))
    return [topic.strip() for topic in topics if topic.strip()]


def _parse_year_range(raw: str) -> tuple[int | None, int | None]:
    value = raw.strip()
    if not value:
        return None, None
    match = YEAR_RANGE_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid year range {raw!r}. Use 2023, 2020-2025, or 2022-.")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    return (min(start, end), max(start, end))


def _parse_top_n(raw: str) -> int:
    value = raw.strip()
    if not value:
        return DEFAULT_TOP_N
    if not value.isdigit() or not 1 <= int(value) <= MAX_TOP_N:
        raise ValueError(f"Paper count must be a number between 1 and {MAX_TOP_N}.")
    return int(value)


def _confirm_plan(plan: Any) -> bool:
    print(f"\nSearch plan ({plan.source}) for {plan.target} target paper(s):")
    for index, planned in enumerate(plan.queries, start=1):
        print(f"  {index}. [{planned.tier}] {planned.query}  -- {planned.intent}")
    try:
        answer = input("Run this search plan? [Y/n]: ").strip().lower()
    except EOFError:
        return True
    return answer in {"", "y", "yes"}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="literature-rag",
        description=(
            "Agentic Literature Review and RAG - Build evidence-grounded reviews from arXiv papers"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--interactive", action="store_true", default=None)
    parser.add_argument("--topic", type=str)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--local-pdf", type=str, metavar="PATH")
    parser.add_argument("--dois", type=str, metavar="DOIS")
    parser.add_argument("--unpaywall-email", type=str, metavar="EMAIL")
    parser.add_argument("--s2-api-key", type=str, default=None, help="Semantic Scholar API key")
    parser.add_argument("--year-range", type=str, default="", metavar="YYYY[-YYYY]")
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, metavar="N")
    parser.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K, metavar="K")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--no-manual-recovery", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--log-file", type=str, metavar="FILE")
    return parser.parse_args(args)


def configure_logging(args: argparse.Namespace) -> None:
    level = (
        "ERROR" if args.quiet and not args.log_file else ("DEBUG" if args.verbose >= 2 else "INFO")
    )
    setup_logging(level, Path(args.log_file) if args.log_file else None)


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    configure_logging(parsed)
    is_interactive = parsed.interactive is None and (
        parsed.topic is None or parsed.objective is None
    )
    if parsed.interactive is True:
        is_interactive = True
    if is_interactive:
        run_interactive_mode(parsed)
    else:
        run_non_interactive_mode(parsed)


def _year_args(parsed: argparse.Namespace) -> tuple[int | None, int | None]:
    if parsed.year_range:
        if parsed.year_min is not None or parsed.year_max is not None:
            raise ValueError("Use --year-range or --year-min/--year-max, not both.")
        return _parse_year_range(parsed.year_range)
    return parsed.year_min, parsed.year_max


def _print_result(result: Any) -> tuple[str, Any | None]:
    if isinstance(result, tuple):
        return result[0], result[1] if len(result) > 1 else None
    return result, None


def run_interactive_mode(parsed: argparse.Namespace) -> None:
    print("\nAgentic Literature Review & RAG\n" + "=" * 34)
    try:
        llm_config = choose_llm()
        check_endpoint(llm_config)
        default_s2 = get_semantic_scholar_key()
        topics = _split_topics(input("\nResearch topic(s), comma-separated: "))
        if not topics:
            raise ValueError("At least one research topic is required.")
        objective = input("Specific analysis objective: ").strip()
        if len(objective) < 10:
            raise ValueError(
                "The analysis objective is required and must describe what the review "
                "should deliver."
            )
        top_n = _parse_top_n(input(f"Target paper count [{DEFAULT_TOP_N}, max {MAX_TOP_N}]: "))
        parsed.top_n = top_n
        parsed.retrieval_k = max(DEFAULT_RETRIEVAL_K, min(top_n * 2, 64))
        parsed.year_range = input("Year range filter, e.g. 2020-2025 [optional]: ").strip()
        parsed.local_pdf = input("Local PDF folder [optional]: ").strip()
        parsed.dois = input("DOIs, comma-separated [optional]: ").strip()
        parsed.unpaywall_email = input("Email for Unpaywall OA lookup [optional]: ").strip()
        parsed.s2_api_key = default_s2
        endpoint_host = llm_config.base_url.split("//", 1)[-1].split("/", 1)[0]
        if input(
            f"PDF evidence will be sent to {endpoint_host}. Continue? [y/N]: "
        ).strip().lower() not in {"y", "yes"}:
            logger.warning("Pipeline cancelled by user")
            return
        failed: list[str] = []
        for topic in topics:
            try:
                report, store = _print_result(
                    _run_pipeline_with_args(topic, objective, llm_config, parsed, confirm_plan=True)
                )
                print("\n" + report)
                if store is not None:
                    _chat_loop(store, llm_config)
            except ValueError as exc:
                failed.append(topic)
                print(f"\nPipeline failed for {topic!r} -> {exc}")
        if len(topics) > 1:
            print(f"\nCompleted {len(topics) - len(failed)}/{len(topics)} review(s)")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled by user")
    except (ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed -> {exc}")


def run_non_interactive_mode(parsed: argparse.Namespace) -> None:
    if not parsed.topic or not parsed.objective:
        logger.error("--topic and --objective are required for non-interactive mode")
        sys.exit(1)
    try:
        llm_config = choose_llm()
        if not parsed.yes and input("Continue? [y/N]: ").strip().lower() not in {"y", "yes"}:
            logger.warning("Pipeline cancelled by user")
            return
        report, _ = _print_result(
            _run_pipeline_with_args(parsed.topic, parsed.objective, llm_config, parsed)
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
    confirm_plan: bool = False,
) -> Any:
    year_min, year_max = _year_args(args)
    return run_pipeline(
        topic=topic,
        analysis_question=analysis_question,
        llm_config=llm_config,
        top_n=args.top_n,
        retrieval_k=args.retrieval_k,
        local_pdf_dir=Path(args.local_pdf).expanduser() if args.local_pdf else None,
        dois=[v.strip() for v in args.dois.split(",") if v.strip()] if args.dois else [],
        unpaywall_email=args.unpaywall_email or "",
        year_min=year_min,
        year_max=year_max,
        s2_api_key=args.s2_api_key or "",
        plan_confirm=_confirm_plan if confirm_plan else None,
    )


def _chat_loop(vector_store: Any, llm_config: Any) -> None:
    print(
        "\nYou can now ask follow-up questions about these papers.\n"
        "Press Enter on an empty line to finish and continue.\n"
    )
    while True:
        try:
            question = input("Question: ").strip()
        except EOFError:
            break
        if not question:
            break
        try:
            from literature_rag.analysis import answer_question

            print("\n" + answer_question(vector_store, question, llm_config, DEFAULT_RETRIEVAL_K))
        except (ValueError, RuntimeError) as exc:
            print(f"Follow-up failed -> {exc}")
        print()


if __name__ == "__main__":
    main()
