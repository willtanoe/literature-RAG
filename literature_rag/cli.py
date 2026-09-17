import re
from pathlib import Path

from literature_rag.config import check_endpoint, choose_llm, get_semantic_scholar_key
from literature_rag.search_agent import SearchPlan
from literature_rag.settings import DEFAULT_RETRIEVAL_K, DEFAULT_TOP_N, MAX_TARGET_PAPERS

YEAR_RANGE_PATTERN = re.compile(r"(\d{4})(?:\s*-\s*(\d{4})?)?")
MAX_TOP_N = MAX_TARGET_PAPERS


def _split_topics(raw: str) -> list[str]:
    topics: list[str] = []
    current: list[str] = []
    in_quotes = False
    for char in raw.replace(";", ","):
        if char == '"':
            in_quotes = not in_quotes
            continue
        if char == "," and not in_quotes:
            topics.append("".join(current))
            current = []
            continue
        current.append(char)
    topics.append("".join(current))
    return [topic.strip() for topic in topics if topic.strip()]


def _parse_year_range(raw: str) -> tuple[int | None, int | None]:
    value = raw.strip()
    if not value:
        return None, None
    match = YEAR_RANGE_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(
            f"Invalid year range {raw!r}. Use a year such as 2023, a range such as "
            "2020-2025, or an open range such as 2022-."
        )
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    if start > end:
        start, end = end, start
    return start, end


def _parse_top_n(raw: str) -> int:
    value = raw.strip()
    if not value:
        return DEFAULT_TOP_N
    if not value.isdigit() or not 1 <= int(value) <= MAX_TOP_N:
        raise ValueError(f"Paper count must be a number between 1 and {MAX_TOP_N}.")
    return int(value)


def _confirm_plan(plan: SearchPlan) -> bool:
    print(f"\nSearch plan ({plan.source}) for {plan.target} target paper(s):")
    for index, planned in enumerate(plan.queries, start=1):
        print(f"  {index}. [{planned.tier}] {planned.query}  -- {planned.intent}")
    try:
        answer = input("Run this search plan? [Y/n]: ").strip().lower()
    except EOFError:
        return True
    return answer in {"", "y", "yes"}


def main() -> None:
    print("\nAgentic Literature Review & RAG")
    print("=" * 34)
    try:
        llm_config = choose_llm()
        check_endpoint(llm_config)
        s2_api_key = get_semantic_scholar_key()
        topics = _split_topics(input("\nResearch topic(s), comma-separated: "))
        if not topics:
            raise ValueError("At least one research topic is required.")
        analysis_question = input("Specific analysis objective: ").strip()
        if len(analysis_question) < 10:
            raise ValueError(
                "The analysis objective is required and must describe what the review "
                "should deliver, for example: build a matrix of methods, datasets, "
                "gaps, and future work."
            )
        top_n = _parse_top_n(input(f"Target paper count [{DEFAULT_TOP_N}, max {MAX_TOP_N}]: "))
        retrieval_k = max(DEFAULT_RETRIEVAL_K, min(top_n * 2, 64))
        year_min, year_max = _parse_year_range(
            input("Year range filter, e.g. 2020-2025 [optional]: ")
        )
        local_folder = input("Local PDF folder [optional]: ").strip()
        doi_values = input("DOIs, comma-separated [optional]: ").strip()
        unpaywall_email = input("Email for Unpaywall OA lookup [optional]: ").strip()
        endpoint_host = llm_config.base_url.split("//", 1)[-1].split("/", 1)[0]
        run_label = "run" if len(topics) == 1 else f"{len(topics)} runs"
        consent = (
            input(
                f"PDF evidence will be sent to {endpoint_host} for {run_label}. Continue? [y/N]: "
            )
            .strip()
            .lower()
        )
        if consent not in {"y", "yes"}:
            print("Pipeline cancelled -> no document content was sent")
            return
        print("Initializing pipeline components ...")
        from literature_rag.pipeline import PlanRejected, run_pipeline

        failed: list[str] = []
        for index, topic in enumerate(topics, start=1):
            if len(topics) > 1:
                print(f"\n=== Run {index}/{len(topics)}: {topic}")
            try:
                report, vector_store = run_pipeline(
                    topic=topic,
                    analysis_question=analysis_question,
                    llm_config=llm_config,
                    top_n=top_n,
                    retrieval_k=retrieval_k,
                    local_pdf_dir=(Path(local_folder).expanduser() if local_folder else None),
                    dois=[value.strip() for value in doi_values.split(",") if value.strip()],
                    unpaywall_email=unpaywall_email,
                    year_min=year_min,
                    year_max=year_max,
                    s2_api_key=s2_api_key,
                    plan_confirm=_confirm_plan,
                )
                print("\n" + report)
                _chat_loop(vector_store, llm_config)
            except PlanRejected as exc:
                failed.append(topic)
                print(f"\nRun cancelled for {topic!r} -> {exc}")
            except (ValueError, RuntimeError) as exc:
                failed.append(topic)
                print(f"\nPipeline failed for {topic!r} -> {exc}")
        if len(topics) > 1:
            completed = len(topics) - len(failed)
            print(f"\nCompleted {completed}/{len(topics)} review(s)")
            for topic in failed:
                print(f"Failed -> {topic}")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled by user")
    except (ValueError, RuntimeError) as exc:
        print(f"\nPipeline failed -> {exc}")


def _chat_loop(vector_store, llm_config) -> None:
    print("\nYou can now ask follow-up questions about these papers.")
    print("Press Enter on an empty line to finish and continue.\n")
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
