from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from literature_rag.__log__ import get_logger
from literature_rag.config import LLMConfig
from literature_rag.papers import (
    Paper,
    load_search_cache,
    save_search_cache,
    search_arxiv,
)

logger = get_logger(__name__)

SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan academic database searches. Return one concise arXiv search "
            "phrase only, without quotes, labels, Boolean syntax, or explanation. Target "
            "a methodology, benchmark, limitation, or competing approach missing from "
            "the current results. Do not repeat a previous query.",
        ),
        (
            "human",
            "Topic: {topic}\nObjective: {objective}\nPrevious queries: {queries}\n"
            "Current paper titles:\n{titles}",
        ),
    ]
)


def iterative_search(
    topic: str,
    objective: str,
    llm_config: LLMConfig,
    max_results: int,
    cache_path: Path | None,
    max_rounds: int = 3,
) -> list[Paper]:
    cache_identity = hashlib.sha256(
        f"{objective}|{max_rounds}|{llm_config.base_url}|{llm_config.model_name}".encode()
    ).hexdigest()
    cached = load_search_cache(cache_path, topic, max_results, cache_identity)
    if cached is not None:
        logger.info(f"Resumed {len(cached)} cached paper(s)")
        return cached

    llm = ChatOpenAI(
        base_url=llm_config.base_url,
        api_key=SecretStr(llm_config.api_key),
        model=llm_config.model_name,
        temperature=0.1,
    )
    planner = SEARCH_PROMPT | llm | StrOutputParser()
    per_round = max(2, (max_results + max_rounds - 1) // max_rounds)
    queries: list[str] = []
    papers: list[Paper] = []
    seen: set[str] = set()

    for round_number in range(1, max_rounds + 1):
        if round_number == 1:
            query = topic
        else:
            logger.info(f"Planning coverage round {round_number}/{max_rounds}")
            try:
                query = planner.invoke(
                    {
                        "topic": topic,
                        "objective": objective,
                        "queries": "; ".join(queries),
                        "titles": "\n".join(f"- {paper.title}" for paper in papers),
                    }
                ).strip()
            except Exception as exc:
                logger.warning(f"Planner unavailable: {exc}")
                break
        if not query or query.lower() in {value.lower() for value in queries}:
            logger.debug("No novel query")
            break
        queries.append(query)
        logger.info(f"Round {round_number}: {query}")
        try:
            candidates = search_arxiv(query, per_round)
        except RuntimeError as exc:
            logger.warning(f"Query produced no usable result: {exc}")
            continue
        added = 0
        for paper in candidates:
            paper_identity = (paper.doi or paper.arxiv_id).lower()
            if paper_identity not in seen:
                seen.add(paper_identity)
                papers.append(paper)
                added += 1
        logger.info(f"Added {added} novel paper(s)")
        if len(papers) >= max_results:
            logger.info("Coverage target reached")
            break
        if added == 0:
            logger.debug("No novel papers")
            break

    if not papers:
        raise RuntimeError("Search agent found no papers.")
    papers = papers[:max_results]
    save_search_cache(cache_path, topic, max_results, papers, cache_identity)
    logger.info(f"Selected {len(papers)} unique paper(s)")
    return papers
