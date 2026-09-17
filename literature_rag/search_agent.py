from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from literature_rag.config import LLMConfig
from literature_rag.papers import (
    QUERY_STOPWORDS,
    Paper,
    load_search_cache,
    save_search_cache,
    search_arxiv,
)
from literature_rag.resilience import atomic_write_json
from literature_rag.settings import (
    LLM_REQUEST_TIMEOUT_SECONDS,
    MAX_PLAN_QUERIES,
    MAX_PLAN_QUERY_TERMS,
)

SEARCH_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan a literature search before any papers are retrieved. Return ONLY "
            "a JSON array, with no markdown fences and no commentary. Produce at most "
            "{max_queries} elements, each an object with these exact keys: "
            '"query" (2 to {max_terms} plain keywords, no quotes, no Boolean operators, '
            'no field prefixes) and "intent" (one short phrase naming the facet it '
            'covers) and "tier" (either "core" for the central topic or "related" for a '
            "neighboring method, benchmark, limitation, or competing approach). Cover "
            "the central topic first, then distinct facets: methods, datasets and "
            "benchmarks, evaluation and metrics, known limitations, competing "
            "approaches, and application domains. Keep each query broad enough to "
            "return results on its own; never repeat the same keyword set.",
        ),
        (
            "human",
            "Topic: {topic}\nObjective: {objective}\nTarget corpus size: {target} papers",
        ),
    ]
)

MAX_SEARCH_ROUNDS = MAX_PLAN_QUERIES + 6
MAX_CONSECUTIVE_EMPTY_ROUNDS = 2
SEARCH_STRATEGY_VERSION = "plan-v3"
MIN_RELAXED_TERMS = 3
MIN_QUERY_QUOTA = 10
MAX_QUERY_QUOTA = 100
PLAN_CACHE_VERSION = 1
QUERY_NOISE = frozenset(["arxiv", "paper", "papers", "query", "search", "keywords", "recent"])


@dataclass(frozen=True)
class PlannedQuery:
    query: str
    intent: str
    tier: str


@dataclass(frozen=True)
class SearchPlan:
    topic: str
    objective: str
    target: int
    queries: tuple[PlannedQuery, ...]
    source: str

    @property
    def signature(self) -> str:
        joined = "|".join(f"{item.query}::{item.tier}" for item in self.queries)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _significant_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [
        token
        for token in dict.fromkeys(tokens)
        if len(token) >= 3 and token not in QUERY_STOPWORDS and token not in QUERY_NOISE
    ]


def _normalize_query(raw: str, max_terms: int = MAX_PLAN_QUERY_TERMS) -> str:
    text = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", str(raw).strip())
    text = re.sub(r"(?i)^(?:query|search|topic)\s*[:=]\s*", "", text)
    text = re.sub(r"\b(?:all|ti|abs|au|cat)\s*:\s*", " ", text)
    text = re.sub(r"\b(?:AND|OR|NOT)\b", " ", text)
    tokens = _significant_tokens(text)
    return " ".join(tokens[:max_terms])


def _query_signature(query: str) -> frozenset[str]:
    return frozenset(_significant_tokens(query))


def _relaxed_query(query: str, level: int) -> str:
    tokens = _significant_tokens(query)
    keep = max(MIN_RELAXED_TERMS, len(tokens) - 2 * level)
    if keep >= len(tokens):
        return ""
    return " ".join(tokens[:keep])


def _per_query_quota(max_results: int, query_count: int) -> int:
    base = (max_results + query_count - 1) // max(query_count, 1)
    return min(max(base * 2, MIN_QUERY_QUOTA), MAX_QUERY_QUOTA)


def _parse_plan(raw: str) -> list[PlannedQuery]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    items: list[dict[str, str]] = []
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    items.append({str(k): str(v) for k, v in item.items()})
                elif isinstance(item, str):
                    items.append({"query": item})
    if not items:
        items = [{"query": line} for line in text.splitlines() if line.strip()]
    planned: list[PlannedQuery] = []
    for item in items:
        query = _normalize_query(item.get("query", ""))
        if not query:
            continue
        tier = item.get("tier", "").strip().lower()
        planned.append(
            PlannedQuery(
                query=query,
                intent=re.sub(r"\s+", " ", item.get("intent", "")).strip() or "unspecified facet",
                tier="core" if tier == "core" else "related",
            )
        )
    return planned


def _deduplicate_plan(queries: list[PlannedQuery]) -> tuple[PlannedQuery, ...]:
    unique: list[PlannedQuery] = []
    seen: list[frozenset[str]] = []
    for item in queries:
        signature = _query_signature(item.query)
        if not signature or signature in seen:
            continue
        seen.append(signature)
        unique.append(item)
        if len(unique) >= MAX_PLAN_QUERIES:
            break
    return tuple(unique)


def _fallback_plan(topic: str, objective: str, target: int, reason: str) -> SearchPlan:
    print(f"Search Plan -> using deterministic fallback plan: {reason}")
    queries = [PlannedQuery(query=topic.strip(), intent="core topic", tier="core")]
    for level in range(1, MAX_PLAN_QUERIES):
        relaxed = _relaxed_query(topic, level)
        if not relaxed:
            break
        queries.append(
            PlannedQuery(query=relaxed, intent=f"relaxation level {level}", tier="related")
        )
    return SearchPlan(
        topic=topic,
        objective=objective,
        target=target,
        queries=_deduplicate_plan(queries),
        source="fallback",
    )


def plan_searches(
    topic: str,
    objective: str,
    llm_config: LLMConfig,
    target: int,
    cache_path: Path | None = None,
) -> SearchPlan:
    cached = _load_plan(cache_path, topic, objective, target)
    if cached is not None:
        print(f"Search Plan -> resumed {len(cached.queries)} planned query(ies)")
        return cached
    llm = ChatOpenAI(
        base_url=llm_config.base_url,
        api_key=SecretStr(llm_config.api_key),
        model=llm_config.model_name,
        temperature=0.1,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )
    planner = SEARCH_PLAN_PROMPT | llm | StrOutputParser()
    print(f"Search Plan -> requesting up to {MAX_PLAN_QUERIES} queries for {target} paper(s)")
    try:
        raw = planner.invoke(
            {
                "topic": topic,
                "objective": objective,
                "target": target,
                "max_queries": MAX_PLAN_QUERIES,
                "max_terms": MAX_PLAN_QUERY_TERMS,
            }
        )
    except Exception as exc:
        return _fallback_plan(topic, objective, target, f"planner unavailable: {exc}")
    planned = _parse_plan(raw)
    if not planned:
        return _fallback_plan(topic, objective, target, "planner returned no usable query")
    topic_first = [PlannedQuery(query=topic.strip(), intent="core topic", tier="core")] + planned
    plan = SearchPlan(
        topic=topic,
        objective=objective,
        target=target,
        queries=_deduplicate_plan(topic_first),
        source="llm",
    )
    for index, item in enumerate(plan.queries, start=1):
        print(f"Search Plan -> {index}. [{item.tier}] {item.query} ({item.intent})")
    _save_plan(cache_path, plan)
    return plan


def plan_as_json(plan: SearchPlan) -> dict:
    return {
        "version": PLAN_CACHE_VERSION,
        "topic": plan.topic,
        "objective": plan.objective,
        "target": plan.target,
        "source": plan.source,
        "signature": plan.signature,
        "queries": [
            {"query": item.query, "intent": item.intent, "tier": item.tier} for item in plan.queries
        ],
    }


def _save_plan(cache_path: Path | None, plan: SearchPlan) -> None:
    if cache_path is None:
        return
    atomic_write_json(cache_path, plan_as_json(plan))


def _load_plan(
    cache_path: Path | None, topic: str, objective: str, target: int
) -> SearchPlan | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            data.get("version") != PLAN_CACHE_VERSION
            or data.get("topic") != topic
            or data.get("objective") != objective
            or data.get("target") != target
        ):
            return None
        queries = tuple(
            PlannedQuery(
                query=str(item["query"]),
                intent=str(item.get("intent", "unspecified facet")),
                tier=str(item.get("tier", "related")),
            )
            for item in data["queries"]
        )
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return None
    if not queries:
        return None
    return SearchPlan(topic, objective, target, queries, str(data.get("source", "cache")))


def iterative_search(
    topic: str,
    objective: str,
    llm_config: LLMConfig,
    max_results: int,
    cache_path: Path | None,
    plan: SearchPlan | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    plan_cache_path: Path | None = None,
) -> list[Paper]:
    if plan is None:
        plan = plan_searches(topic, objective, llm_config, max_results, plan_cache_path)
    cache_identity = hashlib.sha256(
        "|".join(
            [
                objective,
                plan.signature,
                llm_config.base_url,
                llm_config.model_name,
                str(year_min),
                str(year_max),
                SEARCH_STRATEGY_VERSION,
            ]
        ).encode()
    ).hexdigest()
    cached = load_search_cache(cache_path, topic, max_results, cache_identity)
    if cached is not None:
        print(f"Search Agent -> resumed {len(cached)} cached paper(s)")
        return cached

    per_query = _per_query_quota(max_results, len(plan.queries))
    papers: list[Paper] = []
    seen: set[str] = set()
    executed: list[frozenset[str]] = []
    consecutive_empty = 0

    for round_number, planned in enumerate(plan.queries, start=1):
        if len(papers) >= max_results:
            break
        signature = _query_signature(planned.query)
        if not signature or signature in executed:
            continue
        executed.append(signature)
        print(
            f"Search Agent -> plan round {round_number}/{len(plan.queries)} "
            f"[{planned.tier}]: {planned.query}"
        )
        candidates = _search_with_relaxation(
            planned.query, per_query, executed, year_min, year_max, planned.tier
        )
        added = _collect(candidates, papers, seen, max_results)
        print(f"Search Agent -> added {added} novel paper(s) (total {len(papers)})")

    level = 0
    round_number = len(plan.queries)
    while len(papers) < max_results and round_number < MAX_SEARCH_ROUNDS:
        round_number += 1
        level += 1
        relaxed = _relaxed_query(topic, level)
        if not relaxed:
            break
        signature = _query_signature(relaxed)
        if signature in executed:
            continue
        executed.append(signature)
        print(f"Search Agent -> relaxed round {round_number}/{MAX_SEARCH_ROUNDS}: {relaxed}")
        candidates = _search_with_relaxation(
            relaxed, per_query, executed, year_min, year_max, "related"
        )
        added = _collect(candidates, papers, seen, max_results)
        print(f"Search Agent -> added {added} novel paper(s) (total {len(papers)})")
        if added == 0:
            consecutive_empty += 1
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY_ROUNDS:
                print("Search Agent -> stopped: repeated empty rounds")
                break
        else:
            consecutive_empty = 0

    if not papers:
        raise RuntimeError("Search agent found no papers.")
    if len(papers) >= max_results:
        print("Search Agent -> stopped: coverage target reached")
    papers = papers[:max_results]
    save_search_cache(cache_path, topic, max_results, papers, cache_identity)
    print(f"Search Agent -> selected {len(papers)} unique paper(s)")
    if len(papers) < max_results:
        print(
            "Search Agent -> corpus exhausted: "
            f"only {len(papers)} paper(s) found for this topic and its relaxations"
        )
    return papers


def _search_with_relaxation(
    query: str,
    per_query: int,
    executed: list[frozenset[str]],
    year_min: int | None,
    year_max: int | None,
    tier: str,
) -> list[Paper]:
    try:
        return search_arxiv(
            query,
            per_query,
            year_min=year_min,
            year_max=year_max,
            max_terms=MAX_PLAN_QUERY_TERMS,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Search Agent -> query produced no usable result: {exc}")
    relaxed = _relaxed_query(query, 1)
    if not relaxed or _query_signature(relaxed) in executed:
        return []
    executed.append(_query_signature(relaxed))
    print(f"Search Agent -> relaxing query: {relaxed!r}")
    try:
        return search_arxiv(
            relaxed,
            per_query,
            year_min=year_min,
            year_max=year_max,
            max_terms=MAX_PLAN_QUERY_TERMS,
        )
    except (RuntimeError, ValueError):
        return []


def _collect(candidates: list[Paper], papers: list[Paper], seen: set[str], max_results: int) -> int:
    added = 0
    for paper in candidates:
        if len(papers) >= max_results:
            break
        identity = (paper.doi or paper.arxiv_id).lower()
        if identity in seen:
            continue
        seen.add(identity)
        papers.append(paper)
        added += 1
    return added
