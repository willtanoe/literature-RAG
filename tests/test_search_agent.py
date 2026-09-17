from literature_rag.config import LLMConfig
from literature_rag.papers import Paper
from literature_rag.search_agent import (
    _normalize_query,
    _per_query_quota,
    _relaxed_query,
    iterative_search,
    plan_as_json,
    plan_searches,
)


class FakePlanner:
    def __init__(self, responses=None):
        self.responses = responses or ["gap query"]

    def __or__(self, other):
        return self

    def invoke(self, values):
        return self.responses.pop(0) if self.responses else "gap query"


def test_iterative_search_deduplicates_and_resumes(tmp_path, monkeypatch):
    first = Paper("1", "First", (), "e1", "p1")
    second = Paper("2", "Second", (), "e2", "p2")
    calls = []

    def search(query, count, **_kwargs):
        calls.append((query, count))
        return [first] if query == "topic" else [first, second]

    monkeypatch.setattr("literature_rag.search_agent.ChatOpenAI", lambda **_: FakePlanner())
    monkeypatch.setattr("literature_rag.search_agent.StrOutputParser", FakePlanner)
    monkeypatch.setattr("literature_rag.search_agent.SEARCH_PLAN_PROMPT", FakePlanner())
    monkeypatch.setattr("literature_rag.search_agent.search_arxiv", search)
    cache = tmp_path / "search.json"
    config = LLMConfig("https://llm.example/v1", "secret", "model")

    result = iterative_search("topic", "objective", config, 2, cache)
    calls.clear()
    resumed = iterative_search("topic", "objective", config, 2, cache)

    assert result == [first, second]
    assert resumed == result
    assert calls == []


def test_relaxed_query_ladder():
    query = "differential privacy federated learning intrusion detection system"
    first = _relaxed_query(query, 1)
    second = _relaxed_query(query, 2)
    third = _relaxed_query(query, 3)
    assert first == "differential privacy federated learning intrusion"
    assert second == "differential privacy federated"
    assert third == "differential privacy federated"
    assert _relaxed_query("short query", 1) == ""


def test_iterative_search_extends_rounds_until_target(tmp_path, monkeypatch):
    topic = "differential privacy federated learning intrusion detection"
    core = [
        Paper("1", "Core One", (), "e1", "p1"),
        Paper("2", "Core Two", (), "e2", "p2"),
    ]
    relaxed_1 = [
        Paper("3", "Neighbor Three", (), "e3", "p3"),
        Paper("4", "Neighbor Four", (), "e4", "p4"),
    ]
    relaxed_2 = [
        Paper("5", "Neighbor Five", (), "e5", "p5"),
        Paper("6", "Neighbor Six", (), "e6", "p6"),
    ]
    by_query = {
        topic: core,
        "differential privacy federated learning": relaxed_1,
        "differential privacy federated": relaxed_2,
    }

    def search(query, count, **_kwargs):
        if query in by_query:
            return by_query[query]
        raise RuntimeError(f"No arXiv papers found for {query!r}.")

    planner = FakePlanner(responses=["narrow query one", "narrow query two"])
    monkeypatch.setattr("literature_rag.search_agent.ChatOpenAI", lambda **_: planner)
    monkeypatch.setattr("literature_rag.search_agent.StrOutputParser", FakePlanner)
    monkeypatch.setattr("literature_rag.search_agent.SEARCH_PLAN_PROMPT", FakePlanner())
    monkeypatch.setattr("literature_rag.search_agent.search_arxiv", search)
    config = LLMConfig("https://llm.example/v1", "secret", "model")

    result = iterative_search(
        "differential privacy federated learning intrusion detection",
        "objective",
        config,
        6,
        tmp_path / "search.json",
    )

    assert len(result) == 6
    assert [paper.title for paper in result] == [
        "Core One",
        "Core Two",
        "Neighbor Three",
        "Neighbor Four",
        "Neighbor Five",
        "Neighbor Six",
    ]


PLAN_JSON = """```json
[
  {"query": "all:federated AND all:learning privacy", "intent": "core methods", "tier": "core"},
  {"query": "Query: privacy federated learning", "intent": "duplicate wording", "tier": "core"},
  {"query": "differential privacy noise calibration budget accounting extra",
   "intent": "privacy budgets", "tier": "related"},
  {"query": "arxiv recent papers", "intent": "noise only", "tier": "related"}
]
```"""


def test_normalize_query_strips_syntax_noise_and_caps_terms():
    assert (
        _normalize_query("all:federated AND all:learning privacy") == "federated learning privacy"
    )
    assert _normalize_query("2. Query: arXiv recent papers") == ""
    assert (
        _normalize_query("differential privacy noise calibration budget accounting")
        == "differential privacy noise calibration"
    )


def test_per_query_quota_overshoots_but_stays_bounded():
    assert _per_query_quota(80, 8) == 20
    assert _per_query_quota(4, 8) == 10
    assert _per_query_quota(4000, 2) == 100


def test_plan_searches_normalizes_dedupes_and_resumes_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("literature_rag.search_agent.ChatOpenAI", lambda **_: FakePlanner())
    monkeypatch.setattr("literature_rag.search_agent.StrOutputParser", FakePlanner)
    monkeypatch.setattr(
        "literature_rag.search_agent.SEARCH_PLAN_PROMPT", FakePlanner(responses=[PLAN_JSON])
    )
    cache = tmp_path / "search_plan.json"
    config = LLMConfig("https://llm.example/v1", "secret", "model")

    plan = plan_searches("federated privacy", "objective", config, 80, cache)

    assert plan.source == "llm"
    assert [item.query for item in plan.queries] == [
        "federated privacy",
        "federated learning privacy",
        "differential privacy noise calibration",
    ]
    assert plan.queries[0].tier == "core"
    assert plan_as_json(plan)["target"] == 80

    def fail_planner(**_kwargs):
        raise AssertionError("planner must not run when a cached plan matches")

    monkeypatch.setattr("literature_rag.search_agent.ChatOpenAI", fail_planner)
    resumed = plan_searches("federated privacy", "objective", config, 80, cache)

    assert [item.query for item in resumed.queries] == [item.query for item in plan.queries]


def test_plan_searches_falls_back_to_relaxation_ladder(tmp_path, monkeypatch):
    class BrokenPlanner(FakePlanner):
        def invoke(self, values):
            raise RuntimeError("planner offline")

    monkeypatch.setattr("literature_rag.search_agent.ChatOpenAI", lambda **_: BrokenPlanner())
    monkeypatch.setattr("literature_rag.search_agent.StrOutputParser", BrokenPlanner)
    monkeypatch.setattr("literature_rag.search_agent.SEARCH_PLAN_PROMPT", BrokenPlanner())
    config = LLMConfig("https://llm.example/v1", "secret", "model")

    plan = plan_searches(
        "differential privacy federated learning intrusion detection",
        "objective",
        config,
        20,
        tmp_path / "plan.json",
    )

    assert plan.source == "fallback"
    assert plan.queries[0].query.startswith("differential privacy federated")
    assert plan.queries[1].query == "differential privacy federated learning"
    assert all(item.tier == "related" for item in plan.queries[1:])
