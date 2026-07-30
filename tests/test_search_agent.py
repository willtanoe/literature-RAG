from literature_rag.config import LLMConfig
from literature_rag.papers import Paper
from literature_rag.search_agent import iterative_search


class FakePlanner:
    def __or__(self, other):
        return self

    def invoke(self, values):
        return "gap query"


def test_iterative_search_deduplicates_and_resumes(tmp_path, monkeypatch):
    first = Paper("1", "First", (), "e1", "p1")
    second = Paper("2", "Second", (), "e2", "p2")
    calls = []

    def search(query, count):
        calls.append((query, count))
        return [first] if query == "topic" else [first, second]

    monkeypatch.setattr("literature_rag.search_agent.ChatOpenAI", lambda **_: FakePlanner())
    monkeypatch.setattr("literature_rag.search_agent.StrOutputParser", FakePlanner)
    monkeypatch.setattr("literature_rag.search_agent.SEARCH_PROMPT", FakePlanner())
    monkeypatch.setattr("literature_rag.search_agent.search_arxiv", search)
    cache = tmp_path / "search.json"
    config = LLMConfig("https://llm.example/v1", "secret", "model")

    result = iterative_search("topic", "objective", config, 2, cache)
    calls.clear()
    resumed = iterative_search("topic", "objective", config, 2, cache)

    assert result == [first, second]
    assert resumed == result
    assert calls == []
