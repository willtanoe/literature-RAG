import pytest

from literature_rag.config import LLMConfig
from literature_rag.papers import Paper
from literature_rag.pipeline import PlanRejected, _drop_excluded, run_pipeline
from literature_rag.search_agent import PlannedQuery, SearchPlan


def _plan() -> SearchPlan:
    return SearchPlan(
        topic="topic",
        objective="objective",
        target=5,
        queries=(PlannedQuery("topic", "core topic", "core"),),
        source="fallback",
    )


def _stub_plan(monkeypatch):
    monkeypatch.setattr("literature_rag.pipeline.plan_searches", lambda *_, **__: _plan())


def test_pipeline_persists_redacted_early_failure(tmp_path, monkeypatch):
    _stub_plan(monkeypatch)
    monkeypatch.setattr(
        "literature_rag.pipeline.iterative_search",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("failed with secret-key")),
    )
    config = LLMConfig("https://llm.example/v1", "secret-key", "model")

    with pytest.raises(RuntimeError):
        run_pipeline("topic", "objective", config, download_dir=tmp_path)

    failure = (tmp_path / "output" / "failure.txt").read_text()
    assert "secret-key" not in failure
    assert "[REDACTED]" in failure
    assert not (tmp_path / ".lock").exists()


def test_pipeline_writes_run_log(tmp_path, monkeypatch):
    _stub_plan(monkeypatch)
    monkeypatch.setattr(
        "literature_rag.pipeline.iterative_search",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    config = LLMConfig("https://llm.example/v1", "secret-key", "model")

    with pytest.raises(RuntimeError):
        run_pipeline("topic", "objective", config, download_dir=tmp_path)

    log = (tmp_path / "output" / "run.log").read_text(encoding="utf-8")
    assert "run started" in log
    assert not (tmp_path / ".lock").exists()


def test_rejected_plan_stops_before_search_without_failure_file(tmp_path, monkeypatch):
    _stub_plan(monkeypatch)
    monkeypatch.setattr(
        "literature_rag.pipeline.iterative_search",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("search must not run")),
    )
    config = LLMConfig("https://llm.example/v1", "secret-key", "model")

    with pytest.raises(PlanRejected):
        run_pipeline(
            "topic",
            "objective",
            config,
            download_dir=tmp_path,
            plan_confirm=lambda _plan: False,
        )

    assert not (tmp_path / "output" / "failure.txt").exists()
    assert not (tmp_path / ".lock").exists()


def test_drop_excluded_keeps_other_tiers_and_requires_survivors():
    papers = [
        Paper("1", "Core", (), "e1", "", tier="core"),
        Paper("2", "Gone", (), "e2", "", tier="excluded"),
    ]

    assert [paper.arxiv_id for paper in _drop_excluded(papers)] == ["1"]
    with pytest.raises(RuntimeError):
        _drop_excluded([papers[1]])
