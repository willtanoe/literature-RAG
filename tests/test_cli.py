from literature_rag.cli import _confirm_plan, _parse_top_n, _parse_year_range, _split_topics
from literature_rag.search_agent import PlannedQuery, SearchPlan


def test_split_topics_single_topic():
    assert _split_topics("federated learning intrusion detection") == [
        "federated learning intrusion detection"
    ]


def test_split_topics_quoted_titles_with_commas_outside():
    raw = (
        '"Robust Federated Learning, Non-IID", '
        '"Blockchain-Assisted Federated Learning", "Communication-Efficient FL"'
    )
    assert _split_topics(raw) == [
        "Robust Federated Learning, Non-IID",
        "Blockchain-Assisted Federated Learning",
        "Communication-Efficient FL",
    ]


def test_split_topics_plain_comma_and_semicolon_separators():
    assert _split_topics("topic one; topic two, topic three") == [
        "topic one",
        "topic two",
        "topic three",
    ]


def test_split_topics_strips_whitespace_and_quotes():
    assert _split_topics('  " topic A " ,  B  , "" ') == ["topic A", "B"]


def test_split_topics_empty_input():
    assert _split_topics('"" ,  ') == []


def test_parse_year_range_forms():
    assert _parse_year_range("") == (None, None)
    assert _parse_year_range("2023") == (2023, 2023)
    assert _parse_year_range("2020-2025") == (2020, 2025)
    assert _parse_year_range("2022-") == (2022, 2022)
    assert _parse_year_range("2025-2020") == (2020, 2025)


def test_parse_year_range_rejects_garbage():
    import pytest

    with pytest.raises(ValueError):
        _parse_year_range("recent")
    with pytest.raises(ValueError):
        _parse_year_range("20-25")


def test_parse_top_n_defaults_and_bounds():
    from literature_rag.cli import MAX_TOP_N

    assert _parse_top_n("") == 5
    assert _parse_top_n("12") == 12
    assert _parse_top_n("80") == 80
    assert MAX_TOP_N == 80
    assert _parse_top_n(str(MAX_TOP_N)) == MAX_TOP_N
    import pytest

    with pytest.raises(ValueError):
        _parse_top_n("0")
    with pytest.raises(ValueError):
        _parse_top_n(str(MAX_TOP_N + 1))
    with pytest.raises(ValueError):
        _parse_top_n("many")


def test_confirm_plan_lists_queries_and_defaults_to_yes(monkeypatch, capsys):
    plan = SearchPlan(
        topic="topic",
        objective="objective",
        target=80,
        queries=(
            PlannedQuery("federated privacy", "core topic", "core"),
            PlannedQuery("adaptive noise budget", "privacy budgets", "related"),
        ),
        source="llm",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    assert _confirm_plan(plan) is True

    output = capsys.readouterr().out
    assert "Search plan (llm) for 80 target paper(s)" in output
    assert "1. [core] federated privacy  -- core topic" in output

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert _confirm_plan(plan) is False


def test_main_rejects_missing_objective(monkeypatch, capsys):
    import literature_rag.cli as cli

    answers = iter(["topic only", ""])
    monkeypatch.setattr(cli, "choose_llm", lambda: object())
    monkeypatch.setattr(cli, "check_endpoint", lambda _config: None)
    monkeypatch.setattr(cli, "get_semantic_scholar_key", lambda: "")
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    cli.main()

    assert "analysis objective is required" in capsys.readouterr().out
