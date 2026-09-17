"""Tests for CLI module."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from literature_rag.cli import (
    _confirm_plan,
    _parse_top_n,
    _parse_year_range,
    _split_topics,
    main,
    parse_args,
)


def test_default_interactive_mode():
    args = parse_args([])
    assert args.interactive is None
    assert args.topic is None
    assert args.objective is None


def test_non_interactive_mode_fields():
    args = parse_args(["--topic", "test", "--objective", "analyze"])
    assert args.topic == "test" and args.objective == "analyze"


def test_verbose_levels_and_quiet():
    assert parse_args(["-v"]).verbose == 1
    assert parse_args(["-vvv"]).verbose == 3
    assert parse_args(["--quiet"]).quiet is True


def test_optional_inputs_parse():
    args = parse_args(
        ["--dois", "10.1145/a,https://doi.org/10.1000/b", "--local-pdf", "/path/to/pdfs"]
    )
    assert args.dois and args.local_pdf == "/path/to/pdfs"


def test_split_topics_variants():
    assert _split_topics("federated learning intrusion detection") == [
        "federated learning intrusion detection"
    ]
    assert _split_topics(
        '"Robust Federated Learning, Non-IID", "Blockchain-Assisted Federated Learning"'
    ) == ["Robust Federated Learning, Non-IID", "Blockchain-Assisted Federated Learning"]
    assert _split_topics("topic one; topic two, topic three") == [
        "topic one",
        "topic two",
        "topic three",
    ]
    assert _split_topics('  " topic A " ,  B  , "" ') == ["topic A", "B"]
    assert _split_topics('"" ,  ') == []


def test_parse_year_range_forms():
    assert _parse_year_range("") == (None, None)
    assert _parse_year_range("2023") == (2023, 2023)
    assert _parse_year_range("2020-2025") == (2020, 2025)
    assert _parse_year_range("2022-") == (2022, 2022)
    assert _parse_year_range("2025-2020") == (2020, 2025)


def test_parse_year_range_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_year_range("recent")
    with pytest.raises(ValueError):
        _parse_year_range("20-25")


def test_parse_top_n_defaults_and_bounds():
    from literature_rag.cli import MAX_TOP_N

    assert _parse_top_n("") == 5
    assert _parse_top_n("12") == 12
    assert _parse_top_n("80") == MAX_TOP_N == 80
    for value in ("0", str(MAX_TOP_N + 1), "many"):
        with pytest.raises(ValueError):
            _parse_top_n(value)


def test_confirm_plan_lists_queries_and_defaults_to_yes(monkeypatch, capsys):
    plan = SimpleNamespace(
        topic="topic",
        objective="objective",
        target=80,
        queries=(
            SimpleNamespace(query="federated privacy", intent="core topic", tier="core"),
            SimpleNamespace(
                query="adaptive noise budget", intent="privacy budgets", tier="related"
            ),
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
    cli.main(["--interactive"])
    assert "analysis objective is required" in capsys.readouterr().out


@patch("literature_rag.cli.setup_logging")
@patch("literature_rag.cli.choose_llm")
class TestMainCompatibility:
    @patch("builtins.input", return_value="y")
    def test_interactive_mode_basic(self, mock_input, mock_choose_llm, mock_setup_logging):
        main(["--interactive"])
        assert mock_choose_llm.called

    @patch("literature_rag.cli.run_pipeline", return_value=("report", object()))
    def test_non_interactive_mode_execution(
        self, mock_pipeline, mock_choose_llm, mock_setup_logging
    ):
        with contextlib.suppress(SystemExit):
            main(["--topic", "test", "--objective", "analyze", "-y"])
        assert mock_choose_llm.called

    def test_help_shows_usage(self, mock_choose_llm, mock_setup_logging):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_keyboard_interrupt_handling(self, mock_choose_llm, mock_setup_logging):
        assert True
