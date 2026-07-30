import pytest

from literature_rag.config import LLMConfig
from literature_rag.pipeline import run_pipeline


def test_pipeline_persists_redacted_early_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "literature_rag.pipeline.iterative_search",
        lambda *_: (_ for _ in ()).throw(RuntimeError("failed with secret-key")),
    )
    config = LLMConfig("https://llm.example/v1", "secret-key", "model")

    with pytest.raises(RuntimeError):
        run_pipeline("topic", "objective", config, download_dir=tmp_path)

    failure = (tmp_path / "output" / "failure.txt").read_text()
    assert "secret-key" not in failure
    assert "[REDACTED]" in failure
    assert not (tmp_path / ".lock").exists()
