import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from literature_rag.config import (
    LLMConfig,
    _to_llm_config,
    check_endpoint,
    choose_llm,
    get_semantic_scholar_key,
    remove_endpoint,
    remove_model,
)


def _config() -> LLMConfig:
    return LLMConfig("https://llm.example/v1", "secret-key", "model")


def _patch_openai(monkeypatch, behavior):
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **_kwargs: RunnableLambda(behavior))


def test_check_endpoint_accepts_responsive_model(monkeypatch):
    _patch_openai(monkeypatch, lambda _messages: AIMessage(content="OK"))
    check_endpoint(_config())


def test_check_endpoint_rejects_unreachable_model(monkeypatch):
    def fail(_messages):
        raise RuntimeError("connection refused")

    _patch_openai(monkeypatch, fail)
    with pytest.raises(ValueError, match="Endpoint check failed"):
        check_endpoint(_config())


def test_check_endpoint_rejects_empty_response(monkeypatch):
    _patch_openai(monkeypatch, lambda _messages: AIMessage(content=""))
    with pytest.raises(ValueError, match="empty response"):
        check_endpoint(_config())


def test_s2_key_from_environment(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "env-key")
    monkeypatch.setattr(
        "literature_rag.config.keyring.get_password",
        lambda *_: "stored-key",
    )
    assert get_semantic_scholar_key() == "env-key"


def test_s2_key_from_keyring(monkeypatch):
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setattr(
        "literature_rag.config.keyring.get_password",
        lambda *_args: "stored-key",
    )
    assert get_semantic_scholar_key() == "stored-key"


def test_s2_key_prompted_and_stored(monkeypatch):
    stored = {}
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setattr(
        "literature_rag.config.keyring.get_password",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "literature_rag.config.keyring.set_password",
        lambda service, user, secret: stored.update({user: secret}),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "typed-key")
    assert get_semantic_scholar_key() == "typed-key"
    assert stored["semantic-scholar"] == "typed-key"


def test_s2_key_skipped_when_empty(monkeypatch):
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setattr(
        "literature_rag.config.keyring.get_password",
        lambda *_args: None,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert get_semantic_scholar_key() == ""


def test_legacy_key_is_migrated_out_of_plaintext_config(tmp_path, monkeypatch):
    stored = {}
    monkeypatch.setattr(
        "literature_rag.config.keyring.set_password",
        lambda service, reference, secret: stored.update({reference: secret}),
    )
    monkeypatch.setattr(
        "literature_rag.config.keyring.get_password",
        lambda service, reference: stored.get(reference),
    )
    path = tmp_path / "config.json"
    profile = {
        "name": "local",
        "base_url": "https://llm.example/v1",
        "api_key": "secret",
        "models": ["model"],
    }
    config = {"profiles": [profile]}

    result = _to_llm_config(profile, "model", config, path)

    assert result == LLMConfig("https://llm.example/v1", "secret", "model")
    assert "api_key" not in json.loads(path.read_text())["profiles"][0]


def _profile(models=None):
    return {
        "name": "provider",
        "base_url": "https://llm.example/v1",
        "key_ref": "key-ref",
        "models": models or ["model-a", "model-b"],
    }


def test_remove_model_updates_config_and_keeps_endpoint(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    config = {"profiles": [_profile()]}
    answers = iter(["1", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert remove_model(config, path)
    assert config["profiles"][0]["models"] == ["model-b"]
    assert json.loads(path.read_text())["profiles"][0]["models"] == ["model-b"]


def test_remove_model_can_be_cancelled(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    config = {"profiles": [_profile()]}
    answers = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert not remove_model(config, path)
    assert config["profiles"][0]["models"] == ["model-a", "model-b"]
    assert not path.exists()


def test_remove_endpoint_deletes_keyring_entry(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    config = {"profiles": [_profile()]}
    deleted = []
    answers = iter(["1", "yes"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        "literature_rag.config.keyring.delete_password",
        lambda service, reference: deleted.append((service, reference)),
    )

    assert remove_endpoint(config, path)
    assert config["profiles"] == []
    assert json.loads(path.read_text())["profiles"] == []
    assert deleted == [("agentic-literature-rag", "key-ref")]


def test_choose_llm_refreshes_after_model_removal(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"profiles": [_profile()]}))
    answers = iter(["5", "1", "y", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(
        "literature_rag.config.keyring.get_password",
        lambda service, reference: "secret",
    )

    result = choose_llm(path)

    assert result == LLMConfig("https://llm.example/v1", "secret", "model-b")
    assert json.loads(path.read_text())["profiles"][0]["models"] == ["model-b"]
