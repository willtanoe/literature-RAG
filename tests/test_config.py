import json

from literature_rag.config import LLMConfig, _to_llm_config


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
