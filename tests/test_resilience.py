import json
from urllib.error import URLError

import pytest

from literature_rag.resilience import atomic_write_json, redact_secrets, retry, workspace_lock


def test_atomic_json_and_secret_redaction(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"ready": True})
    assert json.loads(path.read_text()) == {"ready": True}
    assert "secret" not in redact_secrets("Authorization: Bearer secret", ("secret",))


def test_workspace_lock_rejects_concurrent_writer(tmp_path):
    with workspace_lock(tmp_path):
        competing_lock = workspace_lock(tmp_path)
        with pytest.raises(RuntimeError, match="already in use"):
            competing_lock.__enter__()
    assert not (tmp_path / ".lock").exists()


def test_retry_recovers_from_transient_network_error(monkeypatch):
    monkeypatch.setattr("literature_rag.resilience.time.sleep", lambda _: None)
    attempts = []

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise URLError("temporary")
        return "ready"

    assert retry(operation, attempts=3, base_delay=0) == "ready"
    assert len(attempts) == 3
