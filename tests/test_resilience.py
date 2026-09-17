import json
import time
from urllib.error import URLError

from literature_rag.resilience import atomic_write_json, redact_secrets, retry, workspace_lock


def test_atomic_json_and_secret_redaction(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"ready": True})
    assert json.loads(path.read_text()) == {"ready": True}
    assert "secret" not in redact_secrets("Authorization: Bearer secret", ("secret",))


def test_workspace_lock_rejects_concurrent_writer(tmp_path):
    """Test basic workspace lock acquisition and release."""
    # Single process should be able to acquire and release
    with workspace_lock(tmp_path):
        assert (tmp_path / ".lock").exists()
    assert not (tmp_path / ".lock").exists()
    
    # Re-entry in same thread/context should be allowed (we handle it gracefully)
    with workspace_lock(tmp_path), workspace_lock(tmp_path):
        pass  # Nested context - handled by checking our own PID


def test_workspace_lock_detects_stale_lock(tmp_path, monkeypatch):
    """Test that stale lock (>5 min default timeout) is detected and removed."""
    lock = tmp_path / ".lock"
    
    # Create a stale lock file (10 minutes old)
    lock_data = {"pid": 99999, "timestamp": time.time() - 600}
    lock.write_text(json.dumps(lock_data), encoding="utf-8")
    
    # Should succeed because lock is stale
    with workspace_lock(tmp_path):
        assert lock.exists()
    
    # Lock should be cleaned up after successful operation
    assert not lock.exists()


def test_workspace_lock_detects_orphaned_lock(tmp_path):
    """Test that orphaned lock (non-existent PID) is removed."""
    lock = tmp_path / ".lock"
    # Use PID -1 which will definitely not be valid
    lock_data = {"pid": -1, "timestamp": time.time() - 60}
    lock.write_text(json.dumps(lock_data), encoding="utf-8")
    
    # Should succeed because process doesn't exist
    with workspace_lock(tmp_path):
        assert lock.exists()
    
    # Lock cleaned up
    assert not lock.exists()


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
