from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError

from literature_rag.__log__ import get_logger

T = TypeVar("T")
SECRET_PATTERN = re.compile(r"(?i)(api[_ -]?key|authorization|bearer)([\s:=]+)(\S+)")

logger = get_logger(__name__)


def atomic_write_text(path: Path, content: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        if mode is not None:
            os.chmod(temporary, mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, mode: int | None = None) -> None:
    atomic_write_text(path, json.dumps(value, indent=2), mode)


def retry(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    retry_after: Callable[[Exception], float | None] | None = None,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            if (
                attempt == attempts
                or isinstance(exc, HTTPError)
                and exc.code < 500
                and exc.code != 429
            ):
                raise
            specified = retry_after(exc) if retry_after else None
            delay = specified if specified is not None else base_delay * 2 ** (attempt - 1)
            time.sleep(delay + random.uniform(0, min(delay * 0.1, 0.25)))
    raise RuntimeError("retry exhausted")


def http_retry_after(exc: Exception) -> float | None:
    if not isinstance(exc, HTTPError) or exc.code != 429:
        return None
    value = exc.headers.get("Retry-After")
    try:
        return min(float(value), 60.0) if value else None
    except ValueError:
        return None


def redact_secrets(message: object, secrets: tuple[str, ...] = ()) -> str:
    result = SECRET_PATTERN.sub(r"\1\2[REDACTED]", str(message))
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


@contextmanager
def workspace_lock(root: Path, timeout_seconds: int = 300) -> Iterator[None]:
    """Acquire exclusive workspace lock with stale-lock detection.
    
    Args:
        root: Workspace root directory to lock.
        timeout_seconds: Maximum lock age in seconds before considered stale (default 5 min).
    
    Raises:
        RuntimeError: If lock already held by running process, or stale lock exists.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    current_pid = os.getpid()
    
    # Try to create exclusive lock
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # Read existing lock file
        try:
            lock_content = json.loads(lock.read_text(encoding="utf-8"))
            locked_pid = lock_content.get("pid")
            lock_time = lock_content.get("timestamp", 0)
            
            # Check if our own lock (shouldn't happen normally, but handle gracefully)
            if locked_pid == current_pid:
                yield
                return
            
            # Check if lock is stale (>5 min default)
            age = time.time() - lock_time
            if age > timeout_seconds:
                logger.warning(f"Removing stale lock from PID {locked_pid} ({int(age)}s old)")
                lock.unlink()
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            elif locked_pid and _is_process_alive(locked_pid):
                raise RuntimeError(f"Workspace in use by PID {locked_pid}")
            else:
                # Orphaned lock (process gone or invalid PID)
                logger.warning(f"Removing orphaned lock from PID {locked_pid}")
                lock.unlink()
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Lock file corrupted: {e}")
            raise RuntimeError(f"Cannot acquire lock: {e}")
        except Exception as e:
            raise RuntimeError(f"Workspace conflict: {e}")
    
    try:
        # Write lock metadata
        lock_data = {
            "pid": current_pid,
            "timestamp": time.time(),
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown"
        }
        os.write(descriptor, json.dumps(lock_data).encode())
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock.unlink(missing_ok=True)
