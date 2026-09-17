from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError

T = TypeVar("T")
SECRET_PATTERN = re.compile(r"(?i)(api[_ -]?key|authorization|bearer)([\s:=]+)(\S+)")


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
def tee_stdout(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original = sys.stdout

    class _Tee:
        def write(self, data: str) -> int:
            original.write(data)
            if data and ("\n" in data or "\r" not in data):
                log_file.write(data.replace("\r", ""))
            return len(data)

        def flush(self) -> None:
            original.flush()
            log_file.flush()

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== run started {datetime.now(timezone.utc).isoformat()} =====\n")
        sys.stdout = _Tee()
        try:
            yield
        finally:
            sys.stdout = original


@contextmanager
def workspace_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Workspace is already in use: {root}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock.unlink(missing_ok=True)
