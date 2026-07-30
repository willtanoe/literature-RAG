# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Quality Gates

Run these checks before submitting a change:

```bash
pytest --cov=literature_rag --cov-report=term-missing
ruff format --check .
ruff check .
mypy literature_rag
python -m compileall -q agentic_literature_rag.py literature_rag tests
```

## Pull Requests

- Keep changes focused and explain user-visible behavior.
- Add or update tests for bug fixes and features.
- Update documentation when configuration, security boundaries, or output formats change.
- Never commit API keys, private PDFs, generated workspaces, model files, or local configuration.
- Preserve cross-platform behavior for Linux, macOS, and Windows.
