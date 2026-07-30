# Agentic Literature Review and RAG

A cross-platform CLI for discovering academic literature on arXiv, managing research PDFs, building a persistent FAISS index with local embeddings, and generating evidence-grounded literature reviews through any OpenAI-compatible LLM endpoint.

The application maintains a dedicated workspace for each research topic. Search results, PDFs, scholarly metadata, and vector indexes are reused across sessions, reducing repeated downloads and embedding work. Final reviews include an evidence matrix, page-level citations, citation auditing, and bibliography exports.

## Key Features

- Iterative arXiv discovery with up to three gap-aware search rounds.
- Paper ingestion from arXiv, DOI lists, and local PDF directories.
- Scholarly metadata enrichment through Crossref and Semantic Scholar.
- Legal open-access PDF discovery through Unpaywall.
- Guided manual recovery when a PDF cannot be downloaded automatically.
- PDF signature validation and deduplication by DOI, arXiv ID, and file hash.
- Local CPU embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Persistent FAISS indexes with automatic invalidation when the corpus changes.
- Safe FAISS persistence using a native index plus JSON documents, without pickle loading.
- Bounded HTTPS downloads, PDF/page/chunk limits, retries, and rate-limit handling.
- Atomic state writes and per-workspace locking for interruption and concurrency safety.
- MMR retrieval for broader evidence coverage.
- Evidence matrices covering methods, datasets, metrics, findings, and limitations.
- Inline source labels containing paper titles and PDF page numbers.
- A critic pass that reviews claims, conflicting findings, limitations, and citations.
- Markdown, JSON, and BibTeX exports.
- Multiple saved LLM endpoints, API keys, and models.

## Pipeline

```text
Research topic and objective
            |
            v
Iterative arXiv search + DOI and local PDF import
            |
            v
Crossref / Semantic Scholar / Unpaywall enrichment
            |
            v
Download, manual recovery, validation, and deduplication
            |
            v
Local embeddings + persistent FAISS index
            |
            v
MMR retrieval + evidence-grounded synthesis
            |
            v
Critic pass + deterministic citation audit
            |
            v
Markdown + JSON + BibTeX exports
```

## Requirements

- Python 3.10 or later.
- Internet access for arXiv, scholarly metadata services, the initial embedding model download, and the LLM endpoint.
- An endpoint compatible with the OpenAI Chat Completions API.
- Sufficient storage for PDFs, the embedding model, and FAISS indexes.

The embedding model is downloaded on first use and then runs locally on the CPU.

## Installation

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Conda or Micromamba

Create the provided environment:

```bash
micromamba create -f environment.yml
micromamba activate literature-rag
```

Equivalent manual setup:

```bash
micromamba create -n literature-rag python=3.11 -c conda-forge
micromamba activate literature-rag
pip install -e .
```

An editable installation provides the `literature-rag` command and is recommended for development. To run without installing the package in editable mode:

```bash
pip install -r requirements.txt
python -m literature_rag
```

## Usage

Start the interactive CLI:

```bash
literature-rag
```

Alternative entry points:

```bash
python -m literature_rag
python agentic_literature_rag.py
```

The CLI prompts for the following values:

| Input | Required | Description |
| --- | --- | --- |
| LLM profile | Yes | OpenAI-compatible endpoint, API key, and model. |
| Research topic | Yes | Primary arXiv query and workspace identifier. |
| Analysis objective | Yes | Review scope and guidance for follow-up searches. |
| Local PDF folder | No | Directory containing additional PDF documents. |
| DOIs | No | Comma-separated DOI values or DOI URLs. |
| Unpaywall email | No | Email used for legal open-access discovery through Unpaywall. |

Example session values:

```text
Research topic: retrieval augmented generation evaluation
Specific analysis objective: Compare factuality metrics and identify unresolved evaluation gaps
Local PDF folder [optional]: ./my_papers
DOIs, comma-separated [optional]: 10.1145/example, https://doi.org/10.1000/example
Email for Unpaywall OA lookup [optional]: researcher@example.com
```

## Manual PDF Recovery

When a PDF cannot be retrieved automatically, the CLI displays:

- The paper title.
- The arXiv URL.
- The DOI URL, when available.
- The download failure reason.
- The exact destination path and filename.

Download the PDF legally to the displayed path, then press Enter to validate it and continue. Enter `s` to skip papers that remain unavailable. The application validates the PDF signature, so login pages, paywall pages, corrupted files, and other HTML responses are rejected.

## Persistent Workspaces

Each normalized research topic receives a deterministic workspace:

```text
literature_projects/<topic>-<id>/
├── papers/                 # PDF corpus
├── faiss_index/            # FAISS index and corpus fingerprint
├── output/
│   ├── report.md           # Final literature review
│   ├── draft.md            # Preserved pre-critic draft
│   ├── failure.txt         # Redacted failure details, when generation fails
│   ├── review.json         # Review metadata and reproducibility manifest
│   └── references.bib      # BibTeX bibliography
├── metadata.json           # Scholarly metadata cache
└── search_results.json     # Final search result cache
```

Rerunning the same topic reuses cached searches, downloaded PDFs, metadata, and the FAISS index. Search cache identity includes the objective, search rounds, endpoint, and model. The index is rebuilt automatically when the PDF corpus, embedding model, chunk size, or chunk overlap changes.

## Report Structure

The generated review contains four primary sections:

1. Evidence Matrix.
2. Executive Summary of Main Methodologies.
3. Comparative Analysis: Pros, Cons, and Performance Trade-offs.
4. Unresolved Research Gaps and Blind Spots.

After drafting the review, a critic LLM pass checks unsupported claims, conflicting findings, overgeneralization, missing limitations, and incorrect citations. A deterministic validator rejects citation IDs that do not correspond to retrieved evidence.

## Default Configuration

| Setting | Default |
| --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding device | CPU |
| Chunk size | `1000` characters |
| Chunk overlap | `150` characters |
| Target paper count | `5` |
| Retrieved chunks | `12` |
| Vector store | FAISS |
| Retrieval strategy | MMR |

These values are defined in `literature_rag/settings.py`.

## LLM Configuration

On first use, the application requests:

1. A profile name.
2. A base URL, such as `https://host.example/v1`.
3. An API key.
4. A model name.

Profiles are stored in `.literature_rag_config.json`. API keys are stored separately in the operating-system keyring. On subsequent runs, the CLI provides options to select an existing profile and model, add another endpoint, or add a model to an existing endpoint.

> **Security notice:** If no supported keyring backend is available, the CLI requests the API key for the current session instead of persisting it. Legacy plaintext keys are migrated to the keyring when possible and removed from the configuration file. The configuration file is excluded through `.gitignore`, but it must never be shared or committed.

Before analysis, the CLI displays the selected endpoint host and requires explicit consent. PDF evidence and analysis prompts are then sent to that endpoint. Use only trusted endpoints when processing sensitive, private, or unpublished documents.

## Project Structure

```text
literature_rag/
├── analysis.py       # Retrieval, synthesis, critic pass, and evidence formatting
├── cli.py            # Interactive terminal workflow
├── config.py         # Endpoint, API key, and model profiles
├── export.py         # Markdown, JSON, and BibTeX exports
├── ingestion.py      # PDF parsing, chunking, embeddings, and FAISS
├── papers.py         # arXiv search, downloads, imports, recovery, and deduplication
├── pipeline.py       # End-to-end orchestration
├── resilience.py     # Atomic writes, retries, redaction, and workspace locks
├── scholarly.py      # Crossref, Semantic Scholar, and Unpaywall integration
├── search_agent.py   # Iterative arXiv search planning
├── settings.py       # Application defaults
├── validation.py     # Deterministic citation ID auditing
└── workspace.py      # Persistent topic workspaces
```

Additional entry points and package files:

- `agentic_literature_rag.py`: compatibility launcher.
- `literature_rag/__main__.py`: enables `python -m literature_rag`.
- `pyproject.toml`: package metadata and the `literature-rag` console command.

## Troubleshooting

### A dependency is missing

```text
ModuleNotFoundError: No module named 'langchain_community'
```

Activate the intended virtual environment and reinstall the project:

```bash
pip install -e .
```

### A manually downloaded file is rejected

Confirm that the file is a genuine PDF rather than a publisher login page, paywall page, or HTML error response. Open it locally, then save a valid PDF using the exact destination path shown by the CLI.

### The first embedding run is slow

The embedding model must be downloaded during the first run. Later runs use the local model cache. Embedding currently runs on the CPU.

### The LLM endpoint fails

Verify that the base URL uses `http://` or `https://`, supports the OpenAI API format, exposes the selected model, and accepts the configured API key. Many providers require a `/v1` suffix.

### Cached results do not reflect a new search

Search results and metadata are cached in the topic workspace. To start an independent review, use a distinct topic or remove the relevant workspace after preserving any required output.

### A stale workspace lock remains after forced termination

Normal exits and `Ctrl+C` remove the lock automatically. If the operating system forcibly terminates the process, verify that no `literature-rag` process is using the workspace, then remove its `.lock` file manually.

## Development

Install development dependencies and run all local quality gates:

```bash
pip install -e ".[dev]"
pytest --cov=literature_rag --cov-report=term-missing
ruff format --check .
ruff check .
mypy literature_rag
python -m compileall -q agentic_literature_rag.py literature_rag tests
```

GitHub Actions runs tests and Ruff on Linux, Windows, and macOS with Python 3.10, 3.11, and 3.12. A separate job runs mypy on Python 3.11.

See `CONTRIBUTING.md` for contribution guidance and `SECURITY.md` for vulnerability reporting.

## Limitations

- arXiv is currently the primary discovery source; DOI values and local PDFs are supplemental inputs.
- Citation counts and external metadata depend on third-party API availability and accuracy.
- Review quality depends on PDF quality, retrieval coverage, and the selected LLM.
- Deterministic auditing validates citation IDs; it does not independently prove the semantic truth of every claim.
- The application does not bypass publisher paywalls or access-control mechanisms.
- Resource limits reduce malformed-document risk but do not provide a complete PDF sandbox.
