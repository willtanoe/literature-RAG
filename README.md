# Agentic Literature Review and RAG

A cross-platform CLI for discovering academic literature on arXiv, managing research PDFs, building a persistent FAISS index with local embeddings, and generating evidence-grounded literature reviews through any OpenAI-compatible LLM endpoint.

The application maintains a dedicated workspace for each research topic. Search results, PDFs, scholarly metadata, and vector indexes are reused across sessions, reducing repeated downloads and embedding work. Final reviews include an evidence matrix, page-level citations, citation auditing, and bibliography exports.

## Key Features

- Up-front search planning: a single planning request turns the topic and objective into a set of complementary arXiv queries (core topic, methods, benchmarks, evaluation, limitations, competing approaches), which the CLI prints for confirmation before anything is downloaded.
- Deterministic query hygiene: planned queries are stripped of Boolean syntax, field prefixes, and filler, capped to four keywords for recall, and deduplicated by keyword set, then extended by a relaxation ladder until the target corpus size is reached.
- Relevance triage into tiers: every candidate is classified from its title and abstract as `core`, `related`, `peripheral`, or `excluded`. Excluded papers are never downloaded, and decisions are cached per topic and objective.
- Tier-separated corpus folders: PDFs are filed under `papers/core`, `papers/related`, and `papers/peripheral`, and PDFs already present anywhere in the workspace are reused instead of downloaded again.
- Tier-aware reading budget: core and related papers get two reading requests each, peripheral papers one, so coverage stays sharp while token use stays bounded. Core papers are read first, so a failure late in a run costs the least important papers.
- Deterministic gap register: limitations and future work stated by core and related papers are collected verbatim with their evidence labels, then paired across different papers and scored by tier, keyword overlap, and gap type to produce ranked thesis candidates.
- Adaptive arXiv discovery: gap-aware planning rounds followed by deterministic query relaxation to neighboring concepts until the target corpus size is reached (core-topic papers first, related work fills the quota).
- Section-aware chunking that labels each chunk (abstract, method, results, limitations, conclusion, future work) and flags chunks containing metrics or future-work statements.
- Full-text per-paper reading: every paper is read individually in section-priority order under a character budget, so results, limitations, and future work are covered rather than sampled by retrieval alone.
- Configurable target paper count (up to 80) and publication year-range filtering.
- Interactive follow-up Q&A over the indexed corpus after each review.
- Guided manual recovery with automatic download retries when a PDF cannot be fetched.
- Paper ingestion from arXiv, DOI lists, and local PDF directories.
- Scholarly metadata enrichment through Crossref and Semantic Scholar, with optional Semantic Scholar API key support (set the `S2_API_KEY` environment variable or enter it once at startup) and offline year inference from arXiv identifiers.
- Legal open-access PDF discovery through Unpaywall.
- Guided manual recovery when a PDF cannot be downloaded automatically.
- PDF signature validation and deduplication by DOI, arXiv ID, and file hash.
- Reference-list page filtering during ingestion so evidence stays focused on substantive content.
- Startup endpoint health check that validates the selected model before any downloads or indexing.
- Run logging to `output/run.log` in each workspace for post-mortem debugging.
- Local CPU embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Persistent FAISS indexes with automatic invalidation when the corpus changes.
- Safe FAISS persistence using a native index plus JSON documents, without pickle loading.
- Bounded HTTPS downloads, PDF/page/chunk limits, retries, and rate-limit handling.
- Atomic state writes and per-workspace locking for interruption and concurrency safety.
- Streaming LLM calls with live elapsed-time and character progress, so long analysis stages stay visible instead of appearing stuck.
- Automatic retry for retryable endpoint errors (timeouts and 5xx) across draft, critic, and repair calls, with a validated-draft fallback when the critic fails.
- Per-paper reading requests are budgeted (24,000 characters and two requests for core and related papers, 12,000 characters and one request for peripheral papers), with partial-failure recovery so one slow paper does not abort the review.
- A live heartbeat is shown before the first streamed token, and retry countdowns honor endpoint `retry_after` guidance.
- MMR retrieval for broader evidence coverage.
- Evidence matrices covering methods, datasets, metrics, findings, and limitations.
- Inline source labels containing paper titles and PDF page numbers.
- A critic pass that reviews claims, conflicting findings, limitations, and citations.
- Markdown, JSON, and BibTeX exports, plus a search plan, a tier-aware paper manifest, and ranked thesis proposals.
- Previous report artifacts are archived under `output/archive/<timestamp>/` at the start of each run instead of being overwritten.
- Multiple saved LLM endpoints, API keys, and models.

## Pipeline

```text
Research topic and objective
            |
            v
Search plan (up-front queries) + CLI confirmation
            |
            v
Plan-driven arXiv search + DOI and local PDF import
            |
            v
Crossref / Semantic Scholar / Unpaywall enrichment
            |
            v
Abstract-based tier triage (core / related / peripheral / excluded)
            |
            v
Download into tier folders, manual recovery, validation, deduplication
            |
            v
Local embeddings + persistent FAISS index
            |
            v
Tier-aware per-paper reading + MMR retrieval
            |
            v
Deterministic gap register + scored thesis candidates
            |
            v
Evidence-grounded synthesis + critic pass + citation audit
            |
            v
Markdown + JSON + BibTeX + plan, manifest, and proposal exports
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
| Analysis objective | Yes | Review scope and guidance for follow-up searches. Must be at least 10 characters, because it drives retrieval and every analysis prompt. |
| Target paper count | No | Number of papers to target (default 5, max 80). Evidence retrieval scales with this setting. |
| Year range filter | No | For example `2020-2025`, `2023`, or `2022-`. |
| Local PDF folder | No | Directory containing additional PDF documents. |
| DOIs | No | Comma-separated DOI values or DOI URLs. |
| Unpaywall email | No | Email used for legal open-access discovery through Unpaywall. |
| Semantic Scholar API key | No | Optional key for higher metadata rate limits; stored in the OS keyring or read from the `S2_API_KEY` environment variable. |

Example session values:

```text
Research topic: retrieval augmented generation evaluation
Specific analysis objective: Compare factuality metrics and identify unresolved evaluation gaps
Local PDF folder [optional]: ./my_papers
DOIs, comma-separated [optional]: 10.1145/example, https://doi.org/10.1000/example
Email for Unpaywall OA lookup [optional]: researcher@example.com
```

After the endpoint consent prompt, the CLI prints the planned queries with their tier and intent and waits for confirmation:

```text
Search plan (llm) for 80 target paper(s):
  1. [core] retrieval augmented generation evaluation  -- core topic
  2. [related] factuality metrics hallucination benchmark  -- evaluation and metrics
Run this search plan? [Y/n]:
```

Answering `n` ends the run before any paper is downloaded. Pressing Enter accepts the plan, which is then cached and reused on later runs of the same topic and objective.

## Manual PDF Recovery

When a PDF cannot be retrieved automatically, the CLI displays:

- The paper title.
- The arXiv URL.
- The DOI URL, when available.
- The download failure reason.
- The exact destination path and filename.

Download the PDF legally to the displayed path, then press Enter to validate it and continue. Enter also retries the automatic download, which resolves transient network failures without manual work. Enter `s` to skip papers that remain unavailable. The application validates the PDF signature, so login pages, paywall pages, corrupted files, and other HTML responses are rejected.

## Persistent Workspaces

Each normalized research topic receives a deterministic workspace:

```text
literature_projects/<topic>-<id>/
├── papers/                     # PDF corpus, filed by relevance tier
│   ├── core/                   # Central papers, two reading requests each
│   ├── related/                # Neighboring work, two reading requests each
│   └── peripheral/             # Background only, one reading request each
├── faiss_index/                # FAISS index and corpus fingerprint
├── output/
│   ├── report.md               # Final literature review
│   ├── draft.md                # Preserved pre-critic draft
│   ├── failure.txt             # Redacted failure details, when generation fails
│   ├── run.log                 # Append-only log of every pipeline run
│   ├── review.json             # Review metadata, gap register, and manifest
│   ├── references.bib          # BibTeX bibliography
│   ├── papers_manifest.json    # Per-paper tier, folder, and metadata
│   ├── search_plan.json        # Executed search plan
│   ├── thesis_proposals.md     # Ranked, evidence-grounded thesis proposals
│   └── archive/<timestamp>/    # Previous run's report artifacts
├── classification.json         # Cached tier decisions per topic and objective
├── metadata.json               # Scholarly metadata cache
├── search_plan.json            # Cached search plan
└── search_results.json         # Final search result cache
```

Rerunning the same topic reuses the cached search plan, searches, tier decisions, downloaded PDFs, metadata, and the FAISS index. Search cache identity includes the objective, the search plan signature, endpoint, and model; tier decisions are keyed by topic and objective. PDFs are located recursively, so papers already filed under a tier folder are never downloaded twice. The index is rebuilt automatically when the PDF corpus, tier assignments, embedding model, chunk size, or chunk overlap changes.

## Report Structure

The generated review contains six primary sections:

1. Evidence Matrix.
2. State of the Art: Reported Results, including a deterministically rendered table of quantitative results extracted from the evidence. Every row carries an evidence citation, and rows without a valid evidence label are discarded, so no unverifiable numbers are shown.
3. Executive Summary of Main Methodologies.
4. Comparative Analysis: Pros, Cons, and Performance Trade-offs.
5. Unresolved Research Gaps and Blind Spots, grounded in the deterministic gap register built from limitations and future work stated by core and related papers.
6. Per-Paper Results, Gaps, and Future Work: a deterministically rendered table with one row per paper and its relevance tier, reporting what each paper did, its datasets, key results, and the limitations and future work stated by that paper itself. Rows without a valid evidence citation are discarded.

Extracted results and per-paper summaries are also exported machine-readably as `sota_results` and `paper_summaries` in `review.json`, alongside the `gap_register` and the scored `thesis_candidates`. A separate `thesis_proposals.md` turns the highest-scoring cross-paper pairings into concrete thesis proposals, each with the gap addressed, what to combine, feasibility, a method sketch, an evaluation plan, and the expected contribution, ending with an evidence-grounded research agenda. `papers_manifest.json` records the tier and folder of every indexed paper, and `search_plan.json` records the queries the run actually executed. To convert the Markdown report to DOCX, run `pandoc report.md -o report.docx`.

After drafting the review, a critic LLM pass checks unsupported claims, conflicting findings, overgeneralization, missing limitations, and incorrect citations. A deterministic validator rejects citation IDs that do not correspond to retrieved evidence.

## Default Configuration

| Setting | Default |
| --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding device | CPU |
| Chunk size | `1000` characters |
| Chunk overlap | `150` characters |
| Target paper count | `5` (maximum `80`) |
| Planned queries | `8`, at most `4` keywords each |
| Reading requests per paper | `2` core, `2` related, `1` peripheral |
| Triage batch size | `20` papers per request, `400` abstract characters |
| Gap register rows | `60`, with `12` scored thesis candidates |
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

The same menu also provides `Remove model` and `Remove endpoint` actions. Removing a model keeps its endpoint available for other models. Removing an endpoint deletes all models under it and attempts to remove its stored API key from the operating-system keyring. Both actions require confirmation and immediately refresh the menu.

> **Security notice:** If no supported keyring backend is available, the CLI requests the API key for the current session instead of persisting it. Legacy plaintext keys are migrated to the keyring when possible and removed from the configuration file. The configuration file is excluded through `.gitignore`, but it must never be shared or committed.

Before analysis, the CLI displays the selected endpoint host and requires explicit consent. PDF evidence and analysis prompts are then sent to that endpoint. Use only trusted endpoints when processing sensitive, private, or unpublished documents.

## Project Structure

```text
literature_rag/
├── analysis.py       # Triage, retrieval, reading, gap register, synthesis, critic pass
├── cli.py            # Interactive terminal workflow and search-plan confirmation
├── config.py         # Endpoint, API key, and model profiles
├── export.py         # Markdown, JSON, BibTeX, manifest, and proposal exports
├── ingestion.py      # PDF parsing, chunking, embeddings, and FAISS
├── papers.py         # arXiv search, downloads, imports, tier filing, and deduplication
├── pipeline.py       # End-to-end orchestration
├── resilience.py     # Atomic writes, retries, redaction, and workspace locks
├── scholarly.py      # Crossref, Semantic Scholar, and Unpaywall integration
├── search_agent.py   # Up-front search planning and plan execution
├── settings.py       # Application defaults
├── validation.py     # Deterministic citation ID auditing
└── workspace.py      # Persistent topic workspaces, tier folders, and output archiving
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

LLM requests use a 120-second read timeout per attempt. Retryable failures are attempted twice. If an endpoint returns `retry_after`, the application waits for that duration (capped at 120 seconds) while displaying a countdown. Per-paper reading sends at most two requests for core and related papers and one for peripheral papers, each capped at roughly 12,000 characters, to avoid oversized requests and Cloudflare 524 timeouts. These limits are defined in `literature_rag/settings.py` as `PAPER_READ_BUDGET_CHARS`, `PAPER_READ_SLICE_CHARS`, `PAPER_READ_MAX_SLICES`, and `TIER_READ_SLICES`.

### Cached results do not reflect a new search

Search results, tier decisions, and metadata are cached in the topic workspace. Editing the objective invalidates the search and triage caches, because both are keyed by it. To start an independent review, use a distinct topic or remove the relevant workspace after preserving any required output.

### Too many papers land in one tier

Tier decisions come from the title and abstract only and are cached in `classification.json`. Delete that file to re-triage the same corpus, or sharpen the analysis objective, which is what the triage prompt judges relevance against.

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
- Relevance tiers are judged from titles and abstracts by the configured LLM, so a wrong tier can under-read a relevant paper. Tiers are cached and can be reset by deleting `classification.json`.
- Papers supplied through DOI lists or local folders are treated as core, because they are hand-picked rather than discovered.
- Citation counts and external metadata depend on third-party API availability and accuracy.
- Review quality depends on PDF quality, retrieval coverage, and the selected LLM.
- Deterministic auditing validates citation IDs; it does not independently prove the semantic truth of every claim.
- The application does not bypass publisher paywalls or access-control mechanisms.
- Resource limits reduce malformed-document risk but do not provide a complete PDF sandbox.
