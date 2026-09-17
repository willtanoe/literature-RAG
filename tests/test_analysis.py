import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from literature_rag.analysis import (
    PAPER_READ_RETRY_HINT,
    PAPER_TABLE_PLACEHOLDER,
    SOTA_PLACEHOLDER,
    _budgeted_slices,
    _compact_documents,
    _inject_paper_table,
    _inject_sota_table,
    _inject_tables,
    _is_retryable_endpoint_error,
    _merge_paper_reads,
    _paper_batches,
    _parse_paper_read,
    _parse_paper_summaries,
    _parse_sota,
    _read_papers,
    _reading_order,
    _render_paper_table,
    _render_sota_table,
    _salvage_paper_read,
    _tier_read_budget,
    answer_question,
    build_gap_register,
    build_thesis_candidates,
    classify_relevance,
    generate_analysis,
)
from literature_rag.config import LLMConfig
from literature_rag.papers import Paper

EVIDENCE_ID = "Pabcdef1234-p1-c1"


def _valid_report(tag: str) -> str:
    return (
        "## 1. Evidence Matrix\n\n"
        "| Paper | Venue/Year | Dataset or Sample | Method | Metrics | Main Finding | "
        f"| Limitations | Evidence Pages |\n|---|---|---|---|---|---|---|---|\n"
        f"| A | 2025 | X | Y | Z | R [{EVIDENCE_ID} | A | p. 1] | None | 1 |\n\n"
        "## 2. State of the Art: Reported Results\n"
        f"Narrative interpreting results. [{EVIDENCE_ID}]\n\n{SOTA_PLACEHOLDER}\n\n"
        "## 3. Executive Summary of Main Methodologies\n"
        f"Summary [{EVIDENCE_ID}].\n\n"
        "## 4. Comparative Analysis: Pros, Cons, and Performance Trade-offs\n"
        f"Comparison [{EVIDENCE_ID}].\n\n"
        "## 5. Unresolved Research Gaps and Blind Spots\n"
        f"Gap [{EVIDENCE_ID}].\n\n"
        "## 6. Per-Paper Results, Gaps, and Future Work\n"
        f"Introductory sentence. [{EVIDENCE_ID}]\n\n{PAPER_TABLE_PLACEHOLDER}\n\n"
        f"{tag}\n"
    )


BROKEN_REPORT = (
    "# Report without any evidence citations but all sections present\n\n"
    "## 1. Evidence Matrix\n\nProse only.\n\n"
    "## 2. State of the Art: Reported Results\n\nProse only.\n\n"
    "## 3. Executive Summary of Main Methodologies\n\nProse only.\n\n"
    "## 4. Comparative Analysis: Pros, Cons, and Performance Trade-offs\n\nProse only.\n\n"
    "## 5. Unresolved Research Gaps and Blind Spots\n\nProse only.\n\n"
    "## 6. Per-Paper Results, Gaps, and Future Work\n\nProse only.\n"
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.document = Document(
            page_content="evidence text",
            metadata={
                "evidence_id": EVIDENCE_ID,
                "paper_id": EVIDENCE_ID.split("-")[0],
                "title": "Paper A",
                "section": "results",
                "chunk_index": 1,
                "has_metrics": True,
                "has_future_work": True,
            },
        )
        self.docstore = self
        self.index_to_docstore_id = {0: "d0"}

    def search(self, document_id: str):
        return self.document

    def as_retriever(self, **_kwargs):
        return self

    def invoke(self, _query):
        return [self.document]

    def similarity_search(self, _query, k=1, filter=None):
        return [self.document]


PAPER_READ_JSON = (
    "{"
    '"what_was_done": "Trained DP-FedAvg on IDS traffic",'
    '"datasets": "NSL-KDD",'
    '"key_results": "95.8% accuracy",'
    '"stated_limitations": "No adaptive budget",'
    '"stated_future_work": "Explore adaptive eps",'
    f'"evidence_ids": ["{EVIDENCE_ID}"],'
    '"results": [{"dataset": "NSL-KDD", "method": "FedAvg-DP",'
    ' "conditions": "eps=2", "metrics": [{"name": "Accuracy", "value": "95.8%"}],'
    f' "evidence_id": "{EVIDENCE_ID}"}}]'
    "}"
)


def _config() -> LLMConfig:
    return LLMConfig("https://llm.example/v1", "secret-key", "model")


def _unpack(analysis):
    return (
        analysis.report,
        analysis.sota_results,
        analysis.paper_summaries,
        analysis.thesis_proposals,
    )


def test_repair_round_restores_citations(tmp_path, monkeypatch):
    script = [
        AIMessage(content=PAPER_READ_JSON),
        AIMessage(content=_valid_report("draft")),
        AIMessage(content=BROKEN_REPORT),
        AIMessage(content=_valid_report("repaired")),
        AIMessage(content="# Thesis Directions: Combining Future Work Across Papers\n"),
    ]
    monkeypatch.setattr(
        "literature_rag.analysis.ChatOpenAI",
        lambda **_kwargs: FakeMessagesListChatModel(responses=list(script)),
    )
    report, sota_results, paper_summaries, directions = _unpack(
        generate_analysis(
            FakeVectorStore(),
            "topic",
            "objective",
            _config(),
            3,
            draft_path=tmp_path / "draft.md",
        )
    )
    assert "repaired" in report
    assert "[REDACTED]" not in report
    assert len(sota_results) == 1
    assert sota_results[0]["metrics"] == [{"name": "Accuracy", "value": "95.8%"}]
    assert len(paper_summaries) == 1
    assert paper_summaries[0]["stated_future_work"] == "Explore adaptive eps"
    assert "Explore adaptive eps" in report
    assert "95.8%" in report
    assert "Thesis Directions" in directions


def test_invalid_revision_falls_back_to_validated_draft(tmp_path, monkeypatch):
    script = [
        AIMessage(content=PAPER_READ_JSON),
        AIMessage(content=_valid_report("draft")),
        AIMessage(content=BROKEN_REPORT),
        AIMessage(content=BROKEN_REPORT),
        AIMessage(content=BROKEN_REPORT),
        AIMessage(content="# Thesis Directions: Combining Future Work Across Papers\n"),
    ]
    monkeypatch.setattr(
        "literature_rag.analysis.ChatOpenAI",
        lambda **_kwargs: FakeMessagesListChatModel(responses=list(script)),
    )
    report = generate_analysis(
        FakeVectorStore(),
        "topic",
        "objective",
        _config(),
        3,
        draft_path=tmp_path / "draft.md",
    ).report
    assert "draft" in report
    attempt = (tmp_path / "critic_attempt.md").read_text()
    assert "without any evidence citations" in attempt


def test_invalid_everywhere_raises(tmp_path, monkeypatch):
    script = [AIMessage(content=PAPER_READ_JSON)] + [
        AIMessage(content=BROKEN_REPORT) for _ in range(4)
    ]
    monkeypatch.setattr(
        "literature_rag.analysis.ChatOpenAI",
        lambda **_kwargs: FakeMessagesListChatModel(responses=list(script)),
    )
    try:
        generate_analysis(
            FakeVectorStore(),
            "topic",
            "objective",
            _config(),
            3,
            draft_path=tmp_path / "draft.md",
        )
    except RuntimeError as exc:
        assert "invalid report" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def _scripted_llm(monkeypatch, script, calls):
    def call(prompt_value):
        text = prompt_value.to_string()
        stage = (
            "read"
            if "read ONE research paper" in text
            else "triage"
            if "triage candidate papers" in text
            else "critic"
            if "exacting academic reviewer" in text
            else "repair"
            if "repair literature review reports" in text
            else "proposals"
            if "thesis advisor" in text
            else "draft"
        )
        action, payload = script.pop(0)
        calls.append(stage)
        if action == "raise":
            raise payload
        return AIMessage(content=payload)

    monkeypatch.setattr(
        "literature_rag.analysis.ChatOpenAI",
        lambda **_kwargs: RunnableLambda(call),
    )


def test_retryable_critic_error_is_retried_once(tmp_path, monkeypatch):
    calls: list[str] = []
    sleeps: list[float] = []
    script = [
        ("return", PAPER_READ_JSON),
        ("return", _valid_report("draft")),
        ("raise", RuntimeError("Error code: 524 - origin timeout")),
        ("return", _valid_report("retried")),
        ("return", "# Thesis Directions: Combining Future Work Across Papers\n"),
    ]
    _scripted_llm(monkeypatch, script, calls)
    monkeypatch.setattr(
        "literature_rag.analysis.time.sleep", lambda seconds: sleeps.append(seconds)
    )
    report = generate_analysis(
        FakeVectorStore(),
        "topic",
        "objective",
        _config(),
        3,
        draft_path=tmp_path / "draft.md",
    ).report
    assert "retried" in report
    assert calls == ["read", "draft", "critic", "critic", "proposals"]
    assert sleeps == [5, 5, 5]


def test_non_retryable_critic_error_falls_back_without_retry(tmp_path, monkeypatch):
    calls: list[str] = []
    sleeps: list[float] = []
    script = [
        ("return", PAPER_READ_JSON),
        ("return", _valid_report("draft")),
        ("raise", ValueError("Error code: 400 - bad request")),
        ("return", "# Thesis Directions: Combining Future Work Across Papers\n"),
    ]
    _scripted_llm(monkeypatch, script, calls)
    monkeypatch.setattr(
        "literature_rag.analysis.time.sleep", lambda seconds: sleeps.append(seconds)
    )
    report = generate_analysis(
        FakeVectorStore(),
        "topic",
        "objective",
        _config(),
        3,
        draft_path=tmp_path / "draft.md",
    ).report
    assert "draft" in report
    assert calls == ["read", "draft", "critic", "proposals"]
    assert sleeps == []


def test_thesis_proposals_failure_is_non_fatal(tmp_path, monkeypatch):
    calls: list[str] = []
    script = [
        ("return", PAPER_READ_JSON),
        ("return", _valid_report("draft")),
        ("return", _valid_report("draft")),
        ("raise", RuntimeError("boom")),
    ]
    _scripted_llm(monkeypatch, script, calls)
    analysis = generate_analysis(
        FakeVectorStore(),
        "topic",
        "objective",
        _config(),
        3,
        draft_path=tmp_path / "draft.md",
    )
    assert "draft" in analysis.report
    assert calls == ["read", "draft", "critic", "proposals"]
    assert "unavailable" in analysis.thesis_proposals


def test_answer_question_uses_qa_prompt(tmp_path, monkeypatch):
    calls: list[str] = []
    script = [("return", f"Answer with citation [{EVIDENCE_ID}]")]
    _scripted_llm(monkeypatch, script, calls)
    answer = answer_question(FakeVectorStore(), "what datasets?", _config(), 3)
    assert EVIDENCE_ID in answer
    assert calls == ["draft"]


def test_connection_reset_is_retryable():
    assert _is_retryable_endpoint_error(
        OSError("[WinError 10054] An existing connection was forcibly closed by the remote host")
    )
    assert _is_retryable_endpoint_error(ConnectionError("connection aborted"))
    assert _is_retryable_endpoint_error(RuntimeError("remote end closed connection"))
    assert _is_retryable_endpoint_error(RuntimeError("Error code: 524 - origin timeout"))
    assert _is_retryable_endpoint_error(RuntimeError("Error code: 429 - rate limited"))
    assert not _is_retryable_endpoint_error(ValueError("Error code: 400 - bad request"))
    assert not _is_retryable_endpoint_error(ValueError("invalid prompt format"))


SOTA_JSON = f"""```json
[
  {{"paper": "Paper A", "dataset": "NSL-KDD", "method": "DP-FedAvg",
    "conditions": "10 clients, eps=4", "metrics": [{{"name": "Accuracy", "value": "97.2%"}}],
    "evidence_id": "{EVIDENCE_ID}"}},
  {{"paper": "Paper B", "dataset": "CICIDS2017", "method": "FedSGD",
    "conditions": "5 clients", "metrics": [{{"name": "F1", "value": "0.91"}}],
    "evidence_id": "Pbadbadbad0-p9-c9"}},
  {{"paper": "Paper C", "dataset": "UNSW-NB15", "method": "Local DP",
    "conditions": "not reported", "metrics": [], "evidence_id": "{EVIDENCE_ID}"}}
]
```"""


def test_parse_sota_drops_invalid_evidence_and_empty_metrics():
    rows = _parse_sota(SOTA_JSON, {EVIDENCE_ID})
    assert len(rows) == 1
    row = rows[0]
    assert row["paper"] == "Paper A"
    assert row["dataset"] == "NSL-KDD"
    assert row["metrics"] == [{"name": "Accuracy", "value": "97.2%"}]
    assert row["evidence_id"] == EVIDENCE_ID


def test_render_sota_table_includes_source_column():
    rows = _parse_sota(SOTA_JSON, {EVIDENCE_ID})
    table = _render_sota_table(rows)
    assert table.startswith("| Paper | Dataset | Method / Settings | Reported Metrics | Source |")
    assert "Paper A" in table
    assert "Accuracy: 97.2%" in table
    assert f"[{EVIDENCE_ID}]" in table


def test_render_sota_table_empty_is_honest():
    table = _render_sota_table([])
    assert "No validated quantitative results" in table


def test_inject_sota_table_replaces_placeholder():
    report = (
        f"## 2. State of the Art: Reported Results\n\nNarrative.\n\n{SOTA_PLACEHOLDER}\n\n"
        "## 3. Next"
    )
    injected = _inject_sota_table(report, "| Paper |\n|---|")
    assert SOTA_PLACEHOLDER not in injected
    assert "| Paper |" in injected


def test_inject_sota_table_inserts_after_heading_when_placeholder_missing():
    report = (
        "## 2. State of the Art: Reported Results\n\nNarrative only.\n\n"
        "## 3. Executive Summary of Main Methodologies\n"
    )
    injected = _inject_sota_table(report, "| Paper |\n|---|")
    assert injected.index("| Paper |") < injected.index("## 3.")


def test_inject_sota_table_appends_when_no_heading():
    injected = _inject_sota_table("Plain report.", "| Paper |\n|---|")
    assert injected.endswith("| Paper |\n|---|\n")


PAPER_JSON = f"""[
  {{"paper": "Paper A", "what_was_done": "Trained DP-FedAvg on IDS traffic",
    "datasets": "NSL-KDD", "key_results": "97.2% accuracy",
    "stated_limitations": "No adaptive budget", "stated_future_work": "Explore adaptive eps",
    "evidence_ids": ["{EVIDENCE_ID}", "Pbadbadbad0-p9-c9"]}},
  {{"paper": "Paper B", "what_was_done": "Proposed a backdoor defense",
    "datasets": "CIFAR-10", "key_results": "not reported",
    "stated_limitations": "not reported", "stated_future_work": "not reported",
    "evidence_ids": ["Pbadbadbad0-p9-c9"]}}
]"""


def test_parse_paper_summaries_drops_rows_without_valid_evidence():
    rows = _parse_paper_summaries(PAPER_JSON, {EVIDENCE_ID})
    assert len(rows) == 1
    row = rows[0]
    assert row["paper"] == "Paper A"
    assert row["evidence_ids"] == [EVIDENCE_ID]
    assert row["stated_future_work"] == "Explore adaptive eps"


def test_render_paper_table_includes_all_columns_and_source():
    rows = _parse_paper_summaries(PAPER_JSON, {EVIDENCE_ID})
    table = _render_paper_table(rows)
    assert table.startswith("| Paper | Tier | What Was Done | Data / Datasets | Key Results | ")
    assert "Explore adaptive eps" in table
    assert f"[{EVIDENCE_ID}]" in table


def test_render_paper_table_empty_is_honest():
    table = _render_paper_table([])
    assert "No validated per-paper summaries" in table


def test_inject_paper_table_replaces_placeholder():
    report = (
        f"## 6. Per-Paper Results, Gaps, and Future Work\n\nIntro.\n\n{PAPER_TABLE_PLACEHOLDER}\n"
    )
    injected = _inject_paper_table(report, "| Paper |\n|---|")
    assert PAPER_TABLE_PLACEHOLDER not in injected
    assert "| Paper |" in injected


def test_inject_tables_replaces_both_placeholders():
    report = (
        f"## 2. State of the Art: Reported Results\n\n{SOTA_PLACEHOLDER}\n\n"
        f"## 6. Per-Paper Results, Gaps, and Future Work\n\n{PAPER_TABLE_PLACEHOLDER}\n"
    )
    injected = _inject_tables(report, "SOTA-TABLE", "PAPER-TABLE")
    assert "SOTA-TABLE" in injected
    assert "PAPER-TABLE" in injected
    assert SOTA_PLACEHOLDER not in injected
    assert PAPER_TABLE_PLACEHOLDER not in injected


def test_paper_batches_group_five_papers_and_prioritize_keyword_chunks():
    documents = []
    for paper_index in range(6):
        paper_id = f"P{paper_index:010x}"
        documents.extend(
            [
                Document(
                    page_content="generic introduction",
                    metadata={"paper_id": paper_id, "evidence_id": f"{paper_id}-p1-c1"},
                ),
                Document(
                    page_content="future work limitations conclusion result accuracy",
                    metadata={"paper_id": paper_id, "evidence_id": f"{paper_id}-p9-c2"},
                ),
            ]
        )
    batches = _paper_batches(documents, ("future work", "limitation", "accuracy"))
    assert len(batches) == 2
    assert len({item.metadata["paper_id"] for item in batches[0]}) == 5
    assert any("future work" in item.page_content for item in batches[0])


def _read_document(paper_id: str, index: int, section: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "paper_id": paper_id,
            "evidence_id": f"{paper_id}-p{index}-c{index}",
            "title": "Paper A",
            "section": section,
            "chunk_index": index,
            "has_future_work": section == "future_work",
            "has_metrics": section == "results",
        },
    )


def test_reading_order_prioritizes_future_work_and_results():
    documents = [
        _read_document("P1", 1, "introduction", "intro"),
        _read_document("P1", 2, "results", "accuracy 95%"),
        _read_document("P1", 3, "future_work", "future work adaptive"),
        _read_document("P1", 4, "related_work", "related"),
    ]
    ordered = _reading_order(documents)
    assert [item.metadata["section"] for item in ordered][:2] == ["future_work", "results"]
    assert ordered[-1].metadata["section"] == "related_work"


def test_budgeted_slices_respect_char_budget_and_slice_cap():
    documents = [_read_document("P1", index, "body", "x" * 9_000) for index in range(1, 8)]
    slices = _budgeted_slices(documents)
    assert 1 <= len(slices) <= 2
    total_chars = sum(len(item.page_content) for group in slices for item in group)
    assert total_chars <= 24_000


def test_parse_paper_read_drops_invalid_evidence():
    raw = (
        "{"
        '"what_was_done": "did work",'
        '"datasets": "NSL-KDD",'
        '"key_results": "95%",'
        '"stated_limitations": "none",'
        '"stated_future_work": "extend",'
        '"evidence_ids": ["P1-p1-c1", "PBAD-p9-c9"],'
        '"results": ['
        '{"dataset": "NSL-KDD", "method": "DP", "conditions": "eps=1",'
        ' "metrics": [{"name": "Accuracy", "value": "95%"}], "evidence_id": "P1-p1-c1"},'
        '{"dataset": "X", "method": "Y", "conditions": "z",'
        ' "metrics": [{"name": "F1", "value": "0.9"}], "evidence_id": "PBAD-p9-c9"}'
        "]}"
    )
    parsed = _parse_paper_read(raw, {"P1-p1-c1"})
    assert parsed["evidence_ids"] == ["P1-p1-c1"]
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["evidence_id"] == "P1-p1-c1"


def test_merge_paper_reads_fills_missing_fields_and_unions_results():
    first = {
        "what_was_done": "part one",
        "datasets": "not reported",
        "key_results": "not reported",
        "stated_limitations": "not reported",
        "stated_future_work": "not reported",
        "evidence_ids": ["P1-p1-c1"],
        "results": [
            {
                "dataset": "A",
                "method": "M",
                "conditions": "c",
                "metrics": [],
                "evidence_id": "P1-p1-c1",
            }
        ],
    }
    second = {
        "what_was_done": "part two",
        "datasets": "NSL-KDD",
        "key_results": "95%",
        "stated_limitations": "small scale",
        "stated_future_work": "extend to IDS",
        "evidence_ids": ["P1-p9-c9"],
        "results": [
            {
                "dataset": "B",
                "method": "N",
                "conditions": "d",
                "metrics": [],
                "evidence_id": "P1-p9-c9",
            }
        ],
    }
    merged = _merge_paper_reads([first, second])
    assert merged["what_was_done"] == "part one"
    assert merged["datasets"] == "NSL-KDD"
    assert merged["stated_future_work"] == "extend to IDS"
    assert merged["evidence_ids"] == ["P1-p1-c1", "P1-p9-c9"]
    assert len(merged["results"]) == 2


CLASSIFY_JSON = """[
  {"index": 0, "tier": "core", "reason": "directly addresses the objective"},
  {"index": 1, "tier": "peripheral", "reason": "shares only background"},
  {"index": 2, "tier": "excluded", "reason": "off topic"},
  {"index": 9, "tier": "core", "reason": "index not in this batch"}
]"""


def _candidate(identifier: str, abstract: str) -> Paper:
    return Paper(identifier, f"Paper {identifier}", (), f"e{identifier}", "p", abstract=abstract)


def test_classify_relevance_assigns_tiers_and_caches_decisions(tmp_path, monkeypatch):
    calls: list[str] = []
    _scripted_llm(monkeypatch, [("return", CLASSIFY_JSON)], calls)
    papers = [
        _candidate("1", "differential privacy for federated intrusion detection"),
        _candidate("2", "generic background on neural networks"),
        _candidate("3", "unrelated astrophysics survey"),
        Paper("4", "No abstract", (), "e4", "p"),
    ]
    cache = tmp_path / "classification.json"

    classified = classify_relevance(papers, "topic", "objective", _config(), cache)

    assert [paper.tier for paper in classified] == ["core", "peripheral", "excluded", "related"]
    assert calls == ["triage"]
    assert json.loads(cache.read_text())["papers"]["1"]["tier"] == "core"

    resumed = classify_relevance(papers, "topic", "objective", _config(), cache)

    assert [paper.tier for paper in resumed] == [paper.tier for paper in classified]
    assert calls == ["triage"]


def test_classify_relevance_keeps_papers_when_triage_fails(tmp_path, monkeypatch):
    calls: list[str] = []
    _scripted_llm(monkeypatch, [("raise", RuntimeError("Error code: 400 - bad"))], calls)
    papers = [_candidate("1", "differential privacy federated learning")]

    classified = classify_relevance(
        papers, "topic", "objective", _config(), tmp_path / "classification.json"
    )

    assert [paper.tier for paper in classified] == ["related"]
    assert calls == ["triage"]


GAP_SUMMARIES = [
    {
        "paper": "A",
        "tier": "core",
        "what_was_done": "trained",
        "datasets": "NSL-KDD",
        "key_results": "95%",
        "stated_limitations": "Static privacy budget limits adaptive noise calibration",
        "stated_future_work": "not reported",
        "evidence_ids": ["P1-p1-c1"],
    },
    {
        "paper": "B",
        "tier": "related",
        "what_was_done": "measured",
        "datasets": "CICIDS2017",
        "key_results": "0.91 F1",
        "stated_limitations": "not reported",
        "stated_future_work": "Explore adaptive noise calibration under privacy budget limits",
        "evidence_ids": ["P2-p2-c2"],
    },
    {
        "paper": "C",
        "tier": "peripheral",
        "what_was_done": "surveyed",
        "datasets": "not reported",
        "key_results": "not reported",
        "stated_limitations": "Static privacy budget limits adaptive noise calibration",
        "stated_future_work": "not reported",
        "evidence_ids": ["P3-p3-c3"],
    },
]


def test_gap_register_covers_only_core_and_related_tiers():
    register = build_gap_register(GAP_SUMMARIES)

    assert [(row["paper"], row["kind"]) for row in register] == [
        ("A", "limitation"),
        ("B", "future_work"),
    ]
    assert "calibration" in register[0]["keywords"]
    assert "under" not in register[1]["keywords"]


def test_thesis_candidates_score_cross_paper_pairings_only():
    candidates = build_thesis_candidates(build_gap_register(GAP_SUMMARIES))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["papers"] == ["A", "B"]
    assert candidate["kinds"] == ["limitation", "future_work"]
    assert candidate["score"] == 15.0
    assert candidate["evidence_ids"] == ["P1-p1-c1", "P2-p2-c2"]
    assert build_thesis_candidates(build_gap_register(GAP_SUMMARIES[:1])) == []


def test_tier_read_budget_shrinks_for_peripheral_papers():
    assert _tier_read_budget("core") == (2, 24_000)
    assert _tier_read_budget("related") == (2, 24_000)
    assert _tier_read_budget("peripheral") == (1, 12_000)
    assert _tier_read_budget("excluded")[0] == 0
    documents = [_read_document("P1", index, "body", "x" * 9_000) for index in range(1, 5)]
    assert len(_budgeted_slices(documents, 1, 12_000)) == 1
    assert _budgeted_slices(documents, 0) == []


class TieredVectorStore:
    def __init__(self) -> None:
        self.documents = [
            _tier_document("Pdddddddddd", "Peripheral Paper", "peripheral", "body"),
            _tier_document("Pcccccccccc", "Core Paper", "core", "results"),
        ]
        self.docstore = self
        self.index_to_docstore_id = {0: "d0", 1: "d1"}

    def search(self, document_id: str):
        return self.documents[0 if document_id == "d0" else 1]

    def as_retriever(self, **_kwargs):
        return self

    def invoke(self, _query):
        return list(self.documents)

    def similarity_search(self, _query, k=1, filter=None):
        return [self.documents[0]]


def _tier_document(paper_id: str, title: str, tier: str, section: str) -> Document:
    return Document(
        page_content=f"{tier} evidence",
        metadata={
            "evidence_id": f"{paper_id}-p1-c1",
            "paper_id": paper_id,
            "title": title,
            "tier": tier,
            "section": section,
            "chunk_index": 1,
        },
    )


def test_reads_core_papers_before_peripheral_papers(tmp_path, monkeypatch, capsys):
    calls: list[str] = []
    core_id = "Pcccccccccc-p1-c1"
    read_json = PAPER_READ_JSON.replace(EVIDENCE_ID, core_id)
    report = _valid_report("draft").replace(EVIDENCE_ID, core_id)
    script = [
        ("return", read_json),
        ("return", read_json),
        ("return", report),
        ("return", report),
        ("return", "# Thesis Proposals: Combining Stated Gaps Across Papers\n"),
    ]
    _scripted_llm(monkeypatch, script, calls)

    analysis = generate_analysis(
        TieredVectorStore(),
        "topic",
        "objective",
        _config(),
        3,
        draft_path=tmp_path / "draft.md",
    )

    output = capsys.readouterr().out
    assert output.index("reading core paper 1/2") < output.index("reading peripheral paper 2/2")
    assert calls == ["read", "read", "draft", "critic", "proposals"]
    tiers = {row["paper"]: row["tier"] for row in analysis.paper_summaries}
    assert tiers == {"Core Paper": "core", "Peripheral Paper": "peripheral"}


class _TwoSliceStore:
    """One core paper large enough to be read in two slices."""

    def __init__(self) -> None:
        self.documents = [
            Document(
                page_content="x" * 9_000,
                metadata={
                    "evidence_id": f"Pssssssssss-p{index}-c{index}",
                    "paper_id": "Pssssssssss",
                    "title": "Sliced Paper",
                    "tier": "core",
                    "section": "results" if index == 1 else "future_work",
                    "chunk_index": index,
                },
            )
            for index in (1, 2)
        ]
        self.docstore = self
        self.index_to_docstore_id = {0: "d0", 1: "d1"}

    def search(self, document_id: str):
        return self.documents[0 if document_id == "d0" else 1]


def _replay_llm(replies: list[str], prompts: list[str]):
    def call(prompt_value):
        prompts.append(prompt_value.to_string())
        return AIMessage(content=replies.pop(0))

    return RunnableLambda(call)


SLICE_READ_JSON = PAPER_READ_JSON.replace(EVIDENCE_ID, "Pssssssssss-p2-c2")
REFUSAL_REPLY = "I'm sorry, but I can't help with that request."


def test_unparsable_slice_is_retried_with_a_json_only_reminder():
    prompts: list[str] = []
    llm = _replay_llm([REFUSAL_REPLY, SLICE_READ_JSON, SLICE_READ_JSON], prompts)

    summaries, sota_rows, _ = _read_papers(llm, _TwoSliceStore(), _config())

    assert len(prompts) == 3
    assert PAPER_READ_RETRY_HINT.strip() in prompts[1]
    assert PAPER_READ_RETRY_HINT.strip() not in prompts[2]
    assert len(summaries) == 1
    assert summaries[0]["stated_future_work"] == "Explore adaptive eps"
    assert len(sota_rows) == 2


def test_failed_first_slice_still_reads_the_remaining_slice(capsys):
    prompts: list[str] = []
    llm = _replay_llm([REFUSAL_REPLY, REFUSAL_REPLY, SLICE_READ_JSON], prompts)

    summaries, _, _ = _read_papers(llm, _TwoSliceStore(), _config())

    assert len(prompts) == 3
    assert len(summaries) == 1
    assert summaries[0]["paper"] == "Sliced Paper"
    output = capsys.readouterr().out
    assert "1 failed slice(s)" in output
    assert "0 paper(s) lost" in output


def test_read_failure_writes_the_raw_reply_for_diagnosis(tmp_path, capsys):
    prompts: list[str] = []
    llm = _replay_llm([REFUSAL_REPLY] * 4, prompts)
    failure_dir = tmp_path / "read_failures"
    stale = failure_dir / "Pold-part1.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("previous run", encoding="utf-8")

    summaries, _, _ = _read_papers(llm, _TwoSliceStore(), _config(), failure_dir)

    assert summaries == []
    assert not stale.exists()
    recorded = sorted(path.name for path in failure_dir.glob("*.txt"))
    assert recorded == ["Pssssssssss-part1.txt", "Pssssssssss-part2.txt"]
    body = (failure_dir / "Pssssssssss-part1.txt").read_text(encoding="utf-8")
    assert REFUSAL_REPLY in body
    assert "no JSON object found" in body
    assert "1 paper(s) lost" in capsys.readouterr().out


def test_read_failure_redacts_the_api_key(tmp_path):
    prompts: list[str] = []
    llm = _replay_llm(["leaked secret-key in the reply"] * 4, prompts)
    failure_dir = tmp_path / "read_failures"

    _read_papers(llm, _TwoSliceStore(), _config(), failure_dir)

    body = (failure_dir / "Pssssssssss-part1.txt").read_text(encoding="utf-8")
    assert "secret-key" not in body
    assert "[REDACTED]" in body


TRUNCATED_READ_JSON = (
    '{"what_was_done": "Trained DP-FedAvg on IDS traffic",'
    '"datasets": "NSL-KDD",'
    '"key_results": "95.8% accuracy",'
    '"stated_limitations": "No adaptive budget",'
    '"stated_future_work": "Explore adaptive eps",'
    '"evidence_ids": ["Pssssssssss-p2-c2"],'
    '"results": [{"dataset": "NSL-KDD", "method": "FedAvg-DP", "conditions": "eps=2'
)

COMMENTED_READ_JSON = (
    "Here is the summary:\n{\n"
    "  // notes that make this object unparsable\n"
    '  "what_was_done": "Trained DP-FedAvg on IDS traffic",\n'
    '  "datasets": "NSL-KDD",\n'
    '  "key_results": "95.8% accuracy",\n'
    '  "stated_limitations": "No adaptive budget",\n'
    '  "stated_future_work": "Explore adaptive eps",\n'
    '  "evidence_ids": ["Pssssssssss-p2-c2", "Pbadbadbad0-p9-c9"]\n}\n'
)


def test_truncated_json_is_closed_and_parsed():
    parsed = _parse_paper_read(TRUNCATED_READ_JSON, {"Pssssssssss-p2-c2"})

    assert parsed["datasets"] == "NSL-KDD"
    assert parsed["stated_future_work"] == "Explore adaptive eps"
    assert parsed["evidence_ids"] == ["Pssssssssss-p2-c2"]
    assert parsed["results"] == []


def test_salvage_recovers_fields_from_unparsable_json():
    parsed = _salvage_paper_read(COMMENTED_READ_JSON, {"Pssssssssss-p2-c2"})

    assert parsed["what_was_done"] == "Trained DP-FedAvg on IDS traffic"
    assert parsed["key_results"] == "95.8% accuracy"
    assert parsed["evidence_ids"] == ["Pssssssssss-p2-c2"]


def test_salvage_rejects_a_reply_without_any_readable_field():
    try:
        _salvage_paper_read(REFUSAL_REPLY, {"Pssssssssss-p2-c2"})
    except ValueError as exc:
        assert "no readable per-paper fields" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_salvaged_slice_is_kept_and_recorded(tmp_path, capsys):
    prompts: list[str] = []
    llm = _replay_llm([COMMENTED_READ_JSON] * 4, prompts)
    failure_dir = tmp_path / "read_failures"

    summaries, _, _ = _read_papers(llm, _TwoSliceStore(), _config(), failure_dir)

    assert len(summaries) == 1
    assert summaries[0]["datasets"] == "NSL-KDD"
    output = capsys.readouterr().out
    assert "salvaged fields from an unparsable reply" in output
    assert "1 paper(s) salvaged" in output
    assert (failure_dir / "Pssssssssss-part1.txt").is_file()


def _context_document(paper_id: str, tier: str, index: int, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "paper_id": paper_id,
            "evidence_id": f"{paper_id}-p{index}-c{index}",
            "tier": tier,
            "chunk_index": index,
        },
    )


def test_compact_documents_weights_core_papers_above_peripheral_ones():
    documents = [
        _context_document("Pcore", "core", 1, "accuracy result table"),
        _context_document("Pcore", "core", 2, "future work limitation"),
        _context_document("Pcore", "core", 3, "unrelated prose"),
        _context_document("Pperi", "peripheral", 1, "accuracy result table"),
        _context_document("Pperi", "peripheral", 2, "future work limitation"),
    ]

    compacted = _compact_documents(documents)

    picked = [(item.metadata["paper_id"], item.metadata["chunk_index"]) for item in compacted]
    assert picked == [("Pcore", 1), ("Pcore", 2), ("Pperi", 1)]


def test_compact_documents_stops_at_the_character_budget():
    documents = [_context_document(f"P{index}", "core", 1, "result " * 200) for index in range(10)]

    compacted = _compact_documents(documents, budget_chars=4_000)

    assert 0 < len(compacted) < 10
    assert sum(len(item.page_content) for item in compacted) <= 4_000


def test_compact_documents_keeps_one_chunk_when_the_budget_is_tiny():
    documents = [_context_document("Pone", "core", 1, "x" * 5_000)]

    assert len(_compact_documents(documents, budget_chars=10)) == 1
