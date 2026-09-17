from literature_rag.validation import audit_citations, validate_report

REPORT = """# Evidence Matrix
| Paper | Venue/Year | Dataset or Sample | Method | Metrics | Main Finding |
| Limitations | Evidence Pages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 2025 | X | Y | Z | Result [Pabc-p1-c1 • Title • p. 1] | None | 1 |
# State of the Art: Reported Results
Results table narrative [Pabc-p1-c1 | A | p. 1].
# Executive Summary of Main Methodologies
Summary [Pabc-p1-c1 • Title • p. 1].
# Comparative Analysis: Pros, Cons, and Performance Trade-offs
Comparison [Pabc-p1-c1 • Title • p. 1].
# Unresolved Research Gaps and Blind Spots
Gap [Pabc-p1-c1 • Title • p. 1].
# Per-Paper Results, Gaps, and Future Work
Summary row [Pabc-p1-c1 | A | p. 1].
"""


def test_valid_report_and_citations():
    allowed = {"Pabc-p1-c1"}
    assert validate_report(REPORT, allowed) == []
    assert "invalid source IDs: none" in audit_citations(REPORT, allowed)


def test_invalid_citation_is_rejected():
    errors = validate_report(REPORT.replace("Pabc", "Pbad"), {"Pabc-p1-c1"})
    assert any("invalid citation IDs" in error for error in errors)


SHORT_REPORT = REPORT.replace("Pabc-p1-c1", "Pabcdef1234-p1-c1")
SHORT_REPORT = SHORT_REPORT.replace(" • Title • p. 1", "").replace(" | A | p. 1", "")


def test_short_form_citations_are_accepted():
    allowed = {"Pabcdef1234-p1-c1"}
    assert validate_report(SHORT_REPORT, allowed) == []
    audit = audit_citations(SHORT_REPORT, allowed)
    assert "total inline citations: 6" in audit
    assert "invalid source IDs: none" in audit


def test_unrelated_brackets_are_not_citations():
    plain = SHORT_REPORT.replace("[Pabcdef1234-p1-c1]", "[OK]")
    errors = validate_report(plain, {"Pabcdef1234-p1-c1"})
    assert any("no evidence citations" in error for error in errors)


def test_mixed_forms_are_audited_together():
    mixed = REPORT + "\nExtra claim [Pabcdef1234-p1-c1]."
    audit = audit_citations(mixed, {"Pabc-p1-c1", "Pabcdef1234-p1-c1"})
    assert "invalid source IDs: none" in audit


def test_evidence_matrix_header_variants_are_accepted():
    allowed = {"Pabc-p1-c1"}
    bold = REPORT.replace("| Paper |", "| **Paper** |")
    lower = REPORT.replace("| Paper |", "| paper |")
    assert validate_report(REPORT, allowed) == []
    assert validate_report(bold, allowed) == []
    assert validate_report(lower, allowed) == []


def test_missing_evidence_matrix_table_is_still_rejected():
    broken = REPORT.replace("| Paper | Venue/Year |", "| Study | Venue/Year |")
    errors = validate_report(broken, {"Pabc-p1-c1"})
    assert any("missing Evidence Matrix table" in error for error in errors)
