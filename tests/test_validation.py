from literature_rag.validation import audit_citations, validate_report

REPORT = """# Evidence Matrix
| Paper | Venue/Year | Dataset or Sample | Method | Metrics | Main Finding |
| Limitations | Evidence Pages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 2025 | X | Y | Z | Result [Pabc-p1-c1 | A | p. 1] | None | 1 |
# Executive Summary of Main Methodologies
Summary [Pabc-p1-c1 | A | p. 1].
# Comparative Analysis: Pros, Cons, and Performance Trade-offs
Comparison [Pabc-p1-c1 | A | p. 1].
# Unresolved Research Gaps and Blind Spots
Gap [Pabc-p1-c1 | A | p. 1].
"""


def test_valid_report_and_citations():
    allowed = {"Pabc-p1-c1"}
    assert validate_report(REPORT, allowed) == []
    assert "invalid source IDs: none" in audit_citations(REPORT, allowed)


def test_invalid_citation_is_rejected():
    errors = validate_report(REPORT.replace("Pabc", "Pbad"), {"Pabc-p1-c1"})
    assert any("invalid citation IDs" in error for error in errors)
