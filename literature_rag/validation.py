import re

REQUIRED_SECTIONS = (
    "Evidence Matrix",
    "State of the Art: Reported Results",
    "Executive Summary of Main Methodologies",
    "Comparative Analysis: Pros, Cons, and Performance Trade-offs",
    "Unresolved Research Gaps and Blind Spots",
    "Per-Paper Results, Gaps, and Future Work",
)

CITATION_LONG_PATTERN = re.compile(r"\[([A-Za-z0-9-]+)\s*\|")
CITATION_SHORT_PATTERN = re.compile(r"\[\s*(P[0-9a-f]{10}-p\d+-c\d+)\s*\]")
EVIDENCE_TABLE_PATTERN = re.compile(r"\|\s*\**\s*paper\s*\**\s*\|", re.IGNORECASE)


def extract_citation_ids(report: str) -> list[str]:
    return CITATION_LONG_PATTERN.findall(report) + CITATION_SHORT_PATTERN.findall(report)


def audit_citations(report: str, allowed_ids: set[str]) -> str:
    cited_ids = extract_citation_ids(report)
    invalid = sorted(set(cited_ids) - allowed_ids)
    valid = sorted(set(cited_ids) & allowed_ids)
    return (
        f"valid source IDs: {valid or 'none'}; "
        f"invalid source IDs: {invalid or 'none'}; "
        f"total inline citations: {len(cited_ids)}"
    )


def validate_report(report: str, allowed_ids: set[str]) -> list[str]:
    errors = [
        f"missing section: {section}" for section in REQUIRED_SECTIONS if section not in report
    ]
    if not EVIDENCE_TABLE_PATTERN.search(report):
        errors.append("missing Evidence Matrix table")
    cited_ids = set(extract_citation_ids(report))
    invalid = sorted(cited_ids - allowed_ids)
    if invalid:
        errors.append(f"invalid citation IDs: {invalid}")
    if not cited_ids & allowed_ids:
        errors.append("report contains no evidence citations")
    return errors
