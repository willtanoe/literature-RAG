import re

REQUIRED_SECTIONS = (
    "Evidence Matrix",
    "Executive Summary of Main Methodologies",
    "Comparative Analysis: Pros, Cons, and Performance Trade-offs",
    "Unresolved Research Gaps and Blind Spots",
)


def audit_citations(report: str, allowed_ids: set[str]) -> str:
    # New delimiter: bullet "•", so match [ID • Title] patterns
    cited_ids = re.findall(r"\[([A-Za-z0-9-]+)\s+\u2022", report)
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
    if "| Paper |" not in report and "|Paper|" not in report.replace(" ", ""):
        errors.append("missing Evidence Matrix table")
    invalid = sorted(set(re.findall(r"\[([A-Za-z0-9-]+)\s+\u2022", report)) - allowed_ids)
    if invalid:
        errors.append(f"invalid citation IDs: {invalid}")
    # Must have at least one citation with new format
    if not re.search(r"\[[A-Za-z0-9-]+\s+\u2022", report):
        errors.append("report contains no evidence citations")
    return errors
