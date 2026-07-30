# Security Policy

## Supported Versions

Security fixes are applied to the latest version on the default branch.

## Reporting a Vulnerability

Do not open a public issue for an unpatched vulnerability. Use GitHub's private vulnerability reporting feature for this repository and include:

- The affected version or commit.
- Reproduction steps or a minimal proof of concept.
- The expected and observed behavior.
- The potential impact.
- Any suggested mitigation.

Avoid including real API keys, private papers, personal data, or other secrets. Acknowledgment and remediation timing depend on severity and reproducibility.

## Security Boundaries

- API keys are stored in the operating-system keyring when available.
- Document evidence is sent to the user-selected LLM endpoint only after explicit confirmation.
- External PDFs and scholarly metadata are untrusted inputs.
- PDF downloads require HTTPS and enforce content and size limits.
- FAISS persistence uses a native index and JSON metadata; pickle deserialization is not used.
- Logs and failure artifacts redact the configured API key.

Users remain responsible for endpoint trust, document-processing authorization, publisher terms, and local workstation security.
