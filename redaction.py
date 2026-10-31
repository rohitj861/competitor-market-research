"""Detect and redact secrets and personal data in free text.

Shared by the Guardrails validators (guards.py), the Langfuse mask (observability.py)
and the progress log, so every path that shows or transmits text redacts the same way.
Findings never contain the matched value, only its type.
"""

import re
from collections.abc import Iterable

REDACTED = "[REDACTED:{}]"

SECRET_PATTERNS: dict[str, re.Pattern] = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
    "langfuse_key": re.compile(r"\b[ps]k-lf-[A-Za-z0-9-]{8,}"),
    "openai_key": re.compile(r"\bsk-(?:proj-|svcacct-|admin-|ant-)?[A-Za-z0-9_-]{20,}"),
    "tavily_key": re.compile(r"\btvly-(?:dev-|prod-)?[A-Za-z0-9]{16,}"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "stripe_key": re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    # user:password@ inside a connection string; group `v` marks the part that is replaced
    "url_credentials": re.compile(r"://(?P<v>[^\s:/@]+:[^\s@/]+)@"),
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|client[_-]?secret|password|passwd|access[_-]?token|auth[_-]?token)"
        r"[\"']?\s*[:=]\s*[\"']?(?P<v>[^\s\"',;]{8,})"),
}

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "us_ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    # +country-code numbers, or North-American (xxx) xxx-xxxx; deliberately not bare digit
    # runs, which would hit years, prices and revenue figures in research reports.
    "phone": re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,5}){2,4}"
                        r"|\(\d{3}\)\s?\d{3}[\s.-]\d{4}|\b\d{3}[.-]\d{3}[.-]\d{4})\b"),
    "payment_card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
}


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _apply(text: str, patterns: dict[str, re.Pattern], category: str, findings: list[dict]) -> str:
    for kind, pattern in patterns.items():
        def repl(m: re.Match, kind=kind) -> str:
            if kind == "payment_card" and not _luhn_ok(m.group(0)):
                return m.group(0)
            findings.append({"category": category, "type": kind})
            if "v" in pattern.groupindex:  # replace only the credential, keep its context
                start, end = m.start("v") - m.start(), m.end("v") - m.start()
                return m.group(0)[:start] + REDACTED.format(kind) + m.group(0)[end:]
            return REDACTED.format(kind)
        text = pattern.sub(repl, text)
    return text


def redact(text: str, *, secrets: bool = True, pii: bool = True) -> tuple[str, list[dict]]:
    """Return (redacted text, findings). Findings list types only, never values."""
    findings: list[dict] = []
    if secrets:
        text = _apply(text, SECRET_PATTERNS, "secret", findings)
    if pii:
        text = _apply(text, PII_PATTERNS, "pii", findings)
    return text, findings


def redact_text(text: str) -> str:
    return redact(text)[0]


def summarize(findings: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["type"]] = counts.get(f["type"], 0) + 1
    return counts


def redact_data(data):
    """Recursively redact secrets and personal data in JSON-like data (used for telemetry)."""
    if isinstance(data, str):
        return redact(data)[0]
    if isinstance(data, dict):
        return {k: redact_data(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [redact_data(v) for v in data]
    return data
