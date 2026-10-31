"""Runtime guardrails built on the Guardrails AI library (guardrails-ai).

Three guards, each a `guardrails.Guard` with locally registered validators (no Hub
token, no remote inference, no model downloads):

- INPUT   (user message, before the agent runs): length, secrets, injection, PII
- UNTRUSTED (web search results, before an agent reads them): instruction-like text, secrets, PII
- OUTPUT  (final answer/report, before it is displayed, stored or exported): non-empty,
           secrets, PII, report structure, citations

Policy (see `Decision.action`):
  pass    - nothing found
  redact  - secrets/PII replaced with [REDACTED:type]; the redacted text is used
  retry   - a report is structurally invalid; the runner asks the agent to repair it
            (bounded by config.MAX_OUTPUT_RETRIES)
  reject  - the text is not used (input: request refused; output: not displayed/exported)
  error   - a validator crashed; treated like reject (fail closed, never bypassed)

These checks reduce risk; they do not guarantee factual accuracy or stop every prompt
injection. Quality scoring lives separately in evaluation.py.
"""

import re
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.classes.validation.validation_result import FailResult, PassResult

from redaction import redact, summarize

# Synchronous validation is intended here (the app is synchronous); silence the per-call notice.
warnings.filterwarnings("ignore", message="Could not obtain an event loop", category=UserWarning)

MAX_INPUT_CHARS = 2000
MIN_INPUT_CHARS = 2

# Instruction-like phrases that try to override the agent's rules. Kept narrow so that
# ordinary research requests ("compare Notion vs Coda", "ignore pricing, focus on
# features") are not flagged.
INJECTION_PATTERNS = [
    r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+|these\s+)*(?:previous|prior|above|earlier|preceding|system|original)\s+"
    r"(?:instructions?|prompts?|rules|directions|guidelines|messages?)",
    r"\b(?:reveal|print|show|output|repeat|leak|dump|display)\s+(?:me\s+)?(?:your|the)\s+(?:full\s+|entire\s+|hidden\s+|original\s+)*"
    r"(?:system\s+prompt|system\s+message|hidden\s+instructions|developer\s+(?:message|prompt)|initial\s+prompt|api\s+keys?|secrets?|environment\s+variables)",
    r"\byou\s+are\s+now\s+(?:dan\b|in\s+developer\s+mode|jailbroken|unrestricted|unfiltered)",
    r"\b(?:jailbreak(?:ing)?\s+(?:mode|prompt)|dan\s+mode|developer\s+mode\s+enabled)\b",
    r"\bdo\s+not\s+(?:follow|obey)\s+(?:your|the|any)\s+(?:rules|guidelines|instructions|system\s+prompt)",
    r"<\|im_start\|>|<\|system\|>|\[/?(?:SYSTEM|INST)\]",
    r"^\s*(?:#{1,3}\s*)?(?:new\s+)?system\s*(?:prompt|instructions?)\s*:",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I | re.M)
UNTRUSTED_REMOVED = "[removed: instruction-like text from an untrusted web page]"


# ---------------------------------------------------------------- validators
@register_validator(name="mra/bounded-length", data_type="string")
class BoundedLength(Validator):
    """Rejects empty/whitespace-only text and text longer than `max_chars`."""

    def __init__(self, min_chars: int = 1, max_chars: int | None = None, **kwargs):
        super().__init__(min_chars=min_chars, max_chars=max_chars, **kwargs)
        self.min_chars, self.max_chars = min_chars, max_chars

    def _validate(self, value: Any, metadata: dict):
        n = len((value or "").strip())
        if n < self.min_chars:
            return FailResult(error_message="empty" if n == 0 else f"too short (minimum {self.min_chars} characters)")
        if self.max_chars and n > self.max_chars:
            return FailResult(error_message=f"too long ({n} characters, maximum {self.max_chars})")
        return PassResult()


@register_validator(name="mra/no-secrets", data_type="string")
class NoSecrets(Validator):
    """Detects credentials (API keys, tokens, private keys, passwords); fix = redact them."""

    def _validate(self, value: Any, metadata: dict):
        fixed, findings = redact(value, pii=False)
        if findings:
            kinds = ", ".join(sorted(summarize(findings)))
            return FailResult(error_message=f"contains credentials ({kinds})", fix_value=fixed)
        return PassResult()


@register_validator(name="mra/no-pii", data_type="string")
class NoPII(Validator):
    """Detects personal data (email, phone, SSN, payment card); fix = redact it."""

    def _validate(self, value: Any, metadata: dict):
        fixed, findings = redact(value, secrets=False)
        if findings:
            kinds = ", ".join(sorted(summarize(findings)))
            return FailResult(error_message=f"contains personal data ({kinds})", fix_value=fixed)
        return PassResult()


@register_validator(name="mra/injection-heuristic", data_type="string")
class InjectionHeuristic(Validator):
    """Flags instruction-override / prompt-extraction phrases; fix = remove the phrase."""

    def _validate(self, value: Any, metadata: dict):
        hits = _INJECTION.findall(value)
        if hits:
            return FailResult(error_message=f"instruction-override pattern ({len(hits)} match(es))",
                              fix_value=_INJECTION.sub(UNTRUSTED_REMOVED, value))
        return PassResult()


_CITE = re.compile(r"\[(\d{1,3})\](?!\()")
_REF_LINE = re.compile(r"^\s*\[(\d{1,3})\]\s+.*?(https?://\S+)", re.M)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)


def _references_section(md: str) -> str:
    m = re.search(r"(?im)^#{1,3}\s*(?:references|sources)\b.*$", md)
    return md[m.end():] if m else ""


def check_report_structure(md: str) -> list[str]:
    """Problems with the report contract described in agents.REPORT_RULES."""
    problems = []
    if not re.search(r"(?m)^#\s+\S", md):
        problems.append("missing the '# <Target>: Competitor Analysis' title")
    if len(_TABLE_ROW.findall(md)) < 4:  # header + separator + target + at least one competitor
        problems.append("missing the comparison table (target row plus competitor rows)")
    if not re.search(r"(?im)^#{1,3}\s*key\s+takeaways", md):
        problems.append("missing the 'Key takeaways' section")
    if not _references_section(md):
        problems.append("missing the 'References' section")
    return problems


def check_citations(md: str) -> list[str]:
    """Every inline [n] must have a reference with a URL, and references must be cited."""
    refs_text = _references_section(md)
    refs = {int(n) for n, _ in _REF_LINE.findall(refs_text)}
    body = md[: len(md) - len(refs_text)] if refs_text else md
    cited = {int(n) for n in _CITE.findall(body)}
    problems = []
    if not refs:
        problems.append("no references with URLs")
    if not cited:
        problems.append("no inline [n] citations")
    if dangling := sorted(cited - refs):
        problems.append(f"citations without a matching reference: {dangling[:10]}")
    return problems


@register_validator(name="mra/report-structure", data_type="string")
class ReportStructure(Validator):
    def _validate(self, value: Any, metadata: dict):
        if problems := check_report_structure(value):
            return FailResult(error_message="; ".join(problems))
        return PassResult()


@register_validator(name="mra/report-citations", data_type="string")
class ReportCitations(Validator):
    def _validate(self, value: Any, metadata: dict):
        if problems := check_citations(value):
            return FailResult(error_message="; ".join(problems))
        return PassResult()


@register_validator(name="mra/answer-citations", data_type="string")
class AnswerCitations(Validator):
    """Advisory: a substantive follow-up answer should cite a source ([n] or a URL)."""

    def _validate(self, value: Any, metadata: dict):
        if len(value.split()) >= 60 and not (_CITE.search(value) or re.search(r"https?://", value)):
            return FailResult(error_message="answer makes claims without citing a source")
        return PassResult()


# ---------------------------------------------------------------- guards
# What each validator's failure means for the overall decision.
REJECT, REDACT, RETRY, ADVISORY = "reject", "redact", "retry", "advisory"


def _guard(name: str, validators: list[tuple[Validator, str]]) -> tuple[Guard, dict[str, str]]:
    guard = Guard(name=name)
    # Guardrails sends usage telemetry to its own servers by default; keep all data local.
    guard.configure(allow_metrics_collection=False)
    guard.use(*(v for v, _ in validators))
    return guard, {type(v).__name__: role for v, role in validators}


def _fix(): return {"on_fail": OnFailAction.FIX}
def _noop(): return {"on_fail": OnFailAction.NOOP}


_GUARDS = {
    "input": _guard("mra-input", [
        (BoundedLength(min_chars=MIN_INPUT_CHARS, max_chars=MAX_INPUT_CHARS, **_noop()), REJECT),
        # A credential in a request is refused outright: it must not reach the model or the logs.
        (NoSecrets(**_noop()), REJECT),
        (InjectionHeuristic(**_noop()), REJECT),
        (NoPII(**_fix()), REDACT),
    ]),
    "untrusted": _guard("mra-untrusted", [
        (InjectionHeuristic(**_fix()), REDACT),
        (NoSecrets(**_fix()), REDACT),
        (NoPII(**_fix()), REDACT),
    ]),
    "report": _guard("mra-output-report", [
        (BoundedLength(min_chars=1, **_noop()), REJECT),
        (NoSecrets(**_fix()), REDACT),
        (NoPII(**_fix()), REDACT),
        (ReportStructure(**_noop()), RETRY),
        (ReportCitations(**_noop()), RETRY),
    ]),
    "answer": _guard("mra-output-answer", [
        (BoundedLength(min_chars=1, **_noop()), REJECT),
        (NoSecrets(**_fix()), REDACT),
        (NoPII(**_fix()), REDACT),
        (AnswerCitations(**_noop()), ADVISORY),
    ]),
}
_LOCK = threading.Lock()  # a Guard keeps per-call history; serialise access across sessions


@dataclass
class Decision:
    action: str  # pass | redact | retry | reject | error
    text: str  # the text to use downstream ("" when rejected)
    reasons: list[str] = field(default_factory=list)
    validators: dict[str, str] = field(default_factory=dict)  # validator -> pass/fail
    advisories: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.action in ("pass", "redact")

    def summary(self) -> dict:
        """Sanitised record for storage and telemetry (no input/output text)."""
        return {"action": self.action, "reasons": self.reasons, "validators": self.validators,
                "advisories": self.advisories}


def validate(stage: str, text: str) -> Decision:
    """Run one guard. Never raises: a validator failure becomes action='error' (fail closed)."""
    guard, roles = _GUARDS[stage]
    try:
        with _LOCK:
            outcome = guard.validate(text if isinstance(text, str) else "")
    except Exception as exc:  # validator crashed or Guardrails failed to run
        return Decision("error", "", [f"guardrail '{stage}' could not run ({type(exc).__name__})"])

    status = {s.validator_name: s.validator_status for s in outcome.validation_summaries}
    validators = {name: status.get(name, "pass") for name in roles}
    failed = [(s.validator_name, roles.get(s.validator_name, REJECT), s.failure_reason or "failed")
              for s in outcome.validation_summaries if s.validator_status == "fail"]
    by_role = {r: [f"{n}: {why}" for n, role, why in failed if role == r]
               for r in (REJECT, RETRY, REDACT, ADVISORY)}
    fixed = outcome.validated_output if isinstance(outcome.validated_output, str) else text

    if by_role[REJECT]:
        return Decision("reject", "", by_role[REJECT] + by_role[REDACT], validators)
    if by_role[RETRY]:
        return Decision("retry", fixed, by_role[RETRY] + by_role[REDACT], validators, by_role[ADVISORY])
    if by_role[REDACT]:
        return Decision("redact", fixed, by_role[REDACT], validators, by_role[ADVISORY])
    return Decision("pass", fixed, [], validators, by_role[ADVISORY])


def sanitize_untrusted(text: str) -> str:
    """Clean text from a web page before an agent reads it. Raises if validation is unavailable."""
    decision = validate("untrusted", text or "")
    if not decision.usable:
        raise RuntimeError("untrusted-content validation unavailable")
    return decision.text
