"""Langfuse tracing, scores and feedback (optional; the app works without it).

One trace per user request: a root observation wraps the agent run and the Langfuse
LangChain callback records every model call, tool call and subagent under it, with
timings, model names and token usage as reported by the provider (nothing is estimated
here). `session_id = thread_id` groups a conversation's traces into one Langfuse session.
A request that pauses for human approval keeps its trace id and later segments join it.

Every payload passes through a mask that redacts secrets and personal data before it
leaves the process. All calls are best-effort: a Langfuse outage is logged and never
breaks research or chat.
"""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import config
from redaction import redact_data

log = logging.getLogger("mra.observability")


def _mask(*, data, **kwargs):
    try:
        return redact_data(data)
    except Exception:  # never transmit unmasked data
        return "[masked: redaction failed]"


def new_trace_id() -> str:
    """32 lowercase hex chars, the trace id format Langfuse (OpenTelemetry) expects."""
    return uuid.uuid4().hex


@dataclass
class TurnTrace:
    trace_id: str
    span: object | None = None

    def update(self, **fields) -> None:
        """Set output/metadata on the root observation (no-op when tracing is off)."""
        if self.span is None:
            return
        try:
            self.span.update(**fields)
        except Exception as exc:
            log.warning("Langfuse update failed: %s", type(exc).__name__)

    def event(self, name: str, metadata: dict | None = None, level: str = "DEFAULT") -> None:
        if self.span is None:
            return
        try:
            self.span.create_event(name=name, metadata=metadata or {}, level=level)
        except Exception as exc:
            log.warning("Langfuse event failed: %s", type(exc).__name__)


class Tracer:
    def __init__(self, settings: config.LangfuseSettings | None = None, **client_kwargs):
        self.settings = settings or config.langfuse_settings()
        self.client = None
        self._health: tuple[float, bool] | None = None
        if not self.settings.enabled:
            return
        try:
            from langfuse import Langfuse

            self.client = Langfuse(public_key=self.settings.public_key, secret_key=self.settings.secret_key,
                                   base_url=self.settings.base_url, mask=_mask, timeout=5, **client_kwargs)
        except Exception as exc:  # misconfigured URL, incompatible SDK, ...
            log.warning("Langfuse disabled: client failed to start (%s)", type(exc).__name__)
            self.settings = config.LangfuseSettings(False, f"unavailable: client failed to start ({type(exc).__name__})")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def status(self, check: bool = False) -> tuple[str, str]:
        """('enabled'|'unavailable'|'disabled', human-readable detail)."""
        if not self.enabled:
            return "disabled", self.settings.reason
        if check and not self.healthy():
            return "unavailable", (f"cannot reach {self.settings.base_url} or the keys were rejected; "
                                   "research continues without tracing")
        return "enabled", f"sending traces to {self.settings.base_url}"

    def healthy(self, max_age: float = 300) -> bool:
        """Cached auth check (one network call at most every `max_age` seconds)."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if self._health and now - self._health[0] < max_age:
            return self._health[1]
        try:
            ok = bool(self.client.auth_check())
        except Exception:
            ok = False
        self._health = (now, ok)
        return ok

    def callbacks(self) -> list:
        """The LangChain callback that traces the agent graph (one per run, never duplicated)."""
        if not self.enabled:
            return []
        try:
            from langfuse.langchain import CallbackHandler

            return [CallbackHandler(public_key=self.settings.public_key)]
        except Exception as exc:
            log.warning("Langfuse callback unavailable: %s", type(exc).__name__)
            return []

    @contextmanager
    def turn(self, *, trace_id: str, name: str, session_id: str, user_id: str, input=None,
             metadata: dict | None = None, tags: list[str] | None = None):
        """Root observation for one request segment; spans created inside join the trace."""
        if not self.enabled:
            yield TurnTrace(trace_id)
            return
        from langfuse import propagate_attributes

        try:
            span_cm = self.client.start_as_current_observation(
                trace_context={"trace_id": trace_id}, name=name, as_type="agent", input=input,
                metadata=metadata)
            span = span_cm.__enter__()
            attrs_cm = propagate_attributes(session_id=session_id, user_id=user_id, trace_name=name, tags=tags,
                                            metadata={k: str(v)[:200] for k, v in (metadata or {}).items()})
            attrs_cm.__enter__()
        except Exception as exc:
            log.warning("Langfuse trace start failed: %s", type(exc).__name__)
            yield TurnTrace(trace_id)
            return
        try:
            yield TurnTrace(trace_id, span)
        except BaseException as exc:
            try:
                span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])
            except Exception:
                pass
            raise
        finally:
            for cm in (attrs_cm, span_cm):
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
            self.flush_in_background()

    def score(self, *, trace_id: str | None, name: str, value, data_type: str, comment: str | None = None,
              score_id: str | None = None, metadata: dict | None = None, session_id: str | None = None,
              dataset_run_id: str | None = None) -> bool:
        """Attach a score to a trace (or session). Same score_id = update, so re-scoring never duplicates."""
        if not self.enabled or not (trace_id or session_id):
            return False
        try:
            self.client.create_score(trace_id=trace_id, session_id=None if trace_id else session_id, name=name,
                                     value=value, data_type=data_type, comment=comment, score_id=score_id,
                                     metadata=metadata, dataset_run_id=dataset_run_id)
            return True
        except Exception as exc:
            log.warning("Langfuse score '%s' failed: %s", name, type(exc).__name__)
            return False

    def flush(self) -> None:
        if self.enabled:
            try:
                self.client.flush()
            except Exception as exc:
                log.warning("Langfuse flush failed: %s", type(exc).__name__)

    def flush_in_background(self) -> None:
        """Export without blocking the UI (an unreachable Langfuse would otherwise stall a rerun)."""
        if self.enabled:
            threading.Thread(target=self.flush, name="langfuse-flush", daemon=True).start()

    def shutdown(self) -> None:
        if self.enabled:
            try:
                self.client.shutdown()
            except Exception:
                pass
