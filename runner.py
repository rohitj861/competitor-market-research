"""Runs one conversation turn end-to-end, independent of the Streamlit UI.

    user text -> input guardrail -> run lock + transcript -> agent (Postgres checkpoint,
    Langfuse trace) -> [HITL pause/resume] -> output guardrail (+ bounded repair) ->
    transcript -> deterministic evaluation scores

The database is the source of truth: the UI only renders what this module stored.
"""

import hashlib
import logging
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

from langgraph.types import Command

import config
import evaluation
import guards
from agents import REPORT_PATH, build_agent
from memory import ConversationStore, DatabaseError, Run, RunConflict
from observability import Tracer, new_trace_id
from redaction import redact_text

log = logging.getLogger("mra.runner")

DEFAULT_TITLE = "New conversation"
REPAIR_PROMPT = """[Automated validation] The final report failed these checks:
{problems}
Fix {path} so it satisfies every rule (keep the content and sources; do not invent references).
For a citation without a matching reference: add that reference with its URL from the research
notes, or remove the citation if the notes have no such source.
Then reply with ONLY the corrected report."""


# ---------------------------------------------------------------- text helpers
def file_text(fd) -> str | None:
    if fd is None:
        return None
    if isinstance(fd, dict):
        content = fd.get("content")
        return "\n".join(content) if isinstance(content, list) else content
    return str(fd)


def msg_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content or ""


def tidy_markdown(md: str) -> str:
    """Normalise the agent's Markdown so it renders cleanly in Streamlit and the PDF."""
    md = md.strip()
    md = re.sub(r"^```(?:markdown|md)?\s*\n(.*)\n```$", r"\1", md, flags=re.S)  # unwrap fenced reply
    # images would make the viewer's browser fetch URLs taken from untrusted pages: keep the alt text
    md = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", md)
    lines = md.split("\n")
    out = []
    for line in lines:
        # keep each "[n] source" reference on its own line
        if re.match(r"^\s*\[\d+\]\s", line) and out and out[-1].strip():
            out.append("")
        # tables and lists need a blank line before them to render
        if (line.lstrip().startswith("|") or re.match(r"^\s*([-*+]|\d+[.)])\s", line)) and out \
                and out[-1].strip() and not re.match(r"^\s*(\||[-*+]\s|\d+[.)]\s)", out[-1]) \
                and not out[-1].startswith((" ", "\t")):
            out.append("")
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)) + "\n"


def progress_lines(namespace: tuple, update) -> list[str]:
    """Turn streamed graph updates into human-readable progress lines (redacted)."""
    if not isinstance(update, dict):
        return []
    msgs = update.get("messages")
    if not isinstance(msgs, list):
        return []
    who = "Orchestrator" if not namespace else "Subagent"
    lines = []
    for msg in msgs:
        for call in getattr(msg, "tool_calls", None) or []:
            args = call.get("args", {})
            name = call.get("name")
            if name == "task":
                lines.append(f"🧭 **{who}** → delegating to `{args.get('subagent_type')}`: "
                             f"{args.get('description', '')[:200]}")
            elif name == "internet_search":
                lines.append(f"🔎 {who} searching: *{args.get('query')}*")
            elif name in ("write_file", "edit_file"):
                lines.append(f"📝 {who} writing `{args.get('file_path')}`")
            elif name == "write_todos":
                lines.append(f"🗒️ {who} updated its plan")
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) == "task":
            lines.append(f"✅ Subagent finished: {msg_text(msg)[:200]}")
    # progress is shown before final-output validation, so it is redacted here
    return [redact_text(line) for line in lines]


RESUME_PREFIX = "[Resume after interruption] "


def turn_finished(snapshot) -> bool:
    """A turn is complete only when the agent's last message is a reply with no pending tool calls.

    A cut-off stream can leave a checkpoint with nothing scheduled (`next == ()`), so the
    absence of pending work is not proof that the turn finished.
    """
    messages = snapshot.values.get("messages") or []
    last = messages[-1] if messages else None
    return not snapshot.next and getattr(last, "type", None) == "ai" and not getattr(last, "tool_calls", None)


def _digest(text: str | None) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


# ---------------------------------------------------------------- results
@dataclass
class Outcome:
    status: str  # completed | awaiting_review | failed | rejected | busy | cancelled
    run_id: str | None = None
    message: dict | None = None  # stored assistant message
    pending: dict | None = None  # HITL interrupt payload
    error: str | None = None
    notices: list[str] = field(default_factory=list)


@dataclass
class ThreadView:
    conversation: object
    messages: list[dict]
    active_run: Run | None
    latest_run: Run | None
    pending: dict | None  # interrupt awaiting approval, restored from the checkpoint


class Runner:
    def __init__(self, store: ConversationStore, tracer: Tracer, *, agent_factory: Callable | None = None,
                 search_client=None, max_output_retries: int = config.MAX_OUTPUT_RETRIES):
        self.store = store
        self.tracer = tracer
        self.agent_factory = agent_factory
        self.search_client = search_client
        self.max_output_retries = max_output_retries

    # -- agent ---------------------------------------------------------------
    def _agent(self, settings: dict):
        factory = self.agent_factory or build_agent
        return factory(depth=settings.get("depth", "standard"), hitl=bool(settings.get("hitl", True)),
                       model=settings.get("model") or config.DEFAULT_MODEL,
                       checkpointer=self.store.saver, on_search_results=self._record_evidence,
                       search_client=self.search_client)

    def _record_evidence(self, run_config: dict, query: str, results: list[dict]) -> None:
        run_id = (run_config or {}).get("configurable", {}).get("app_run_id")
        if run_id:
            self.store.add_evidence(run_id, results)

    def _state(self, agent, thread_id: str):
        return agent.get_state({"configurable": {"thread_id": thread_id}})

    # -- read ----------------------------------------------------------------
    def view(self, owner: str, thread_id: str) -> ThreadView | None:
        conv = self.store.get(owner, thread_id)
        if conv is None:
            return None
        active = self.store.active_run(thread_id)
        pending = None
        if active and active.status == "awaiting_review":
            snapshot = self._state(self._agent(active.settings), thread_id)
            pending = snapshot.interrupts[0].value if snapshot.interrupts else None
        return ThreadView(conv, self.store.messages(thread_id), active, self.store.latest_run(thread_id), pending)

    # -- write ---------------------------------------------------------------
    def submit(self, owner: str, thread_id: str, text: str, submission_id: str, settings: dict,
               on_event: Callable[[str], None] | None = None, trace_id: str | None = None) -> Outcome:
        """Handle a new user message. Re-submitting the same submission_id never runs twice."""
        conv = self.store.get(owner, thread_id)
        if conv is None:
            raise PermissionError("Conversation not found for this user.")
        trace_id = trace_id or new_trace_id()
        decision = guards.validate("input", text)

        if not decision.usable:
            return self._reject_input(owner, conv, text, submission_id, settings, decision, trace_id)

        agent = self._agent(settings)
        state = self._state(agent, thread_id).values
        report = file_text((state.get("files") or {}).get(REPORT_PATH))
        # The agent has not seen any request yet (a refused first message never reached it).
        first_turn = not any(getattr(m, "type", None) == "human" for m in state.get("messages") or [])
        prompt = f"Research the competitors of: {decision.text}" if first_turn else decision.text
        try:
            run, created = self.store.start_run(thread_id, submission_id, settings={**settings, "prompt": prompt},
                                                trace_id=trace_id, report_before=_digest(report))
        except RunConflict as exc:
            return Outcome("busy", error=str(exc))
        if not created:  # duplicate submission (double click, rerun): report the existing run
            return self._existing(run)

        self.store.add_message(thread_id, run.run_id, "user", decision.text, guardrail=decision.summary(),
                               trace_id=trace_id)
        if conv.title == DEFAULT_TITLE:
            self.store.rename(owner, thread_id, decision.text[:60])
        self.store.touch(thread_id, settings)
        notices = ["Personal data in your message was redacted before it was sent."] \
            if decision.action == "redact" else []
        out = self._execute(owner, run, {"messages": [{"role": "user", "content": prompt}]}, on_event,
                            input_decision=decision, input_text=decision.text)
        out.notices = notices + out.notices
        return out

    def review(self, owner: str, thread_id: str, decisions: list[dict],
               on_event: Callable[[str], None] | None = None) -> Outcome:
        """Resume a run paused for human approval with approve/edit/reject decisions."""
        run = self._owned_active(owner, thread_id)
        if run is None or run.status != "awaiting_review":
            return Outcome("busy", error="There is no step waiting for approval in this conversation.")
        if not self.store.transition(run.run_id, "awaiting_review", "running"):
            return Outcome("busy", error="This approval was already submitted (maybe in another tab).")
        return self._execute(owner, run, Command(resume={"decisions": decisions}), on_event)

    def retry(self, owner: str, thread_id: str, on_event: Callable[[str], None] | None = None) -> Outcome:
        """Continue a failed or interrupted run from its last checkpoint (no completed step re-runs)."""
        run = self.store.latest_run(thread_id)
        if self.store.get(owner, thread_id) is None or run is None:
            return Outcome("busy", error="Nothing to retry.")
        if not (run.status == "failed" or run.stale):
            return Outcome("busy", error="The last run did not fail; nothing to retry.")
        try:
            if not self.store.transition(run.run_id, run.status, "running"):
                return Outcome("busy", error="The run was already restarted (maybe in another tab).")
        except RunConflict as exc:
            return Outcome("busy", error=str(exc))
        snapshot = self._state(self._agent(run.settings), thread_id)
        human = [m for m in snapshot.values.get("messages", []) if getattr(m, "type", None) == "human"]
        prompt = run.settings["prompt"]
        started = bool(human) and msg_text(human[-1]).removeprefix(RESUME_PREFIX) == prompt
        if snapshot.next:
            payload = None  # pending work in the checkpoint: resume it
        elif started and turn_finished(snapshot):
            payload = "finalize"  # the graph finished; only post-processing failed
        elif started:  # cut off with nothing scheduled: ask the agent to carry on from its saved state
            payload = {"messages": [{"role": "user", "content": RESUME_PREFIX + prompt}]}
        else:
            payload = {"messages": [{"role": "user", "content": prompt}]}  # never started
        return self._execute(owner, run, payload, on_event)

    def cancel(self, owner: str, thread_id: str) -> Outcome:
        """Abandon a paused, failed or interrupted run. Its unfinished steps are not executed."""
        run = self._owned_active(owner, thread_id) or self.store.latest_run(thread_id)
        if run is None or self.store.get(owner, thread_id) is None:
            return Outcome("cancelled")
        if run.status == "running" and not run.stale:
            return Outcome("busy", error="The run is still working; it cannot be cancelled from here.")
        if run.status in ("awaiting_review", "running", "failed"):
            self.store.transition(run.run_id, run.status, "cancelled")
        return Outcome("cancelled", run_id=run.run_id)

    # -- internals -----------------------------------------------------------
    def _owned_active(self, owner: str, thread_id: str) -> Run | None:
        if self.store.get(owner, thread_id) is None:
            return None
        return self.store.active_run(thread_id)

    def _existing(self, run: Run) -> Outcome:
        msgs = [m for m in self.store.messages(run.thread_id) if m["run_id"] == run.run_id and m["role"] == "assistant"]
        status = {"awaiting_review": "awaiting_review", "running": "busy", "completed": "completed",
                  "rejected": "rejected", "cancelled": "cancelled"}.get(run.status, "failed")
        return Outcome(status, run.run_id, msgs[0] if msgs else None, error=run.error,
                       notices=["This message was already submitted; showing its result."])

    def _reject_input(self, owner, conv, text, submission_id, settings, decision, trace_id) -> Outcome:
        try:
            run, created = self.store.start_run(conv.thread_id, submission_id, settings=settings, trace_id=trace_id,
                                                report_before=None, status="rejected")
        except RunConflict as exc:
            return Outcome("busy", error=str(exc))
        if not created:
            return self._existing(run)
        reasons = "; ".join(r.split(": ", 1)[-1] for r in decision.reasons) or "validation unavailable"
        shown = redact_text(text).strip() or "(empty message)"
        if len(shown) > guards.MAX_INPUT_CHARS:
            shown = shown[: guards.MAX_INPUT_CHARS] + " …[truncated]"
        self.store.add_message(conv.thread_id, run.run_id, "user", shown, kind="blocked",
                               guardrail=decision.summary(), trace_id=trace_id)
        reply = (f"Request not sent to the research agents: {reasons}. "
                 + ("Remove credentials from your message; never paste API keys here." if "credential" in reasons
                    else "Please rephrase your request."))
        msg = self.store.add_message(conv.thread_id, run.run_id, "assistant", reply, kind="blocked",
                                     guardrail=decision.summary(), trace_id=trace_id)
        with self.tracer.turn(trace_id=trace_id, name="research-turn", session_id=conv.thread_id, user_id=owner,
                              input=shown, metadata={"run_id": run.run_id, "outcome": "input_rejected"},
                              tags=["guardrail-rejected"]) as turn:
            turn.update(output=reply)
            self._score_guardrail(trace_id, "guardrail.input", decision, run.run_id)
        return Outcome("rejected", run.run_id, msg, error=reply)

    def _score_guardrail(self, trace_id: str, name: str, decision: guards.Decision, run_id: str) -> None:
        self.tracer.score(trace_id=trace_id, name=name, value=decision.action, data_type="CATEGORICAL",
                          comment="; ".join(decision.reasons)[:500] or None, score_id=f"{run_id}-{name}",
                          metadata=decision.summary())

    @contextmanager
    def _keepalive(self, run_id: str):
        """Heartbeat while a run executes, so only a dead process makes a run look stale."""
        stop = threading.Event()

        def beat():
            while not stop.wait(20):
                try:
                    self.store.heartbeat(run_id)
                except Exception:
                    pass

        threading.Thread(target=beat, name=f"heartbeat-{run_id[:8]}", daemon=True).start()
        try:
            yield
        finally:
            stop.set()

    def _execute(self, owner: str, run: Run, payload, on_event, *, input_decision=None,
                 input_text: str | None = None) -> Outcome:
        with self._keepalive(run.run_id):
            return self._execute_inner(owner, run, payload, on_event, input_decision, input_text)

    def _execute_inner(self, owner: str, run: Run, payload, on_event, input_decision, input_text) -> Outcome:
        settings, thread_id = run.settings, run.thread_id

        def emit(line: str) -> None:
            if on_event:
                try:
                    on_event(line)
                except Exception as exc:  # a display problem must not fail the research
                    log.warning("Progress display failed: %s", type(exc).__name__)
        run_log = list(run.log or [])
        agent = self._agent(settings)
        config_ = {"configurable": {"thread_id": thread_id, "app_run_id": run.run_id}, "recursion_limit": 250,
                   "callbacks": self.tracer.callbacks(), "run_name": "research-agent",
                   "metadata": {"app_run_id": run.run_id, "depth": settings.get("depth")}}
        segment = "research-turn" if input_text is not None else "research-turn (resumed)"
        attempts = run.attempts
        try:
            with self.tracer.turn(trace_id=run.trace_id, name=segment, session_id=thread_id, user_id=owner,
                                  input=input_text, tags=[f"depth:{settings.get('depth')}"],
                                  metadata={"run_id": run.run_id, "model": settings.get("model"),
                                            "hitl": settings.get("hitl")}) as turn:
                if input_decision is not None:
                    self._score_guardrail(run.trace_id, "guardrail.input", input_decision, run.run_id)
                while True:
                    if payload != "finalize":
                        last_beat = 0.0
                        for namespace, update in agent.stream(payload, config_, stream_mode="updates",
                                                              subgraphs=True):
                            for _node, node_update in update.items():
                                for line in progress_lines(namespace, node_update):
                                    run_log.append(line)
                                    emit(line)
                            if time.monotonic() - last_beat > 5:  # persist progress for page reloads
                                self.store.heartbeat(run.run_id, run_log)
                                last_beat = time.monotonic()
                    snapshot = agent.get_state(config_)
                    if snapshot.interrupts:
                        self.store.update_run(run.run_id, status="awaiting_review", log=run_log)
                        turn.event("awaiting_human_review",
                                   {"steps": len(snapshot.interrupts[0].value.get("action_requests", []))})
                        return Outcome("awaiting_review", run.run_id, pending=snapshot.interrupts[0].value)
                    if not turn_finished(snapshot):  # never finalize a partial run as an (empty) answer
                        raise RuntimeError("The agents stopped before finishing this step. "
                                           "Use Retry from last step to continue.")

                    kind, response = self._response(snapshot, run.report_before)
                    decision = guards.validate(kind, response)
                    turn.event("guardrail.output", decision.summary(),
                               level="DEFAULT" if decision.usable else "WARNING")
                    if decision.action == "retry" and attempts < self.max_output_retries:
                        attempts += 1
                        line = f"🛠️ Report failed validation ({'; '.join(decision.reasons)[:200]}); " \
                               f"asking the agents to repair it (attempt {attempts}/{self.max_output_retries})"
                        run_log.append(line)
                        emit(line)
                        prompt = REPAIR_PROMPT.format(problems="\n".join(f"- {r}" for r in decision.reasons),
                                                      path=REPORT_PATH)
                        self.store.update_run(run.run_id, attempts=attempts, log=run_log,
                                              settings={**settings, "prompt": prompt})
                        payload = {"messages": [{"role": "user", "content": prompt}]}
                        continue
                    if decision.action == "retry":  # repair budget used up: try the safe mechanical fix
                        decision = self._drop_dangling(kind, decision, run_log, emit)
                    return self._finish(owner, run, kind, decision, run_log, turn)
        except DatabaseError:
            raise
        except Exception as exc:
            error = redact_text(f"{type(exc).__name__}: {exc}")[:1000]
            log.warning("Run %s failed: %s", run.run_id, type(exc).__name__)
            try:
                self.store.update_run(run.run_id, status="failed", error=error, log=run_log)
            except DatabaseError:
                pass
            return Outcome("failed", run.run_id, error=error)
        except BaseException:  # page reload / rerun stops the script: leave the run resumable
            try:
                self.store.update_run(run.run_id, status="failed", log=run_log,
                                      error="Interrupted before finishing (page reloaded or app stopped).")
            except Exception:
                pass
            raise

    def _drop_dangling(self, kind: str, decision: guards.Decision, run_log: list, emit) -> guards.Decision:
        """Remove citation markers that point to no reference, if that is all that is wrong.

        Only markers are removed (never sources added), and the result must pass the full
        output guard again; otherwise the original flagged decision is kept.
        """
        fixed, removed = guards.drop_dangling_citations(decision.text)
        if not removed:
            return decision
        again = guards.validate(kind, fixed)
        if not again.usable:
            return decision
        note = f"Removed citation marker(s) {removed} that had no matching reference."
        run_log.append(f"🛠️ {note}")
        emit(f"🛠️ {note}")
        again.action = "repaired"
        again.reasons = [note] + again.reasons
        return again

    def _response(self, snapshot, report_before: str | None) -> tuple[str, str]:
        """A rewritten /final_report.md is a report; otherwise the orchestrator's reply is an answer."""
        report = file_text((snapshot.values.get("files") or {}).get(REPORT_PATH))
        if report and _digest(report) != report_before:
            return "report", tidy_markdown(report)
        messages = snapshot.values.get("messages") or []
        last_ai = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
        return "answer", tidy_markdown(msg_text(last_ai)) if last_ai and msg_text(last_ai).strip() else ""

    def _finish(self, owner: str, run: Run, kind: str, decision: guards.Decision, run_log: list, turn) -> Outcome:
        summary = decision.summary()
        notices = []
        if decision.action == "retry":  # repair budget used up: show it, clearly flagged
            summary["action"] = "flagged"
            notices.append("The report still fails these checks after automatic repair: "
                           + "; ".join(decision.reasons))
        if decision.action == "redact":
            notices.append("Sensitive data in the response was redacted.")
        if decision.action == "repaired":
            notices.append(decision.reasons[0])
        if decision.usable or decision.action == "retry":
            content, stored_kind = decision.text, kind
        else:
            content, stored_kind = ("The response was withheld because it failed output validation: "
                                    + ("; ".join(decision.reasons) or "validation unavailable") + "."), "blocked"
        msg = self.store.add_message(run.thread_id, run.run_id, "assistant", content, kind=stored_kind,
                                     guardrail=summary, trace_id=run.trace_id)
        self.tracer.score(trace_id=run.trace_id, name="guardrail.output", value=summary["action"],
                          data_type="CATEGORICAL", comment="; ".join(decision.reasons)[:500] or None,
                          score_id=f"{run.run_id}-guardrail.output", metadata=summary)
        if stored_kind != "blocked":
            self._evaluate(run, msg, kind, content)
        self.store.update_run(run.run_id, status="completed", log=run_log)
        turn.update(output=content[:20000], metadata={"kind": stored_kind, "guardrail": summary["action"]})
        return Outcome("completed", run.run_id, self.store.get_message(owner, msg["message_id"]), notices=notices)

    def _evaluate(self, run: Run, msg: dict, kind: str, content: str) -> None:
        """Deterministic quality scores for every shown response (free; no model calls)."""
        user = [m for m in self.store.messages(run.thread_id) if m["run_id"] == run.run_id and m["role"] == "user"]
        question = user[0]["content"] if user else ""
        e = evaluation.EvalInput(kind=kind, response=content, question=question,
                                 target=evaluation.target_from_report(content) if kind == "report" else None,
                                 depth=run.settings.get("depth"), evidence=self.store.evidence(thread_id=run.thread_id))
        scores = evaluation.evaluate(e)
        self.store.set_evaluation(msg["message_id"], {"version": evaluation.EVALUATOR_VERSION,
                                                      "scores": [s.as_dict() for s in scores]})
        for s in scores:
            if s.value is not None:
                self.tracer.score(trace_id=run.trace_id, name=f"eval.{s.name}", value=s.value, data_type="NUMERIC",
                                  comment=s.comment[:500], score_id=f"{run.run_id}-eval.{s.name}",
                                  metadata={"evaluator": s.source})

    # -- feedback ------------------------------------------------------------
    def feedback(self, owner: str, message_id: str, value: int | None) -> dict | None:
        """Thumbs up (1) / down (-1) / cleared (None) on an assistant message, mirrored to its trace."""
        msg = self.store.set_feedback(owner, message_id, value)
        if msg and msg.get("trace_id"):
            if value is None:
                self.tracer.delete_score(f"{message_id}-user_feedback")
                return msg
            self.tracer.score(trace_id=msg["trace_id"], name="user_feedback", value=1 if value > 0 else 0,
                              data_type="BOOLEAN", score_id=f"{message_id}-user_feedback",
                              comment="thumbs up" if value > 0 else "thumbs down")
        return msg
