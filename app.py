"""Streamlit front end for the Deep Agents market-research pipeline.

Run with:  streamlit run app.py   (after `python manage.py init-db`)

Conversations, transcripts and agent memory live in PostgreSQL (see memory.py); this
file only renders them. The open conversation is kept in the URL (?thread=<id>), so a
browser refresh or an app/Docker restart reopens it.
"""

import os
import re
import uuid
from datetime import timezone

import streamlit as st

import config  # loads .env
import guards
from agents import DEPTHS
from config import DEFAULT_MODEL, OWNER_ID
from memory import ConversationStore, DatabaseError, RunConflict, checkpointer, open_pool, schema_problems
from observability import Tracer
from pdf_export import markdown_to_pdf
from runner import DEFAULT_TITLE, Runner

# Treat the .env placeholders ("YOUR_..._HERE", "sk-...") as missing keys.
for _key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
    if config.is_placeholder(os.getenv(_key)):
        os.environ.pop(_key, None)

st.set_page_config(page_title="Competitor Research Agents", page_icon="🔎", layout="wide")

ss = st.session_state
ss.setdefault("submission_id", str(uuid.uuid4()))  # idempotency key for the next message
ss.setdefault("notices", [])
ss.setdefault("pdf", {})  # message_id -> PDF bytes
ss.setdefault("output_message", None)  # assistant message shown in the output panel


# ---------------------------------------------------------------- services
@st.cache_resource(show_spinner="Connecting to the database…")
def services() -> tuple[ConversationStore, Runner, Tracer]:
    """One pool/checkpointer/tracer per app process, shared by all sessions and reruns."""
    pool = open_pool()
    if problems := schema_problems(pool):
        pool.close()
        raise DatabaseError("The database is not initialised (" + "; ".join(problems) + "). "
                            "Run `python manage.py init-db` (Docker: `docker compose run --rm init-db`), "
                            "then reload this page.")
    store = ConversationStore(pool, checkpointer(pool))
    tracer = Tracer()
    return store, Runner(store, tracer), tracer


try:
    store, runner, tracer = services()
except (config.ConfigError, DatabaseError) as exc:
    st.title("🔎 Competitor Market Research")
    st.error(f"**Persistent conversation memory is not available.**\n\n{exc}")
    st.info("The app does not fall back to temporary memory, so conversations are never silently lost. "
            "See README → Configuration and Database setup.")
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def tracing_status() -> tuple[str, str]:
    return tracer.status(check=True)


def open_thread(thread_id: str | None) -> None:
    ss.thread_id = thread_id
    ss.output_message = None
    if thread_id:
        st.query_params["thread"] = thread_id
    else:
        st.query_params.pop("thread", None)


def main() -> None:
    # Restore the open conversation from the URL after a refresh/restart.
    if "thread_id" not in ss:
        wanted = st.query_params.get("thread")
        ss.thread_id = wanted if wanted and store.get(OWNER_ID, wanted) else None


    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("💬 Conversations")
        if st.button("➕ New conversation", width="stretch", type="primary" if ss.thread_id else "secondary"):
            open_thread(None)
            st.rerun()
        try:
            conversations = store.list(OWNER_ID)
        except DatabaseError as exc:
            st.error(str(exc))
            conversations = []
        with st.container(height=260 if len(conversations) > 6 else "content", border=False):
            for conv in conversations:
                current = conv.thread_id == ss.thread_id
                label = ("▶ " if current else "") + conv.title
                if st.button(label, key=f"open_{conv.thread_id}", width="stretch",
                             help=f"Updated {conv.updated_at.astimezone(timezone.utc):%d %b %Y %H:%M} UTC",
                             disabled=current):
                    open_thread(conv.thread_id)
                    st.rerun()
            if not conversations:
                st.caption("No saved conversations yet.")

        current_conv = store.get(OWNER_ID, ss.thread_id) if ss.thread_id else None
        if ss.thread_id and current_conv is None:  # deleted in another tab, or not ours
            open_thread(None)
            ss.notices.append("That conversation no longer exists; starting a new one.")
        if current_conv:
            with st.expander("✏️ Rename / delete this conversation"):
                new_title = st.text_input("Title", value=current_conv.title, key=f"title_{current_conv.thread_id}")
                col1, col2 = st.columns(2)
                if col1.button("Save title", width="stretch") and new_title.strip() != current_conv.title:
                    store.rename(OWNER_ID, current_conv.thread_id, new_title)
                    st.rerun()
                if col2.button("🗑️ Delete", width="stretch"):
                    ss.confirm_delete = current_conv.thread_id
                    st.rerun()
            if ss.get("confirm_delete") == current_conv.thread_id:  # explicit second step before deleting
                with st.container(border=True):
                    st.warning(f"Delete **{current_conv.title}**? Its full transcript, research notes, agent memory "
                               "(checkpoints) and feedback are permanently deleted. This cannot be undone.")
                    st.caption("Traces already sent to Langfuse are not deleted; remove them in Langfuse if needed.")
                    col1, col2 = st.columns(2)
                    if col1.button("🗑️ Delete permanently", type="primary", width="stretch"):
                        ss.confirm_delete = None
                        try:
                            store.delete(OWNER_ID, current_conv.thread_id)
                            open_thread(None)
                            st.rerun()
                        except (RunConflict, DatabaseError) as exc:
                            st.error(str(exc))
                    if col2.button("Keep it", width="stretch"):
                        ss.confirm_delete = None
                        st.rerun()

        st.header("⚙️ Settings")
        # Keys from .env are never sent to the browser; a value typed here overrides them.
        openai_key = st.text_input("OpenAI API key", type="password",
                                   placeholder="configured in .env" if os.getenv("OPENAI_API_KEY") else "sk-…")
        tavily_key = st.text_input("Tavily API key", type="password",
                                   placeholder="configured in .env" if os.getenv("TAVILY_API_KEY") else "tvly-…")
        saved = current_conv.settings if current_conv else {}
        # Keyed per conversation: each conversation opens with its own saved settings, and changing
        # defaults never silently resets a widget the user already set.
        scope = ss.thread_id or "new"
        model = st.text_input("Model", value=saved.get("model", DEFAULT_MODEL), key=f"model_{scope}",
                              help="Any LangChain `provider:model` string.")
        depth_options = list(DEPTHS)
        depth = st.radio(
            "Research depth",
            depth_options,
            index=depth_options.index(saved.get("depth", "standard")),
            format_func=str.capitalize,
            key=f"depth_{scope}",
            help="Basic = fast & short · Standard = balanced · Advanced = deep search, SWOT, recommendations",
        )
        hitl = st.toggle("Human-in-the-loop approval", value=saved.get("hitl", True), key=f"hitl_{scope}",
                         help="Pause before every subagent step so you can approve, edit or reject it.")
        if openai_key and not config.is_placeholder(openai_key):
            os.environ["OPENAI_API_KEY"] = openai_key
        if tavily_key and not config.is_placeholder(tavily_key):
            os.environ["TAVILY_API_KEY"] = tavily_key

        st.header("🩺 Status")
        st.caption("🗄️ Memory: PostgreSQL (persistent)")
        state, detail = tracing_status()
        icon = {"enabled": "🟢", "unavailable": "🟠", "disabled": "⚪"}[state]
        st.caption(f"{icon} Langfuse tracing: {state} — {detail}")
        st.caption("🛡️ Guardrails AI: input, web-content and output validation active (local validators)")
        st.caption(f"👤 Single-user mode: conversations belong to `{OWNER_ID}` (no login)")

    settings = {"depth": depth, "hitl": hitl, "model": model}


    # ---------------------------------------------------------------- run helpers
    def missing_keys() -> list[str]:
        needed = ["TAVILY_API_KEY"] + (["OPENAI_API_KEY"] if model.startswith("openai:") else [])
        return [k for k in needed if config.is_placeholder(os.getenv(k))]


    def handle(outcome) -> None:
        ss.notices = list(outcome.notices)
        if outcome.status in ("busy",) and outcome.error:
            ss.notices.append(f"⚠️ {outcome.error}")
        if outcome.message and outcome.message.get("kind") in ("report", "answer"):
            ss.output_message = outcome.message["message_id"]


    def run_with_progress(fn, *args) -> None:
        with st.status("Agents at work…", expanded=True) as status:
            outcome = fn(*args, on_event=st.markdown)
            status.update(label="Agents paused / finished", state="complete")
        handle(outcome)
        st.rerun()


    def export_check(message: dict) -> guards.Decision:
        """Re-validate stored content right before export; exports never bypass the output guard."""
        decision = guards.validate(message["kind"], message["content"])
        if decision.action == "retry":  # structurally flagged but safe: exportable with the warning shown
            decision.action = "flagged"
        return decision


    # ---------------------------------------------------------------- main
    st.title("🔎 Competitor Market Research")
    st.caption("LangChain Deep Agents · discovery → research → synthesis → quality control")

    view = runner.view(OWNER_ID, ss.thread_id) if ss.thread_id else None
    messages = view.messages if view else []
    active = view.active_run if view else None
    latest = view.latest_run if view else None
    reviewing = bool(view and view.pending)
    busy_elsewhere = bool(active and active.status == "running" and not active.stale)

    for note in ss.notices:  # shown once, then cleared
        st.info(note)
    ss.notices = []

    # ---- transcript (from the database)
    for msg in messages:
        with st.chat_message(msg["role"]):
            guard = msg.get("guardrail") or {}
            if msg["kind"] == "blocked":
                (st.warning if msg["role"] == "assistant" else st.markdown)(msg["content"])
                continue
            if msg["role"] == "user":
                st.markdown(msg["content"])
                if guard.get("action") == "redact":
                    st.caption("🛡️ Personal data was redacted before sending.")
                continue
            if msg["kind"] == "report":
                created = msg["created_at"].astimezone(timezone.utc)
                with st.expander(f"📄 Report · {created:%d %b %Y %H:%M} UTC", expanded=msg is messages[-1]):
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
            if guard.get("action") == "redact":
                st.caption("🛡️ Sensitive data in this response was redacted.")
            elif guard.get("action") == "repaired":
                st.caption("🛠️ " + (guard.get("reasons") or ["Broken citation markers were removed."])[0])
            elif guard.get("action") == "flagged":
                st.caption("⚠️ Failed structural checks after automatic repair: " + "; ".join(guard.get("reasons", [])))
            for adv in guard.get("advisories", []):
                st.caption(f"ℹ️ {adv.split(': ', 1)[-1]}")
            cols = st.columns([1, 1, 3, 6])
            fb = msg.get("feedback")
            if cols[0].button("👍" if fb != 1 else "👍 ✓", key=f"up_{msg['message_id']}", help="Helpful"):
                runner.feedback(OWNER_ID, msg["message_id"], None if fb == 1 else 1)
                st.rerun()
            if cols[1].button("👎" if fb != -1 else "👎 ✓", key=f"down_{msg['message_id']}", help="Not helpful"):
                runner.feedback(OWNER_ID, msg["message_id"], None if fb == -1 else -1)
                st.rerun()
            if cols[2].button("📦 Export", key=f"out_{msg['message_id']}", help="Show in the output panel below"):
                ss.output_message = msg["message_id"]
                st.rerun()
            scores = (msg.get("evaluation") or {}).get("scores") or []
            if scores:
                with cols[3].popover("📊 Quality checks"):
                    st.caption("Automatic, deterministic checks — not a guarantee of accuracy.")
                    for s in scores:
                        value = "unavailable" if s["value"] is None else f"{s['value']:.2f}"
                        st.markdown(f"**{s['name']}**: {value} — {s['comment']}")

    # ---- activity of the latest run
    if latest and latest.log:
        with st.expander(f"Agent activity ({len(latest.log)} events)", expanded=latest.status != "completed"):
            for line in latest.log:
                st.markdown(line)

    # ---- human-in-the-loop review (restored from the checkpoint after refresh/restart)
    if reviewing:
        requests = view.pending.get("action_requests", [])
        st.subheader("🙋 Approval needed")
        st.info(f"The orchestrator wants to run {len(requests)} subagent step(s). "
                "Review the task(s), edit them if needed, then approve or reject.")
        edited = []
        for i, req in enumerate(requests):
            args = req.get("args", {})
            st.markdown(f"**Step {i + 1}: `{args.get('subagent_type', req.get('name'))}`**")
            edited.append(st.text_area("Task description", value=args.get("description", ""),
                                       key=f"desc_{active.run_id}_{len(active.log)}_{i}", height=120))
        feedback = st.text_input("Feedback if rejecting (tells the orchestrator what to change)",
                                 key=f"fb_{active.run_id}_{len(active.log)}")

        col1, col2, col3 = st.columns(3)
        if col1.button("✅ Approve", type="primary", width="stretch"):
            decisions = []
            for req, desc in zip(requests, edited):
                if desc != req["args"].get("description"):
                    decisions.append({"type": "edit",
                                      "edited_action": {"name": req["name"], "args": {**req["args"], "description": desc}}})
                else:
                    decisions.append({"type": "approve"})
            run_with_progress(runner.review, OWNER_ID, ss.thread_id, decisions)
        if col2.button("❌ Reject", width="stretch"):
            message = feedback or "The human reviewer rejected this step."
            run_with_progress(runner.review, OWNER_ID, ss.thread_id,
                              [{"type": "reject", "message": message} for _ in requests])
        if col3.button("⏹️ Cancel run", width="stretch"):
            handle(runner.cancel(OWNER_ID, ss.thread_id))
            st.rerun()

    # ---- errors / interrupted runs
    if latest and (latest.status == "failed" or latest.stale) and not reviewing:
        st.error(latest.error or "The last run stopped before finishing (the app was restarted or the page reloaded).")
        # The checkpointer keeps progress, so a transient failure (rate limit, timeout) can resume.
        col1, col2 = st.columns(2)
        if col1.button("🔁 Retry from last step", width="stretch"):
            run_with_progress(runner.retry, OWNER_ID, ss.thread_id)
        if col2.button("Discard this run", width="stretch"):
            handle(runner.cancel(OWNER_ID, ss.thread_id))
            st.rerun()
    if busy_elsewhere:
        st.warning("A run is in progress for this conversation in another tab or session. "
                   "Reload this page in a moment to see the result.")

    # ---- input
    # the first accepted message names the target (a refused one never reached the agents)
    first = not any(m["role"] == "user" and m["kind"] != "blocked" for m in messages)
    with st.form("query", clear_on_submit=True):
        label = "Product or company name" if first else "Follow-up question (or “Research the competitors of: …”)"
        placeholder = "e.g. Notion, Zomato, Tesla Model 3" if first else "e.g. Which competitor is cheapest for a 10-person team?"
        text = st.text_input(label, placeholder=placeholder, disabled=reviewing or busy_elsewhere)
        submitted = st.form_submit_button("🚀 Run research" if first else "💬 Send", type="primary",
                                          disabled=reviewing or busy_elsewhere)

    if submitted:
        if not text.strip():
            st.warning("Enter a product or company name." if first else "Enter a question.")
        elif absent := missing_keys():
            st.warning("Add your " + " and ".join(absent) + " in the sidebar (or the .env file).")
        else:
            if not ss.thread_id:
                conv = store.create(OWNER_ID, DEFAULT_TITLE, settings)
                open_thread(conv.thread_id)
            submission_id = ss.submission_id
            ss.submission_id = str(uuid.uuid4())  # the next message gets a new key; reruns reuse this one
            run_with_progress(runner.submit, OWNER_ID, ss.thread_id, text, submission_id, settings)

    # ---- final output: two formats — 1. Markdown, 2. PDF
    exportable = [m for m in messages if m["role"] == "assistant" and m["kind"] in ("report", "answer")]
    reports = [m for m in exportable if m["kind"] == "report"]
    selected = next((m for m in exportable if m["message_id"] == ss.output_message), None) \
        or (reports[-1] if reports else None)
    if selected and not reviewing:
        decision = export_check(selected)
        target = (current_conv.title if current_conv else "report")
        m = re.search(r"(?m)^#\s+(.+?)(?::\s*competitor analysis)?\s*$", selected["content"], re.I)
        target = m.group(1).strip() if m else target
        slug = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-") or "report"
        title = f"{target} – Competitor Report"

        st.divider()
        st.subheader("📦 Output")
        if not decision.usable and decision.action != "flagged":
            st.error("Export blocked: the content failed output validation ("
                     + ("; ".join(decision.reasons) or decision.action) + ").")
        else:
            if decision.action == "flagged":
                st.warning("This report failed structural checks: " + "; ".join(decision.reasons))
            content = decision.text
            fmt = st.radio("Choose output format", ["1. Markdown", "2. PDF"], horizontal=True,
                           label_visibility="collapsed")

            if fmt == "1. Markdown":
                st.download_button("⬇️ Download Markdown (.md)", content, file_name=f"{slug}_competitor_report.md",
                                   mime="text/markdown", type="primary")
                view_mode = st.segmented_control("View", ["Rendered", "Source"], default="Rendered",
                                                 label_visibility="collapsed")
                with st.container(border=True):
                    if view_mode == "Source":
                        st.code(content, language="markdown")
                    else:
                        st.markdown(content)
            else:
                key = selected["message_id"]
                if key not in ss.pdf:
                    try:
                        ss.pdf[key] = markdown_to_pdf(content, title=title)
                    except Exception as exc:
                        ss.pdf[key] = b""
                        st.error(f"PDF export failed: {exc}")
                pdf = ss.pdf[key]
                if pdf:
                    col_dl, col_save = st.columns(2)
                    col_dl.download_button("⬇️ Download PDF", pdf, file_name=f"{slug}_competitor_report.pdf",
                                           mime="application/pdf", type="primary", width="stretch")
                    # Some antivirus tools (e.g. McAfee "Phishing.JD") falsely flag browser-downloaded
                    # PDFs that contain links; writing the file locally avoids the download scan.
                    if col_save.button("💾 Save PDF to reports folder", width="stretch"):
                        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports",
                                           f"{slug}_competitor_report.pdf")
                        try:
                            os.makedirs(os.path.dirname(out), exist_ok=True)
                            with open(out, "wb") as fh:
                                fh.write(pdf)
                        except OSError as exc:  # e.g. a reports/ bind mount created by root on Linux
                            st.error(f"Could not save the PDF ({type(exc).__name__}). Make sure the reports folder "
                                     "is writable (Linux Docker hosts: `sudo chown 1000:1000 reports`), or use "
                                     "Download PDF.")
                        else:
                            st.success(f"Saved to `reports/{os.path.basename(out)}` in the app folder")
                            if hasattr(os, "startfile"):
                                os.startfile(out)
                    pages = []
                    try:
                        import pymupdf  # optional: in-app PDF page preview

                        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
                            pages = [page.get_pixmap(dpi=110).tobytes("png") for page in doc]
                    except ImportError:
                        pass
                    if pages:
                        st.caption(f"Preview · {len(pages)} page(s)")
                        with st.container(border=True, height=900):
                            for png in pages:
                                st.image(png, width="stretch")


try:
    main()
except DatabaseError as exc:  # e.g. PostgreSQL restarted or unreachable mid-session
    st.error(f"**Database problem:** {exc}")
    st.info("Saved conversations are safe in PostgreSQL. Reload the page once the database is back; "
            "an interrupted run can then be continued with Retry from last step.")
except PermissionError:  # the conversation was deleted in another tab while this page was open
    open_thread(None)
    st.warning("That conversation no longer exists. Reload the page to start a new one.")
