"""Streamlit front end for the Deep Agents market-research pipeline.

Run with:  streamlit run app.py
"""

import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# Treat the .env placeholders ("your-...-here", "sk-...") as missing keys.
for _key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
    if os.getenv(_key, "").startswith("your-") or os.getenv(_key, "").endswith("..."):
        os.environ.pop(_key)

from agents import DEFAULT_MODEL, DEPTHS, REPORT_PATH, build_agent  # noqa: E402
from pdf_export import markdown_to_pdf  # noqa: E402

st.set_page_config(page_title="Competitor Research Agents", page_icon="🔎", layout="wide")

ss = st.session_state
ss.setdefault("phase", "idle")  # idle | review | done | error
ss.setdefault("log", [])
ss.setdefault("pending", None)
ss.setdefault("report", None)
ss.setdefault("pdf", None)
ss.setdefault("error", None)


# ---------------------------------------------------------------- helpers
def _file_text(fd) -> str | None:
    if fd is None:
        return None
    if isinstance(fd, dict):
        content = fd.get("content")
        return "\n".join(content) if isinstance(content, list) else content
    return str(fd)


def _msg_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content or ""


def tidy_markdown(md: str) -> str:
    """Normalise the agent's Markdown so it renders cleanly in Streamlit and the PDF."""
    md = md.strip()
    md = re.sub(r"^```(?:markdown|md)?\s*\n(.*)\n```$", r"\1", md, flags=re.S)  # unwrap fenced reply
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


def pdf_preview_images(pdf_bytes: bytes) -> list[bytes]:
    """Render PDF pages to PNG for an in-app preview (needs PyMuPDF; optional)."""
    try:
        import pymupdf
    except ImportError:
        return []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [page.get_pixmap(dpi=110).tobytes("png") for page in doc]


def _log_messages(namespace: tuple, update) -> None:
    """Turn streamed graph updates into human-readable progress lines."""
    if not isinstance(update, dict):
        return
    msgs = update.get("messages")
    if not isinstance(msgs, list):
        return
    who = "Orchestrator" if not namespace else "Subagent"
    for msg in msgs:
        for call in getattr(msg, "tool_calls", None) or []:
            args = call.get("args", {})
            name = call.get("name")
            if name == "task":
                ss.log.append(f"🧭 **{who}** → delegating to `{args.get('subagent_type')}`: "
                              f"{args.get('description', '')[:200]}")
            elif name == "internet_search":
                ss.log.append(f"🔎 {who} searching: *{args.get('query')}*")
            elif name in ("write_file", "edit_file"):
                ss.log.append(f"📝 {who} writing `{args.get('file_path')}`")
            elif name == "write_todos":
                ss.log.append(f"🗒️ {who} updated its plan")
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) == "task":
            ss.log.append(f"✅ Subagent finished: {_msg_text(msg)[:200]}")


def run_agent(payload) -> None:
    """Run (or resume) the agent until it finishes or pauses for human review."""
    config = {"configurable": {"thread_id": ss.thread_id}, "recursion_limit": 250}
    try:
        with st.status("Agents at work…", expanded=True) as status:
            for namespace, update in ss.agent.stream(
                payload, config, stream_mode="updates", subgraphs=True
            ):
                for _node, node_update in update.items():
                    before = len(ss.log)
                    _log_messages(namespace, node_update)
                    for line in ss.log[before:]:
                        st.markdown(line)
            status.update(label="Agents paused / finished", state="complete")

        snapshot = ss.agent.get_state(config)
        if snapshot.interrupts:
            ss.pending = snapshot.interrupts[0].value
            ss.phase = "review"
            return

        files = snapshot.values.get("files", {}) or {}
        report = _file_text(files.get(REPORT_PATH))
        if not report:  # fall back to the orchestrator's final answer
            report = _msg_text(snapshot.values["messages"][-1])
        ss.report = tidy_markdown(report)
        ss.pdf = None
        ss.pending = None
        ss.phase = "done"
    except Exception as exc:  # surface API/key errors in the UI
        ss.error = f"{type(exc).__name__}: {exc}"
        ss.phase = "error"


def start_run(target: str, depth: str, hitl: bool, model: str) -> None:
    ss.agent = build_agent(depth=depth, hitl=hitl, model=model)
    ss.thread_id = str(uuid.uuid4())
    ss.target = target
    ss.log, ss.report, ss.pdf, ss.pending, ss.error = [], None, None, None, None
    run_agent({"messages": [{"role": "user", "content": f"Research the competitors of: {target}"}]})


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OpenAI API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    tavily_key = st.text_input("Tavily API key", value=os.getenv("TAVILY_API_KEY", ""), type="password")
    model = st.text_input("Model", value=DEFAULT_MODEL, help="Any LangChain `provider:model` string.")
    depth = st.radio(
        "Research depth",
        list(DEPTHS),
        index=1,
        format_func=str.capitalize,
        help="Basic = fast & short · Standard = balanced · Advanced = deep search, SWOT, recommendations",
    )
    hitl = st.toggle("Human-in-the-loop approval", value=True,
                     help="Pause before every subagent step so you can approve, edit or reject it.")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

# ---------------------------------------------------------------- main
st.title("🔎 Competitor Market Research")
st.caption("LangChain Deep Agents · discovery → research → synthesis → quality control")

busy = ss.phase == "review"
with st.form("query"):
    target = st.text_input("Product or company name", placeholder="e.g. Notion, Zomato, Tesla Model 3",
                           disabled=busy)
    submitted = st.form_submit_button("🚀 Run research", type="primary", disabled=busy)

if submitted:
    if not target.strip():
        st.warning("Enter a product or company name.")
    elif not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")):
        st.warning("Add your OpenAI and Tavily API keys in the sidebar (or a .env file).")
    else:
        start_run(target.strip(), depth, hitl, model)
        st.rerun()

if ss.log:
    with st.expander(f"Agent activity ({len(ss.log)} events)", expanded=ss.phase != "done"):
        for line in ss.log:
            st.markdown(line)

# ---- human-in-the-loop review
if ss.phase == "review" and ss.pending:
    requests = ss.pending.get("action_requests", [])
    st.subheader("🙋 Approval needed")
    st.info(f"The orchestrator wants to run {len(requests)} subagent step(s). "
            "Review the task(s), edit them if needed, then approve or reject.")
    edited = []
    for i, req in enumerate(requests):
        args = req.get("args", {})
        st.markdown(f"**Step {i + 1}: `{args.get('subagent_type', req.get('name'))}`**")
        edited.append(st.text_area("Task description", value=args.get("description", ""),
                                   key=f"desc_{ss.thread_id}_{len(ss.log)}_{i}", height=120))
    feedback = st.text_input("Feedback if rejecting (tells the orchestrator what to change)",
                             key=f"fb_{ss.thread_id}_{len(ss.log)}")

    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", width="stretch"):
        decisions = []
        for req, desc in zip(requests, edited):
            if desc != req["args"].get("description"):
                decisions.append({"type": "edit",
                                  "edited_action": {"name": req["name"], "args": {**req["args"], "description": desc}}})
            else:
                decisions.append({"type": "approve"})
        run_agent(Command(resume={"decisions": decisions}))
        st.rerun()
    if col2.button("❌ Reject", width="stretch"):
        message = feedback or "The human reviewer rejected this step."
        run_agent(Command(resume={"decisions": [{"type": "reject", "message": message} for _ in requests]}))
        st.rerun()
    if col3.button("⏹️ Cancel run", width="stretch"):
        ss.phase, ss.pending = "idle", None
        st.rerun()

# ---- errors
if ss.phase == "error":
    st.error(ss.error)
    # The checkpointer keeps progress, so a transient failure (rate limit, timeout) can resume.
    if ss.get("agent") and st.button("🔁 Retry from last step"):
        ss.error = None
        run_agent(None)
        st.rerun()

# ---- final output: two formats — 1. Markdown, 2. PDF
if ss.phase == "done" and ss.report:
    st.success(f"Research on **{ss.target}** complete.")
    slug = re.sub(r"[^a-z0-9]+", "-", ss.target.lower()).strip("-") or "report"
    title = f"{ss.target} – Competitor Report"

    st.subheader("📦 Output")
    fmt = st.radio("Choose output format", ["1. Markdown", "2. PDF"], horizontal=True,
                   label_visibility="collapsed")

    if fmt == "1. Markdown":
        st.download_button("⬇️ Download Markdown (.md)", ss.report, file_name=f"{slug}_competitor_report.md",
                           mime="text/markdown", type="primary")
        view = st.segmented_control("View", ["Rendered", "Source"], default="Rendered",
                                    label_visibility="collapsed")
        with st.container(border=True):
            if view == "Source":
                st.code(ss.report, language="markdown")
            else:
                st.markdown(ss.report)
    else:
        if ss.get("pdf") is None:
            try:
                ss.pdf = markdown_to_pdf(ss.report, title=title)
            except Exception as exc:
                ss.pdf = b""
                st.error(f"PDF export failed: {exc}")
        if ss.pdf:
            col_dl, col_save = st.columns(2)
            col_dl.download_button("⬇️ Download PDF", ss.pdf, file_name=f"{slug}_competitor_report.pdf",
                                   mime="application/pdf", type="primary", width="stretch")
            # Some antivirus tools (e.g. McAfee "Phishing.JD") falsely flag browser-downloaded
            # PDFs that contain links; writing the file locally avoids the download scan.
            if col_save.button("💾 Save PDF to reports folder", width="stretch"):
                out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports",
                                   f"{slug}_competitor_report.pdf")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as fh:
                    fh.write(ss.pdf)
                st.success(f"Saved to `reports/{os.path.basename(out)}` in the app folder")
                if hasattr(os, "startfile"):
                    os.startfile(out)
            pages = pdf_preview_images(ss.pdf)
            if pages:
                st.caption(f"Preview · {len(pages)} page(s)")
                with st.container(border=True, height=900):
                    for png in pages:
                        st.image(png, width="stretch")
