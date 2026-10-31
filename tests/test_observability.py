"""Langfuse integration with the real SDK, exporting to memory instead of a Langfuse server."""

import uuid

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import config
from observability import Tracer, new_trace_id

SYNTHETIC_KEY = "sk-proj-SYNTHETICSYNTHETICSYNTHETIC0000"


@pytest.fixture
def traced(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")  # the SDK reads it too
    exporter = InMemorySpanExporter()
    # a unique synthetic public key gives a fresh SDK client; the base URL is unreachable on purpose
    settings = config.LangfuseSettings(True, "enabled", f"pk-lf-test-{uuid.uuid4().hex[:8]}", "sk-lf-test",
                                       "http://127.0.0.1:9")
    tracer = Tracer(settings, span_exporter=exporter)
    yield tracer, exporter
    tracer.shutdown()


def _spans(tracer, exporter):
    tracer.flush()
    return exporter.get_finished_spans()


def test_disabled_tracer_is_a_safe_no_op():
    tracer = Tracer(config.LangfuseSettings(False, "disabled (LANGFUSE_TRACING_ENABLED=false)"))
    assert tracer.status() == ("disabled", "disabled (LANGFUSE_TRACING_ENABLED=false)")
    assert tracer.callbacks() == []
    with tracer.turn(trace_id=new_trace_id(), name="t", session_id="s", user_id="u") as turn:
        turn.update(output="x")
        turn.event("e")
    assert tracer.score(trace_id="a" * 32, name="n", value=1, data_type="NUMERIC") is False


def test_turn_groups_langchain_spans_by_session_and_masks_secrets(traced):
    from langchain_core.runnables import RunnableLambda

    tracer, exporter = traced
    trace_id, thread_id = new_trace_id(), str(uuid.uuid4())
    chain = RunnableLambda(lambda x: f"echo {x}")
    with tracer.turn(trace_id=trace_id, name="research-turn", session_id=thread_id, user_id="local",
                     input=f"question with {SYNTHETIC_KEY}") as turn:
        chain.invoke("hello", config={"callbacks": tracer.callbacks()})
        turn.update(output="done")
    spans = _spans(tracer, exporter)
    assert len(spans) >= 2  # root observation + the LangChain run
    assert {format(s.context.trace_id, "032x") for s in spans} == {trace_id}
    assert all(s.attributes.get("session.id") == thread_id for s in spans)
    assert all(s.attributes.get("user.id") == "local" for s in spans)
    dumped = " ".join(str(dict(s.attributes)) for s in spans)
    assert SYNTHETIC_KEY not in dumped and "[REDACTED:openai_key]" in dumped


def test_resumed_segments_join_the_same_trace(traced):
    tracer, exporter = traced
    trace_id = new_trace_id()
    for name in ("research-turn", "research-turn (resumed)"):
        with tracer.turn(trace_id=trace_id, name=name, session_id="thread", user_id="u"):
            pass
    assert {format(s.context.trace_id, "032x") for s in _spans(tracer, exporter)} == {trace_id}


def test_errors_are_recorded_and_re_raised(traced):
    tracer, exporter = traced
    with pytest.raises(ValueError):
        with tracer.turn(trace_id=new_trace_id(), name="t", session_id="s", user_id="u"):
            raise ValueError("model failed")
    span = _spans(tracer, exporter)[-1]
    assert span.attributes.get("langfuse.observation.level") == "ERROR"


def test_scores_target_the_right_trace(traced, monkeypatch):
    tracer, _ = traced
    sent = []
    monkeypatch.setattr(tracer.client, "create_score", lambda **kw: sent.append(kw))
    assert tracer.score(trace_id="b" * 32, name="eval.relevance", value=0.5, data_type="NUMERIC",
                        score_id="run-eval.relevance")
    assert sent[0]["trace_id"] == "b" * 32 and sent[0]["score_id"] == "run-eval.relevance"


def test_langfuse_outage_never_breaks_the_app(traced, monkeypatch):
    tracer, _ = traced

    def down(**kw):
        raise ConnectionError("langfuse down")

    monkeypatch.setattr(tracer.client, "create_score", down)
    assert tracer.score(trace_id="c" * 32, name="x", value=1, data_type="NUMERIC") is False
    monkeypatch.setattr(tracer.client, "auth_check", lambda: (_ for _ in ()).throw(ConnectionError("down")))
    state, detail = tracer.status(check=True)
    assert state == "unavailable" and "research continues" in detail
    tracer.flush()  # must not raise even though the endpoint is unreachable


def test_agent_run_traces_subagents_tools_and_scores(traced, store, fake_model, search_client, monkeypatch):
    from agents import build_agent
    from runner import Runner
    from tests.fakes import SETTINGS

    tracer, exporter = traced
    scores = []
    monkeypatch.setattr(tracer.client, "create_score", lambda **kw: scores.append(kw))
    runner = Runner(store, tracer, search_client=search_client,
                    agent_factory=lambda **kw: build_agent(**{**kw, "model": fake_model}))
    conv = store.create("trace-owner")
    out = runner.submit("trace-owner", conv.thread_id, "Notion", str(uuid.uuid4()), SETTINGS)
    assert out.status == "completed"
    spans = _spans(tracer, exporter)
    names = {s.name for s in spans}
    assert {"research-turn", "internet_search", "task", "FakeResearchModel"} <= names, names
    assert {format(s.context.trace_id, "032x") for s in spans} == {out.message["trace_id"]}
    assert {s.attributes.get("session.id") for s in spans} == {conv.thread_id}
    assert {kw["trace_id"] for kw in scores} == {out.message["trace_id"]}
    assert {"guardrail.input", "guardrail.output", "eval.citation_support"} <= {kw["name"] for kw in scores}
    # retrieved page text is masked/sanitised before export: the injection phrase never leaves
    assert "Ignore all previous instructions" not in " ".join(str(dict(s.attributes)) for s in spans
                                                              if s.name == "internet_search")
