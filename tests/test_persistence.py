"""Persistent memory end-to-end: real deep-agent graph + PostgresSaver + application tables.

The model and web search are fakes (tests/fakes.py); everything else is the real code path.
"""

import uuid

import pytest
from langchain_core.messages import HumanMessage

import memory
from runner import Runner
from tests.fakes import SETTINGS

OWNER = "test-owner"


def submit(runner, thread_id, text, submission_id=None, settings=SETTINGS, **kw):
    lines = []
    out = runner.submit(OWNER, thread_id, text, submission_id or str(uuid.uuid4()), settings,
                        on_event=lines.append, **kw)
    return out, lines


def orchestrator_inputs(fake_model):
    return [c["messages"] for c in fake_model.calls if c["role"] == "orchestrator"]


# ---------------------------------------------------------------- schema
def test_init_db_is_repeatable_and_complete(pool):
    assert memory.schema_problems(pool) == []
    assert memory.init_db(pool)[-1] == "application tables up to date"
    assert memory.schema_problems(pool) == []


# ---------------------------------------------------------------- conversations
def test_conversation_crud_is_owner_scoped(store):
    conv = store.create(OWNER, "First")
    assert store.get(OWNER, conv.thread_id).title == "First"
    assert store.get("someone-else", conv.thread_id) is None
    assert store.get(OWNER, "not-a-uuid") is None
    assert conv.thread_id in [c.thread_id for c in store.list(OWNER)]
    assert conv.thread_id not in [c.thread_id for c in store.list("someone-else")]
    assert not store.rename("someone-else", conv.thread_id, "Hijacked")
    assert store.rename(OWNER, conv.thread_id, "  Renamed   title ")
    assert store.get(OWNER, conv.thread_id).title == "Renamed title"
    assert not store.delete("someone-else", conv.thread_id)
    assert store.get(OWNER, conv.thread_id) is not None


# ---------------------------------------------------------------- research + memory
def test_research_turn_produces_validated_report_with_evidence(runner, store, tracer):
    conv = store.create(OWNER)
    out, lines = submit(runner, conv.thread_id, "Notion")
    assert out.status == "completed", out.error
    msg = out.message
    assert msg["kind"] == "report" and msg["content"].startswith("# Notion: Competitor Analysis")
    assert msg["guardrail"]["action"] == "pass"
    assert any("delegating to `researcher_agent`" in line for line in lines)
    assert store.get(OWNER, conv.thread_id).title == "Notion"  # titled from the first message

    evidence = store.evidence(thread_id=conv.thread_id)
    assert {e["url"] for e in evidence} == {"https://acme.example/pricing", "https://www.globex.example/news/share/"}
    assert all("Ignore all previous instructions" not in e["content"] for e in evidence)  # sanitised

    scores = {s["name"]: s for s in msg["evaluation"]["scores"]}
    assert scores["citation_support"]["value"] == 1.0  # both cited URLs were retrieved
    assert scores["completeness"]["value"] == 1.0
    sent = {s["name"]: s for s in tracer.scores}
    assert sent["guardrail.input"]["value"] == "pass" and sent["guardrail.output"]["value"] == "pass"
    assert {s["trace_id"] for s in tracer.scores} == {msg["trace_id"]}
    assert {t["session_id"] for t in tracer.turns} == {conv.thread_id}


def test_follow_up_uses_earlier_turns_from_the_checkpoint(runner, store, fake_model):
    conv = store.create(OWNER)
    submit(runner, conv.thread_id, "Notion")
    out, _ = submit(runner, conv.thread_id, "What did I ask you to research?")
    assert out.status == "completed"
    assert out.message["kind"] == "answer"
    assert "Notion" in out.message["content"]
    # the model really received the earlier turn, restored from PostgreSQL
    last = orchestrator_inputs(fake_model)[-1]
    humans = [m.content for m in last if isinstance(m, HumanMessage)]
    assert humans[0] == "Research the competitors of: Notion" and humans[-1] == "What did I ask you to research?"


def test_threads_are_isolated(runner, store):
    a, b = store.create(OWNER), store.create(OWNER)
    submit(runner, a.thread_id, "Notion. The internal project codename is BLUE-HERON-42.")
    submit(runner, b.thread_id, "Figma")
    in_a, _ = submit(runner, a.thread_id, "What codename did I mention?")
    in_b, _ = submit(runner, b.thread_id, "What codename did I mention?")
    assert "BLUE-HERON-42" in in_a.message["content"]
    assert "BLUE-HERON-42" not in in_b.message["content"]
    assert all("BLUE-HERON-42" not in m["content"] for m in store.messages(b.thread_id))


def test_history_survives_application_restart(database_url, runner, store, tracer, fake_model, search_client):
    from agents import build_agent

    conv = store.create(OWNER)
    submit(runner, conv.thread_id, "Notion")
    before = store.messages(conv.thread_id)

    # a brand-new pool, checkpointer, store and runner = the app after a restart
    pool2 = memory.open_pool(database_url, wait_seconds=10)
    try:
        store2 = memory.ConversationStore(pool2, memory.checkpointer(pool2))
        runner2 = Runner(store2, tracer, search_client=search_client,
                         agent_factory=lambda **kw: build_agent(**{**kw, "model": fake_model}))
        view = runner2.view(OWNER, conv.thread_id)
        assert [m["message_id"] for m in view.messages] == [m["message_id"] for m in before]
        out = runner2.submit(OWNER, conv.thread_id, "Remind me what I asked about", str(uuid.uuid4()), SETTINGS)
        assert "Notion" in out.message["content"]
    finally:
        pool2.close()


def test_resubmission_and_reruns_never_duplicate(runner, store, fake_model):
    conv = store.create(OWNER)
    first, _ = submit(runner, conv.thread_id, "Notion", submission_id="same-submission")
    calls = len(fake_model.calls)
    again, _ = submit(runner, conv.thread_id, "Notion", submission_id="same-submission")
    assert again.message["message_id"] == first.message["message_id"]
    assert len(fake_model.calls) == calls  # the agent did not run again
    msgs = store.messages(conv.thread_id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    # the checkpoint holds the question exactly once
    state = runner._state(runner._agent(SETTINGS), conv.thread_id)
    humans = [m for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert len(humans) == 1


def test_conflicting_run_in_same_thread_is_blocked(runner, store):
    conv = store.create(OWNER)
    run, _ = store.start_run(conv.thread_id, "other-tab", settings=SETTINGS, trace_id="0" * 32, report_before=None)
    out, _ = submit(runner, conv.thread_id, "Notion")
    assert out.status == "busy" and "already in progress" in out.error
    assert store.messages(conv.thread_id) == []
    # a crashed run (stale heartbeat) no longer blocks the conversation
    with store.pool.connection() as conn:
        conn.execute("UPDATE conversation_runs SET updated_at = now() - interval '1 hour' WHERE run_id = %s",
                     (run.run_id,))
    out, _ = submit(runner, conv.thread_id, "Notion")
    assert out.status == "completed"
    assert store.get_run(run.run_id).status == "failed"


def test_secret_in_input_is_rejected_and_never_stored(runner, store, fake_model, tracer):
    conv = store.create(OWNER)
    key = "sk-proj-SYNTHETICSYNTHETICSYNTHETIC0000"
    out, _ = submit(runner, conv.thread_id, f"Research Notion with key {key}")
    assert out.status == "rejected" and "credentials" in out.error
    assert fake_model.calls == []
    with store.pool.connection() as conn:
        rows = conn.execute("SELECT content, guardrail::text AS g FROM conversation_messages "
                            "WHERE thread_id = %s", (conv.thread_id,)).fetchall()
    assert rows and all(key not in r["content"] and key not in r["g"] for r in rows)
    assert any(s["name"] == "guardrail.input" and s["value"] == "reject" for s in tracer.scores)


def test_secret_in_output_is_redacted_before_storage(runner, store, fake_model):
    conv = store.create(OWNER)
    submit(runner, conv.thread_id, "Notion")
    fake_model.leak_secret = True
    out, _ = submit(runner, conv.thread_id, "Summarise the pricing")
    assert out.message["guardrail"]["action"] == "redact"
    assert "sk-proj-FAKE" not in out.message["content"] and "[REDACTED:" in out.message["content"]


def test_invalid_report_gets_one_bounded_repair(runner, store, fake_model):
    fake_model.bad_first_report = True
    conv = store.create(OWNER)
    out, lines = submit(runner, conv.thread_id, "Notion")
    assert out.status == "completed"
    assert out.message["guardrail"]["action"] == "pass"
    assert any("asking the agents to repair it (attempt 1/1)" in line for line in lines)
    assert store.latest_run(conv.thread_id).attempts == 1


def test_repair_budget_exhausted_is_flagged(store, tracer, fake_model, search_client):
    from agents import build_agent

    fake_model.bad_first_report = True
    strict = Runner(store, tracer, search_client=search_client, max_output_retries=0,
                    agent_factory=lambda **kw: build_agent(**{**kw, "model": fake_model}))
    conv = store.create(OWNER)
    out, _ = submit(strict, conv.thread_id, "Notion")
    assert out.message["guardrail"]["action"] == "flagged" and out.notices


def test_failure_then_retry_resumes_without_duplicates(runner, store, fake_model):
    conv = store.create(OWNER)
    fake_model.fail_next = 1
    out, _ = submit(runner, conv.thread_id, "Notion")
    assert out.status == "failed" and "simulated provider outage" in out.error
    out = runner.retry(OWNER, conv.thread_id)
    assert out.status == "completed" and out.message["kind"] == "report"
    assert [m["role"] for m in store.messages(conv.thread_id)] == ["user", "assistant"]


# ---------------------------------------------------------------- human in the loop
def test_hitl_pause_survives_restart_and_resumes(runner, store, tracer):
    hitl = {**SETTINGS, "hitl": True}
    conv = store.create(OWNER)
    out, _ = submit(runner, conv.thread_id, "Notion", settings=hitl)
    assert out.status == "awaiting_review"
    assert out.pending["action_requests"][0]["args"]["subagent_type"] == "researcher_agent"

    view = runner.view(OWNER, conv.thread_id)  # e.g. after a browser refresh
    assert view.pending and view.active_run.status == "awaiting_review"
    blocked, _ = submit(runner, conv.thread_id, "another question")
    assert blocked.status == "busy"

    out = runner.review(OWNER, conv.thread_id, [{"type": "approve"}])
    assert out.status == "awaiting_review"  # next step: synthesizer
    assert runner.review(OWNER, conv.thread_id, [{"type": "approve"}]).status == "completed"
    trace_ids = {t["trace_id"] for t in tracer.turns}
    assert len(trace_ids) == 1  # all three segments belong to one request trace


def test_cancelled_review_does_not_block_the_next_message(runner, store):
    conv = store.create(OWNER)
    out, _ = submit(runner, conv.thread_id, "Notion", settings={**SETTINGS, "hitl": True})
    assert out.status == "awaiting_review"
    assert runner.cancel(OWNER, conv.thread_id).status == "cancelled"
    assert runner.view(OWNER, conv.thread_id).pending is None
    # HITL toggled off for the next turn on the same persisted thread
    out, _ = submit(runner, conv.thread_id, "What did I ask about?")
    assert out.status == "completed" and "Notion" in out.message["content"]


# ---------------------------------------------------------------- feedback & deletion
def test_feedback_is_stored_and_linked_to_the_response_trace(runner, store, tracer):
    conv = store.create(OWNER)
    out, _ = submit(runner, conv.thread_id, "Notion")
    message_id = out.message["message_id"]
    assert runner.feedback("someone-else", message_id, 1) is None
    assert runner.feedback(OWNER, message_id, -1)["feedback"] == -1
    fb = [s for s in tracer.scores if s["name"] == "user_feedback"]
    assert fb[-1]["trace_id"] == out.message["trace_id"] and fb[-1]["value"] == 0
    assert fb[-1]["score_id"] == f"{message_id}-user_feedback"  # re-voting updates, never duplicates


def test_delete_removes_transcript_and_checkpoints(runner, store, pool):
    conv = store.create(OWNER)
    keep = store.create(OWNER)
    submit(runner, conv.thread_id, "Notion")
    submit(runner, keep.thread_id, "Figma")

    def counts(thread_id):
        with pool.connection() as conn:
            return (conn.execute("SELECT count(*) AS n FROM checkpoints WHERE thread_id = %s", (thread_id,)).fetchone()["n"],
                    conn.execute("SELECT count(*) AS n FROM conversation_messages WHERE thread_id = %s",
                                 (thread_id,)).fetchone()["n"])

    assert counts(conv.thread_id)[0] > 0
    assert store.delete(OWNER, conv.thread_id)
    assert counts(conv.thread_id) == (0, 0)
    assert store.get(OWNER, conv.thread_id) is None
    assert counts(keep.thread_id)[0] > 0 and counts(keep.thread_id)[1] == 2  # other threads untouched


def test_database_outage_gives_actionable_error():
    with pytest.raises(memory.DatabaseError, match="Could not connect"):
        memory.open_pool("postgresql://nobody:nothing@127.0.0.1:9/none", wait_seconds=2)


def test_stream_ending_early_is_resumable_not_finalized(store, tracer, fake_model, search_client):
    """Regression: a stream that stops before the graph finishes must not store an empty answer."""
    from agents import build_agent

    class StopsEarly:
        def __init__(self, agent, cut):
            self.agent, self.cut = agent, cut

        def stream(self, *args, **kwargs):
            for i, item in enumerate(self.agent.stream(*args, **kwargs)):
                yield item
                if self.cut and i == 0:
                    return  # e.g. the consumer was interrupted mid-run

        def get_state(self, *args, **kwargs):
            return self.agent.get_state(*args, **kwargs)

    calls = {"n": 0}

    def factory(**kw):
        calls["n"] += 1
        return StopsEarly(build_agent(**{**kw, "model": fake_model}), cut=calls["n"] == 2)

    runner = Runner(store, tracer, search_client=search_client, agent_factory=factory)
    conv = store.create(OWNER)
    out, _ = submit(runner, conv.thread_id, "Notion")  # factory call 1: state read, 2: cut stream
    assert out.status == "failed" and "Retry" in out.error
    assert [m["role"] for m in store.messages(conv.thread_id)] == ["user"]  # no empty answer stored
    out = runner.retry(OWNER, conv.thread_id)
    assert out.status == "completed" and out.message["kind"] == "report"
