"""Repeatable evaluation experiments over eval_dataset.json.

    python manage.py eval                         # offline: fixtures + real guardrails, free
    python manage.py eval --langfuse              # same, recorded as a Langfuse dataset run
    python manage.py eval --mode live --confirm-cost [--items ID ...]   # real agent (paid)
    python manage.py eval --judge --confirm-cost  # add LLM-as-judge scores (paid)

Offline mode scores the stored fixture responses, so it checks the evaluators and guardrails
without calling any model. Live mode runs every turn through the real application path
(input guard, agent, Postgres memory, output guard) in throwaway conversations owned by
"eval:<run name>", deleted afterwards unless --keep-threads is given.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config
import evaluation
import guards

COST_NOTE = """Paid mode requested. Approximate cost per research item (varies with depth/model):
  live agent : ~15-40 model calls and ~5-40 Tavily searches (basic is cheapest)
  --judge    : +1 judge-model call per item (JUDGE_MODEL, default MODEL)
Re-run with --confirm-cost to proceed, and use --items to limit the run."""


def add_arguments(p):
    p.add_argument("--mode", choices=["offline", "live"], default="offline")
    p.add_argument("--judge", action="store_true", help="add LLM-as-judge rubric scores (paid)")
    p.add_argument("--judge-model", default=None, help="judge model (default: JUDGE_MODEL or MODEL)")
    p.add_argument("--confirm-cost", action="store_true", help="required for paid modes")
    p.add_argument("--items", nargs="*", help="only run these item ids")
    p.add_argument("--langfuse", action="store_true", help="upload the dataset and record a Langfuse dataset run")
    p.add_argument("--run-name", default=None)
    p.add_argument("--keep-threads", action="store_true", help="live mode: keep the eval conversations")
    p.add_argument("--output", default="eval_results", help="folder for the JSON results")
    return p


def _field(item, name):
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _text(turn: dict) -> str:
    return "x" * (guards.MAX_INPUT_CHARS + 500) if turn["text"] == "__OVERSIZED__" else turn["text"]


# ---------------------------------------------------------------- tasks
def offline_task(item) -> dict:
    """Guardrails run for real; the agent's response comes from the item's fixture."""
    spec, meta = _field(item, "input"), _field(item, "metadata") or {}
    fixture = meta.get("fixture") or {}
    final = spec["turns"][-1]
    decision_in = guards.validate("input", _text(final))
    out = {"guardrail_input": decision_in.action, "kind": None, "response": "", "evidence": [],
           "guardrail_output": None, "question": decision_in.text or ""}
    if not decision_in.usable or not fixture:
        return out
    decision_out = guards.validate(fixture["kind"], fixture["response"])
    return {**out, "kind": fixture["kind"], "response": decision_out.text, "evidence": fixture.get("evidence", []),
            "guardrail_output": decision_out.action}


class LiveTask:
    """Runs every turn through the real Runner (paid model + search calls)."""

    def __init__(self, run_name: str, tracer):
        import memory
        from runner import Runner

        self.pool = memory.open_pool(max_size=4)
        if problems := memory.schema_problems(self.pool):
            raise memory.DatabaseError("Run `python manage.py init-db` first: " + "; ".join(problems))
        self.store = memory.ConversationStore(self.pool, memory.checkpointer(self.pool))
        self.runner = Runner(self.store, tracer)
        self.owner = f"eval:{run_name}"
        self.threads: list[str] = []

    def __call__(self, *, item, **kwargs) -> dict:
        from langfuse import get_client

        spec = _field(item, "input")
        settings = {"depth": spec.get("depth", "basic"), "hitl": False, "model": config.DEFAULT_MODEL}
        threads: dict[str, str] = {}
        outcome, final = None, spec["turns"][-1]
        for i, turn in enumerate(spec["turns"]):
            if turn["thread"] not in threads:
                threads[turn["thread"]] = self.store.create(self.owner, f"eval {_field(item, 'id')}", settings).thread_id
                self.threads.append(threads[turn["thread"]])
            trace_id = None
            if i == len(spec["turns"]) - 1 and self.runner.tracer.enabled:
                trace_id = get_client().get_current_trace_id()  # link the final turn to the dataset run
            outcome = self.runner.submit(self.owner, threads[turn["thread"]], _text(turn), str(uuid.uuid4()),
                                         settings, trace_id=trace_id)
        thread_id = threads[final["thread"]]
        msgs = self.store.messages(thread_id)
        user = next((m for m in reversed(msgs) if m["role"] == "user"), {})
        msg = outcome.message or {}
        return {"status": outcome.status, "kind": msg.get("kind"), "response": msg.get("content", ""),
                "guardrail_input": (user.get("guardrail") or {}).get("action"),
                "guardrail_output": (msg.get("guardrail") or {}).get("action") if msg.get("kind") != "blocked" else None,
                "evidence": self.store.evidence(thread_id=thread_id), "question": user.get("content", ""),
                "trace_id": msg.get("trace_id"), "error": outcome.error}

    def cleanup(self, keep: bool) -> None:
        if not keep:
            for thread_id in self.threads:
                self.store.delete(self.owner, thread_id)
        self.pool.close()


# ---------------------------------------------------------------- scoring
def score_item(*, input, output, expected_output, metadata, judge_model: str | None = None) -> list[dict]:
    """Deterministic scores + expectation checks (+ optional judge) for one dataset item."""
    expected = expected_output or {}
    results: list[dict] = []

    def check(name, ok: bool, comment: str):
        results.append({"name": f"check.{name}", "value": 1.0 if ok else 0.0, "comment": comment,
                        "status": "ok", "source": evaluation.EVALUATOR_VERSION})

    for key in ("guardrail_input", "guardrail_output", "kind"):
        if key in expected:
            check(key, output.get(key) == expected[key], f"expected {expected[key]}, got {output.get(key)}")
    for term in expected.get("absent_terms", []):
        check("absent_terms", term not in (output.get("response") or ""), f"'{term}' must not appear")

    if output.get("kind") in ("report", "answer") and output.get("response"):
        e = evaluation.EvalInput(kind=output["kind"], response=output["response"],
                                 question=output.get("question") or input["turns"][-1]["text"],
                                 target=input["turns"][0]["text"] if output["kind"] == "report" else None,
                                 depth=input.get("depth"), evidence=output.get("evidence") or None,
                                 required_terms=expected.get("required_terms", []))
        scores = evaluation.evaluate(e)
        if judge_model:
            try:
                scores += evaluation.llm_judge(e, judge_model)
            except Exception as exc:
                results.append({"name": "judge_error", "value": None, "comment": type(exc).__name__,
                                "status": "unavailable", "source": f"llm-judge:{judge_model}"})
        by_name = {s.name: s for s in scores}
        results += [s.as_dict() for s in scores]
        for name, minimum in expected.get("min_scores", {}).items():
            v = by_name[name].value if name in by_name else None
            check(f"min_{name}", v is not None and v >= minimum, f"{name}={v} (minimum {minimum})")
        for name, maximum in expected.get("max_scores", {}).items():
            v = by_name[name].value if name in by_name else None
            check(f"max_{name}", v is not None and v <= maximum, f"{name}={v} (maximum {maximum})")
        for name in expected.get("unavailable", []):
            check(f"unavailable_{name}", name in by_name and by_name[name].status == "unavailable",
                  f"{name} must be unavailable without evidence")
    return results


def _as_langfuse(results: list[dict]):
    from langfuse import Evaluation

    return [Evaluation(name=r["name"], value=r["value"], comment=r["comment"][:500],
                       metadata={"evaluator": r["source"]}) for r in results if r["value"] is not None]


# ---------------------------------------------------------------- main
def main(args) -> int:
    data = evaluation.load_dataset()
    items = [i for i in data["items"] if not args.items or i["id"] in args.items]
    if args.mode == "live":  # crafted-fixture cases (e.g. a deliberately leaked key) cannot be reproduced live
        skipped = [i["id"] for i in items if i.get("live") is False]
        items = [i for i in items if i.get("live") is not False]
        if skipped:
            print(f"Skipping offline-only items in live mode: {', '.join(skipped)}")
    if not items:
        print("No dataset items match --items.")
        return 1
    paid = args.mode == "live" or args.judge
    if paid and not args.confirm_cost:
        print(COST_NOTE)
        return 2
    judge_model = (args.judge_model or config.get("JUDGE_MODEL") or config.DEFAULT_MODEL) if args.judge else None
    run_name = args.run_name or f"{args.mode}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    meta = {"mode": args.mode, "evaluator": evaluation.EVALUATOR_VERSION,
            "judge": f"{judge_model}:{evaluation.JUDGE_RUBRIC_VERSION}" if judge_model else "none",
            "model": config.DEFAULT_MODEL if args.mode == "live" else "fixtures"}

    from observability import Tracer

    tracer = Tracer() if (args.langfuse or args.mode == "live") else None
    if args.langfuse and not (tracer and tracer.enabled):
        print(f"--langfuse needs Langfuse configured: {tracer.settings.reason}")
        return 1
    live = LiveTask(run_name, tracer) if args.mode == "live" else None
    task = live or (lambda *, item, **kw: offline_task(item))

    def evaluator(*, input, output, expected_output, metadata, **kw):
        return _as_langfuse(score_item(input=input, output=output, expected_output=expected_output,
                                       metadata=metadata, judge_model=judge_model))

    records = []
    started = time.monotonic()
    try:
        if args.langfuse:
            client = tracer.client
            client.create_dataset(name=data["name"], description=data["description"])
            for item in items:
                client.create_dataset_item(dataset_name=data["name"], id=item["id"], input=item["input"],
                                           expected_output=item["expected_output"],
                                           metadata={"category": item["category"], "description": item["description"],
                                                     "fixture": item.get("fixture")})
            dataset = client.get_dataset(data["name"])
            wanted = {i["id"] for i in items}
            dataset.items = [i for i in dataset.items if i.id in wanted]
            result = dataset.run_experiment(name=f"mra-{args.mode}", run_name=run_name, task=task,
                                            evaluators=[evaluator], max_concurrency=1 if live else 4, metadata=meta)
            for r in result.item_results:
                records.append({"id": r.item.id, "output": r.output,
                                "scores": [{"name": e.name, "value": e.value, "comment": e.comment}
                                           for e in r.evaluations]})
            tracer.flush()
            print(f"Langfuse dataset run '{run_name}' recorded on dataset '{data['name']}'.")
        else:
            for item in items:
                spec = {"id": item["id"], "input": item["input"], "metadata": {"fixture": item.get("fixture")}}
                output = task(item=spec)
                records.append({"id": item["id"], "output": output,
                                "scores": score_item(input=item["input"], output=output,
                                                     expected_output=item["expected_output"],
                                                     metadata=spec["metadata"], judge_model=judge_model)})
    finally:
        if live:
            live.cleanup(args.keep_threads)

    failed = []
    print(f"\n{'item':<34} {'checks':<9} scores")
    for rec in records:
        checks = [s for s in rec["scores"] if s["name"].startswith("check.")]
        bad = [s for s in checks if s["value"] == 0.0]
        failed += [(rec["id"], s) for s in bad]
        quality = ", ".join(f"{s['name']}={'n/a' if s['value'] is None else round(s['value'], 2)}"
                            for s in rec["scores"] if not s["name"].startswith("check."))
        print(f"{rec['id']:<34} {len(checks) - len(bad)}/{len(checks):<7} {quality}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.json"
    path.write_text(json.dumps({"run": run_name, "metadata": meta, "definitions": evaluation.SCORE_DEFINITIONS,
                                "seconds": round(time.monotonic() - started, 1), "results": records},
                               indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {path}")
    for item_id, s in failed:
        print(f"FAILED {item_id}: {s['name']} — {s['comment']}")
    return 1 if failed else 0
