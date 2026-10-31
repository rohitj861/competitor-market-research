"""Evaluation module, the experiment command, and the existing Markdown/PDF export."""

import json

import evaluation
from evaluation import EvalInput
from pdf_export import markdown_to_pdf
from runner import tidy_markdown
from tests.fakes import good_report

EVIDENCE = [
    {"url": "https://acme.example/pricing", "title": "Acme pricing", "content": "Acme costs $12 per user per month."},
    {"url": "https://www.globex.example/news/share/", "title": "Globex", "content": "Globex holds 35% of the market."},
]


def scores(e):
    return {s.name: s for s in evaluation.evaluate(e)}


def test_good_report_scores_high_with_evidence():
    s = scores(EvalInput(kind="report", response=good_report("Notion"), target="Notion", depth="basic",
                         evidence=EVIDENCE))
    assert s["relevance"].value == 1.0
    assert s["completeness"].value == 1.0
    assert s["citation_integrity"].value == 1.0
    assert s["citation_support"].value == 1.0  # www/trailing-slash differences normalised
    assert 0 < s["groundedness"].value < 1  # "$10 per user" is not in the evidence
    assert "$10" in s["groundedness"].comment


def test_evidence_dependent_scores_are_unavailable_without_evidence():
    s = scores(EvalInput(kind="report", response=good_report("Notion"), target="Notion", evidence=None))
    for name in ("citation_support", "groundedness"):
        assert s[name].value is None and s[name].status == "unavailable"


def test_unretrieved_citation_lowers_support_even_if_well_formed():
    report = good_report("Notion").replace("https://globex.example/news/share", "https://made-up.example/x")
    s = scores(EvalInput(kind="report", response=report, target="Notion", evidence=EVIDENCE))
    assert s["citation_integrity"].value == 1.0  # formally correct...
    assert s["citation_support"].value == 0.5  # ...but only half the sources were retrieved


def test_dangling_citations_lower_integrity():
    report = good_report("Notion").replace("[2] Globex market share", "Globex market share")
    assert scores(EvalInput(kind="report", response=report, target="Notion"))["citation_integrity"].value < 1


def test_wrong_target_lowers_relevance_and_required_terms_drive_answer_completeness():
    assert scores(EvalInput(kind="report", response=good_report("Coda"), target="Notion"))["relevance"].value == 0
    s = scores(EvalInput(kind="answer", response="Globex is cheapest.", question="Which competitor is cheapest?",
                         required_terms=["Globex", "Notion"]))
    assert s["completeness"].value == 0.5 and "Notion" in s["completeness"].comment


def test_score_records_carry_evaluator_version():
    s = evaluation.evaluate(EvalInput(kind="answer", response="x", question="y"))[0].as_dict()
    assert s["source"] == evaluation.EVALUATOR_VERSION


def test_offline_experiment_passes_and_writes_results(tmp_path, capsys):
    import manage

    code = manage.main(["eval", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    files = list(tmp_path.glob("offline-*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["metadata"]["judge"] == "none" and data["definitions"]
    ids = {r["id"] for r in data["results"]}
    assert {"thread-isolation", "failure-secret-in-input", "research-no-evidence"} <= ids
    assert "sk-proj-SYNTHETICSYNTHETICSYNTHETIC0000" not in files[0].read_text(encoding="utf-8")  # guards redact it


def test_paid_modes_require_explicit_cost_confirmation(capsys):
    import manage

    assert manage.main(["eval", "--mode", "live"]) == 2
    assert manage.main(["eval", "--judge"]) == 2
    assert "--confirm-cost" in capsys.readouterr().out


def test_markdown_and_pdf_export_still_work():
    md = tidy_markdown("```markdown\n" + good_report("Notion") + "\n```")
    assert md.startswith("# Notion: Competitor Analysis")
    pdf = markdown_to_pdf(md, title="Notion – Competitor Report")
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000
