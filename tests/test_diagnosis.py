import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainjudge.cli import main
from trainjudge.dataset_audit import audit_dataset
from trainjudge.diagnosis import (
    COST,
    FORMAT,
    KNOWLEDGE,
    PROMPT,
    UNCLEAR,
    diagnose,
    format_diagnosis,
)

DEMO = Path(__file__).parent.parent / "demo"


def run(name, goal, **kwargs):
    return diagnose(goal, audit_dataset(DEMO / name / "data.jsonl"), model="Qwen3-0.6B", **kwargs)


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# The plan's verification gate: SQL is fine-tunable, policy docs is a knowledge gap.
def test_sql_demo_is_a_format_gap():
    d = run("sql_generation", "improve SQL generation for our shop database")
    assert d.classification == FORMAT
    assert d.confidence == "high"
    assert d.fine_tune_recommended
    assert d.profile.output_shape == "sql"
    assert not d.bfsi


def test_policy_demo_is_a_knowledge_gap():
    d = run("policy_docs", "make it answer from our internal support policy documents")
    assert d.classification == KNOWLEDGE
    assert d.confidence == "high"
    assert not d.fine_tune_recommended
    assert d.profile.distinct_completions == 36
    assert d.profile.source_field_rate == 1.0
    assert not d.bfsi  # retail warranty "claims" alone aren't BFSI


def test_bfsi_transactions_is_a_format_gap_with_sensitive_data():
    d = run("bfsi_transactions", "categorize bank transaction narrations into our category JSON")
    assert d.classification == FORMAT
    assert d.profile.output_shape == "json"
    assert d.bfsi
    assert set(d.audit.sensitive) == {
        "card number",
        "Aadhaar number",
        "PAN",
        "UPI ID",
        "phone number",
    }
    text = " ".join(format_diagnosis(d).split())
    assert "Sensitive data in 14 rows" in text
    assert "DPDP Act 2023" in text


def test_bfsi_loan_faq_is_a_knowledge_gap_with_regulated_facts():
    d = run(
        "bfsi_loan_faq",
        "answer customer questions about our loan and FD interest rates and charges",
    )
    assert d.classification == KNOWLEDGE
    assert d.bfsi
    assert "interest rates" in d.regulated_terms
    text = " ".join(format_diagnosis(d).split())
    assert "regulator circulars" in text
    assert "effective dates" in text


def test_cost_goal_is_a_distillation_candidate():
    d = run("bfsi_transactions", "we call an expensive API model for this; distill to a cheaper model")
    assert d.classification == COST
    assert d.fine_tune_recommended
    assert d.secondary == FORMAT


def test_small_dataset_without_prompting_is_a_prompt_gap(tmp_path):
    rows = [{"prompt": f"Reply in JSON for item {i}", "completion": json.dumps({"id": i})} for i in range(20)]
    d = diagnose(
        "make it reply in our JSON format",
        audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows)),
        tried_prompting=False,
    )
    assert d.classification == PROMPT
    assert not d.fine_tune_recommended


def test_tried_prompting_counts_against_prompt_gap(tmp_path):
    rows = [{"prompt": f"Reply in JSON for item {i}", "completion": json.dumps({"id": i})} for i in range(20)]
    audit = audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows))
    goal = "make it reply in our JSON format"
    assert diagnose(goal, audit, tried_prompting=True).classification == FORMAT


def test_grounded_answers_are_format_not_knowledge(tmp_path):
    context = " ".join(f"Clause {i}: the notice period is {i} days for plan {i}." for i in range(20))
    rows = [
        {
            "prompt": f"{context}\nQuestion: what is the notice period for plan {i}?",
            "completion": f"The notice period for plan {i} is {i} days.",
        }
        for i in range(60)
    ]
    d = diagnose("answer questions about contracts", audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows)))
    assert d.profile.grounded
    assert d.classification != KNOWLEDGE


def test_no_signal_is_unclear(tmp_path):
    rows = [{"prompt": f"p{i}", "completion": f"word{i} other{i}"} for i in range(300)]
    d = diagnose("make it better", audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows)))
    assert d.classification == UNCLEAR
    assert "No recommendation" in format_diagnosis(d)


def test_high_stakes_bfsi_goal_warns():
    d = run("bfsi_loan_faq", "fine-tune it to approve or reject personal loan applications")
    assert d.high_stakes_terms == ["approve", "reject"]
    assert "human in the loop" in " ".join(format_diagnosis(d).split())


def test_high_stakes_terms_need_bfsi_context():
    d = run("sql_generation", "reject SQL that doesn't follow our conventions")
    assert d.high_stakes_terms == []


def test_goal_terms_match_whole_words_only():
    d = run("sql_generation", "improve the sqlite dialect output")
    assert "sql" not in " ".join(d.evidence.get(FORMAT, []))


def test_cli_diagnose_text_and_json():
    runner = CliRunner()
    args = [
        "diagnose",
        "--dataset",
        str(DEMO / "policy_docs" / "data.jsonl"),
        "--model",
        "Qwen3-0.6B",
        "--goal",
        "answer from our policy documents",
    ]

    result = runner.invoke(main, args)
    assert result.exit_code == 0
    assert "Classification: KNOWLEDGE GAP" in result.output
    assert "❌ Do not fine-tune for this goal." in result.output

    result = runner.invoke(main, [*args, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["classification"] == "knowledge"
    assert data["fine_tune_recommended"] is False
    assert data["audit"]["counts"]["duplicate"] == 12


@pytest.mark.parametrize("flag, expected", [("--tried-prompting", True), ("--not-tried-prompting", False)])
def test_cli_tried_prompting_flag(flag, expected):
    args = [
        "diagnose",
        "--dataset",
        str(DEMO / "sql_generation" / "data.jsonl"),
        "--model",
        "m",
        "--goal",
        "sql",
        "--json",
        flag,
    ]
    result = CliRunner().invoke(main, args)
    assert json.loads(result.output)["tried_prompting"] is expected
