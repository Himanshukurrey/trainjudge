import json
from pathlib import Path

import pytest

from trainjudge import evaluation, runs, tasks, training, verdict, verification
from trainjudge.eval_json import (
    CORRECT,
    GOLD_ERROR,
    MISSING_FIELDS,
    NO_JSON,
    NOT_AN_OBJECT,
    WRONG_FIELDS,
    evaluate,
    extract_json,
    normalize_value,
    score_example,
)
from trainjudge.regression_check import build_suite

DEMO = Path(__file__).parent.parent / "demo"
GOLD = json.dumps({"diagnosis": "asthma", "icd10": "J45.909", "severity": "mild"})


@pytest.mark.parametrize(
    "output, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('<think>\n\n</think>\n\n{"a": 1}', {"a": 1}),
        ('Here you go:\n```json\n{"a": 1}\n```\nHope that helps!', {"a": 1}),
        ('The answer is {"a": 1} as requested.', {"a": 1}),
        ('Options: [1, 2]. Final: {"a": 1}', {"a": 1}),  # objects beat earlier arrays
        ("[1, 2]", [1, 2]),
        ("{broken json", None),
        ("no json here", None),
    ],
)
def test_extract_json(output, expected):
    assert extract_json(output) == expected


def test_normalize_value():
    assert normalize_value("  Asthma ") == normalize_value("asthma")
    assert normalize_value(3) == normalize_value(3.0) == normalize_value("3")
    assert normalize_value(None) is None
    assert normalize_value(True) is True


@pytest.mark.parametrize(
    "output, outcome, wrong, missing",
    [
        ('{"diagnosis": "Asthma", "icd10": "J45.909", "severity": "MILD"}', CORRECT, [], []),
        ('{"diagnosis": "asthma", "icd10": "J45.909", "severity": "mild", "note": "x"}', CORRECT, [], []),
        ('{"diagnosis": "asthma", "icd10": "J45.9", "severity": "mild"}', WRONG_FIELDS, ["icd10"], []),
        ('{"diagnosis": "asthma", "severity": "mild"}', MISSING_FIELDS, [], ["icd10"]),
        ('["asthma", "J45.909"]', NOT_AN_OBJECT, [], ["diagnosis", "icd10", "severity"]),
        ("The patient has asthma.", NO_JSON, [], ["diagnosis", "icd10", "severity"]),
    ],
)
def test_score_example(output, outcome, wrong, missing):
    result = score_example("p", GOLD, output)
    assert result.outcome == outcome
    assert result.wrong_fields == wrong and result.missing_fields == missing


def test_field_score_and_extra_fields():
    result = score_example("p", GOLD, '{"diagnosis": "asthma", "icd10": "X", "extra": 1}')
    assert result.field_score == pytest.approx(1 / 3)
    assert result.extra_fields == ["extra"]


def test_gold_must_be_an_object():
    assert score_example("p", "not json", "{}").outcome == GOLD_ERROR
    assert score_example("p", "[1]", "{}").outcome == GOLD_ERROR


def test_report_metrics():
    rows = [{"prompt": f"p{i}", "completion": GOLD} for i in range(4)] + [
        {"prompt": "bad", "completion": "x"}
    ]
    outputs = [GOLD, GOLD, '{"diagnosis": "asthma"}', "none", "{}"]
    report = evaluate(rows, outputs)
    assert report.accuracy == 0.5  # the gold error is excluded
    assert report.field_accuracy == pytest.approx((1 + 1 + 1 / 3 + 0) / 4)
    assert report.per_field() == {"diagnosis": 0.75, "icd10": 0.5, "severity": 0.5}
    data = report.to_dict()
    assert data["metric"] == "json_exact_match"
    assert data["secondary"]["label"] == "Field-level accuracy"


def test_task_detection():
    assert tasks.detect([{"completion": GOLD}] * 5) == tasks.JSON
    assert tasks.detect([{"completion": "SELECT 1;"}] * 5) == tasks.SQL
    with pytest.raises(tasks.UnknownTask):
        tasks.detect([{"completion": "The dose is 20 mg."}] * 5)


# --- end to end: a domain demo through verify, with a fake model --------------------


@pytest.fixture
def trained_domain_run(tmp_path):
    dataset = DEMO / "domains" / "education" / "question_tagging" / "data.jsonl"
    prepared = training.prepare_run(dataset, "Qwen3-0.6B", tmp_path, goal="tag exam questions")
    adapters = prepared.run_dir / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"")
    record = runs.read_run_json(prepared.run_dir)
    record["status"] = "trained"
    record["training"] = {"train_loss_drop_pct": 90.0, "duration_s": 60}
    runs.write_run_json(prepared.run_dir, record)
    return prepared.run_dir


def fake_model(run_dir):
    gold = {r["prompt"]: r["completion"] for r in evaluation.test_rows(run_dir)}
    suite = {i.prompt: i for i in build_suite()}
    passing = {
        "bullets": "- a\n- b\n- c", "lowercase": "fine.", "max_words": "short.",
        "ends_with": "Ok. That is all.",
        "json_keys": '{"name": "x", "color": "y"}', "json_list": '["a", "b", "c", "d"]',
        "numbered": "1. a\n2. b\n3. c", "uncertain": "I don't know.",
    }  # fmt: skip

    def generate(model, prompts, adapter_path=None, decoding=None, on_progress=None):
        out = []
        for p in prompts:
            if p in gold:
                # The base model answers in prose; the adapter returns the gold JSON.
                out.append(gold[p] if adapter_path else "This question is about mathematics.")
            else:
                item = suite[p]
                out.append(str(item.arg) if item.check == "number" else passing[item.check])
        return out

    return generate


def test_verify_a_json_domain_demo_without_a_database(trained_domain_run):
    result = verification.verify_run(trained_domain_run, generate=fake_model(trained_domain_run))
    assert result.verdict.outcome == verdict.IMPROVED
    assert result.verdict.metric == "json_exact_match"
    assert result.verdict.secondary_label == "Field-level accuracy"

    saved = json.loads((trained_domain_run / "eval" / "finetuned.json").read_text(encoding="utf-8"))
    assert saved["task"] == "json" and saved["database"] is None
    assert set(saved["per_field"]) == {"subject", "topic", "difficulty"}

    report = (trained_domain_run / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert "## Task metric: JSON exact-match accuracy" in report
    assert "| `difficulty` |" in report
    assert "Database:" not in report and "--db" not in report
    card = (trained_domain_run / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "structured-extraction" in card and "exact_match" in card
    assert "JSON exact-match accuracy" in verdict.format_box(result.verdict)


def test_sql_task_still_needs_a_database(tmp_path):
    prepared = training.prepare_run(DEMO / "sql_generation" / "data.jsonl", "m", tmp_path)
    with pytest.raises(runs.RunError, match="needs --db"):
        evaluation.resolve_task(prepared.run_dir)
    assert evaluation.resolve_task(prepared.run_dir, db_path=DEMO / "sql_generation" / "shop.sql")[0] == "sql"


def test_prose_answers_have_no_automatic_eval(tmp_path):
    prepared = training.prepare_run(DEMO / "policy_docs" / "data.jsonl", "m", tmp_path)
    with pytest.raises(runs.RunError, match="no automatic eval"):
        evaluation.resolve_task(prepared.run_dir)
