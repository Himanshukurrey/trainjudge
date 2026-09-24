import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainjudge import evaluation, regression_check, runs, training, verdict, verification
from trainjudge.cli import main
from trainjudge.regression_check import (
    FORMAT_COMPLIANCE,
    HALLUCINATION,
    INSTRUCTION_FOLLOWING,
    build_suite,
)

DEMO = Path(__file__).parent.parent / "demo" / "sql_generation"
SHOP = DEMO / "shop.sql"


# --- regression checks -------------------------------------------------------


@pytest.mark.parametrize(
    "check, arg, text, expected",
    [
        ("bullets", 3, "- a\n- b\n- c", True),
        ("bullets", 3, "- a\n- b", False),
        ("bullets", 3, "Here you go:\n- a\n- b\n- c", False),
        ("lowercase", None, "coffee is a drink.", True),
        ("lowercase", None, "Coffee is a drink.", False),
        ("max_words", 14, "short answer here", True),
        ("max_words", 3, "one two three four", False),
        ("ends_with", "That is all.", "Chess is old. That is all.", True),
        ("ends_with", "That is all.", "Chess is old. **That is all.**", True),
        ("ends_with", "That is all.", "Chess is old.", False),
        ("number", 42, "42", True),
        ("number", 42, "**42**", True),
        ("number", 42, "17 + 25 = 42.", False),
        ("number", 425, "425.", True),
        ("json_keys", ["name", "color"], '{"name": "apple", "color": "red"}', True),
        ("json_keys", ["name", "color"], '```json\n{"name": "apple", "color": "red"}\n```', True),
        ("json_keys", ["name", "color"], '{"name": "apple"}', False),
        ("json_keys", ["name", "color"], "name: apple", False),
        ("json_list", 4, '["a", "b", "c", "d"]', True),
        ("json_list", 4, "[0, 1, 2, 3]", False),
        ("json_list", 4, '["a", "b"]', False),
        ("numbered", 3, "1. a\n2. b\n3. c", True),
        ("numbered", 3, "1. a", False),
        ("uncertain", None, "I'm not aware of any such prize.", True),
        ("uncertain", None, "That town doesn't seem to exist.", True),
        ("uncertain", None, "It was won by Dr. Jane Smith.", False),
    ],
)
def test_checks(check, arg, text, expected):
    assert regression_check.CHECKS[check](text, arg) is expected


def test_suite_is_deterministic_and_balanced():
    suite = build_suite()
    assert suite == build_suite()
    assert len({i.id for i in suite}) == len(suite) == 60
    by_category = {c: sum(i.category == c for i in suite) for c in regression_check.CATEGORIES}
    assert by_category == {INSTRUCTION_FOLLOWING: 30, FORMAT_COMPLIANCE: 18, HALLUCINATION: 12}


def test_think_blocks_are_ignored():
    item = next(i for i in build_suite() if i.check == "number")
    [result] = regression_check.score([item], [f"<think>\nlet me add\n</think>\n\n{item.arg}"])
    assert result.passed


# --- verdict -----------------------------------------------------------------


def test_mcnemar():
    assert verdict.mcnemar_p(0, 0) == 1.0
    assert verdict.mcnemar_p(5, 5) == 1.0
    assert verdict.mcnemar_p(10, 0) == pytest.approx(2 / 2**10)
    assert verdict.mcnemar_p(89, 2) < 1e-20


def task_eval(correct: list[bool]) -> dict:
    examples = [
        {"prompt": f"q{i}", "outcome": "correct" if c else "wrong_result"} for i, c in enumerate(correct)
    ]
    return {
        "metric": "sql_execution_accuracy",
        "accuracy": sum(correct) / len(correct),
        "lenient_accuracy": sum(correct) / len(correct),
        "scored": len(correct),
        "examples": examples,
    }


def reg_eval(rates: dict[str, float]) -> dict:
    return {
        "summary": {
            c: {"rate": rates.get(c, 0.8), "total": 20, "passed": 0} for c in regression_check.CATEGORIES
        }
    }


def test_improved():
    v = verdict.decide(
        task_eval([False] * 60 + [True] * 40),
        task_eval([True] * 90 + [False] * 10),
        reg_eval({}),
        reg_eval({}),
        train_loss_drop_pct=80,
    )
    assert v.outcome == verdict.IMPROVED
    assert v.gained == 60 and v.lost == 10
    assert v.significant


def test_regressed_when_general_capability_drops():
    v = verdict.decide(
        task_eval([False] * 60 + [True] * 40),
        task_eval([True] * 100),
        reg_eval({INSTRUCTION_FOLLOWING: 0.8}),
        reg_eval({INSTRUCTION_FOLLOWING: 0.7}),
    )
    assert v.outcome == verdict.REGRESSED
    assert any("instruction-following regressed -10.0" in r for r in v.reasons)


def test_small_drop_within_tolerance_is_not_a_regression():
    v = verdict.decide(
        task_eval([False] * 60 + [True] * 40),
        task_eval([True] * 100),
        reg_eval({FORMAT_COMPLIANCE: 0.80}),
        reg_eval({FORMAT_COMPLIANCE: 0.76}),
    )
    assert v.outcome == verdict.IMPROVED


def test_rejected_when_gain_is_small_even_if_loss_dropped():
    v = verdict.decide(
        task_eval([True] * 61 + [False] * 39),
        task_eval([True] * 63 + [False] * 37),
        reg_eval({}),
        reg_eval({}),
        train_loss_drop_pct=34,
    )
    assert v.outcome == verdict.REJECTED
    assert "training loss dropped 34%" in v.reasons[0]


def test_rejected_when_gain_is_not_significant():
    base = [True] * 50 + [False] * 50
    tuned = base[:]
    for i in range(50, 56):
        tuned[i] = True  # 6 gained
    for i in range(2):
        tuned[i] = False  # 2 lost -> +4 points, p ~ 0.29
    v = verdict.decide(task_eval(base), task_eval(tuned), reg_eval({}), reg_eval({}))
    assert v.improvement_points == pytest.approx(4.0)
    assert v.outcome == verdict.REJECTED
    assert "isn't statistically significant" in v.reasons[0]


def test_decide_rejects_mismatched_test_sets():
    with pytest.raises(ValueError):
        verdict.decide(task_eval([True]), task_eval([True, False]), reg_eval({}), reg_eval({}))


def test_box_has_consistent_width():
    v = verdict.decide(
        task_eval([False] * 60 + [True] * 40),
        task_eval([True] * 100),
        reg_eval({INSTRUCTION_FOLLOWING: 0.8}),
        reg_eval({INSTRUCTION_FOLLOWING: 0.5}),
    )
    lines = verdict.format_box(v).splitlines()
    assert len({len(line) for line in lines}) == 1
    assert "TRAINJUDGE VERDICT" in lines[1]
    assert any("⚠ REGRESSED" in line for line in lines)


# --- end to end with a fake model --------------------------------------------


@pytest.fixture
def trained_run(tmp_path):
    prepared = training.prepare_run(
        DEMO / "data.jsonl", "Qwen3-0.6B", tmp_path, goal="improve SQL generation"
    )
    adapters = prepared.run_dir / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"")
    record = runs.read_run_json(prepared.run_dir)
    record["status"] = "trained"
    record["training"] = {
        "train_loss_drop_pct": 90.0,
        "first_train_loss": 2.0,
        "final_train_loss": 0.2,
        "first_val_loss": 2.5,
        "final_val_loss": 0.3,
        "best_val_loss": 0.3,
        "duration_s": 60,
    }
    runs.write_run_json(prepared.run_dir, record)
    return prepared.run_dir


def good_model(run_dir, break_general=False):
    gold = {r["prompt"]: r["completion"] for r in evaluation.test_rows(run_dir)}
    suite = {i.prompt: i for i in build_suite()}
    answers = {
        "bullets": "- a\n- b\n- c",
        "lowercase": "fine.",
        "max_words": "short.",
        "ends_with": "Ok. That is all.",
        "json_keys": '{"name": "x", "color": "y"}',
        "json_list": '["a", "b", "c", "d"]',
        "numbered": "1. a\n2. b\n3. c",
        "uncertain": "I don't know.",
    }

    def generate(model, prompts, adapter_path=None, decoding=None, on_progress=None):
        out = []
        for p in prompts:
            if p in gold:
                out.append(gold[p] if adapter_path else "I can't answer from this schema.")
            else:
                item = suite[p]
                if adapter_path and break_general:
                    out.append("SELECT 1;")
                elif item.check == "number":
                    out.append(str(item.arg))
                else:
                    out.append(answers[item.check])
        return out

    return generate


def test_verify_run_improved(trained_run):
    result = verification.verify_run(trained_run, SHOP, generate=good_model(trained_run))
    assert result.verdict.outcome == verdict.IMPROVED
    assert [p.name for p in result.artifacts] == [
        "MODEL_CARD.md",
        "EXPERIMENT_REPORT.md",
        "eval_results.json",
    ]
    results = json.loads((trained_run / "eval_results.json").read_text(encoding="utf-8"))
    assert results["verdict"] == "IMPROVED"
    assert results["task"]["finetuned"] == 1.0
    report = (trained_run / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert "Verdict: ✓ IMPROVED" in report and "## Reproduce" in report
    card = (trained_run / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert card.startswith("---\nbase_model: Qwen/Qwen3-0.6B")
    assert runs.read_run_json(trained_run)["verdict"]["outcome"] == "IMPROVED"


def test_verify_run_regressed(trained_run):
    result = verification.verify_run(trained_run, SHOP, generate=good_model(trained_run, break_general=True))
    assert result.verdict.outcome == verdict.REGRESSED
    report = (trained_run / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert "General-capability items that broke" in report
    assert "Not recommended for deployment" in (trained_run / "MODEL_CARD.md").read_text(encoding="utf-8")


def test_verify_reuses_saved_evals(trained_run):
    verification.verify_run(trained_run, SHOP, generate=good_model(trained_run))

    def fail(*args, **kwargs):
        raise AssertionError("should have reused saved evals")

    logs = []
    verification.verify_run(trained_run, generate=fail, log=logs.append)
    assert any("Using saved" in line for line in logs)


def test_verify_reruns_limited_evals(trained_run):
    gen = good_model(trained_run)
    evaluation.evaluate_target(trained_run, "baseline", SHOP, limit=5, generate=gen)
    logs = []
    verification.verify_run(trained_run, SHOP, generate=gen, log=logs.append)
    assert "Evaluating baseline model on the held-out test split..." in logs


def test_verify_requires_trained_run(tmp_path):
    prepared = training.prepare_run(DEMO / "data.jsonl", "m", tmp_path)
    with pytest.raises(runs.RunError, match="hasn't finished training"):
        verification.verify_run(prepared.run_dir, SHOP)


def test_cli_verify_needs_run_json(tmp_path):
    result = CliRunner().invoke(main, ["verify", str(tmp_path)])
    assert result.exit_code != 0
    assert "no run.json" in result.output


def test_single_item_drop_is_not_a_regression():
    # 18 items: one lost item is 5.6 points, past the 5-point tolerance, but noise-level.
    one = verdict.RegressionResult(FORMAT_COMPLIANCE, 18 / 18, 17 / 18, 18, 5.0)
    two = verdict.RegressionResult(FORMAT_COMPLIANCE, 18 / 18, 16 / 18, 18, 5.0)
    assert one.delta_points < -5 and one.items_lost == 1 and not one.regressed
    assert two.items_lost == 2 and two.regressed
