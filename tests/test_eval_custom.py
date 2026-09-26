import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainjudge import eval_custom, evaluation, runs, verdict, verification
from trainjudge.cli import main
from trainjudge.scorers import key_facts

DEMO = Path(__file__).parent.parent / "demo"
POLICY = DEMO / "policy_docs" / "data.jsonl"
KEY_FACTS = Path(key_facts.__file__)


def write_scorer(tmp_path, body, name="scorer.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_load_a_file_and_default_function(tmp_path):
    path = write_scorer(tmp_path, "def score(prompt, output, gold):\n    return output == gold\n")
    scorer = eval_custom.load(str(path))
    assert scorer.spec == f"{path.resolve()}:score"
    assert scorer.sha256 and not scorer.wants_row
    assert eval_custom.load(f"{path}:score").spec == scorer.spec


def test_load_a_named_function_and_a_module(tmp_path):
    path = write_scorer(tmp_path, "def check(prompt, output, gold, row):\n    return True\n")
    assert eval_custom.load(f"{path}:check").wants_row
    scorer = eval_custom.load("trainjudge.textutil:strip_think")
    assert scorer.spec == "trainjudge.textutil:strip_think" and scorer.sha256


def test_relative_path_resolves_against_base(tmp_path):
    write_scorer(tmp_path, "def score(p, o, g):\n    return True\n")
    assert eval_custom.load("scorer.py", base=tmp_path).source == (tmp_path / "scorer.py").resolve()


@pytest.mark.parametrize(
    "spec, message",
    [
        ("missing.py", "not found"),
        ("no_such_module_xyz:score", "can't import"),
        ("trainjudge.textutil:nothing_here", "is not a function"),
    ],
)
def test_load_errors(spec, message):
    with pytest.raises(eval_custom.ScorerError, match=message):
        eval_custom.load(spec)


def test_import_error_is_reported(tmp_path):
    path = write_scorer(tmp_path, "raise RuntimeError('boom')\n")
    with pytest.raises(eval_custom.ScorerError, match="failed to import: RuntimeError: boom"):
        eval_custom.load(str(path))


def test_windows_drive_paths_are_not_split_on_the_drive_colon():
    assert eval_custom._split(r"C:\scorers\s.py") == (r"C:\scorers\s.py", "score")
    assert eval_custom._split(r"C:\scorers\s.py:check") == (r"C:\scorers\s.py", "check")


def test_evaluate_bools_dicts_and_errors(tmp_path):
    path = write_scorer(
        tmp_path,
        "def score(prompt, output, gold, row):\n"
        "    if prompt == 'boom':\n        raise ValueError('bad row')\n"
        "    if prompt == 'partial':\n        return {'correct': False, 'score': 0.5, 'reason': 'half'}\n"
        "    assert row['prompt'] == prompt\n"
        "    return output == gold\n",
    )
    rows = [{"prompt": p, "completion": "yes"} for p in ("a", "b", "partial", "boom")]
    rows.append({"prompt": "no gold"})
    report = eval_custom.evaluate(
        eval_custom.load(str(path)), rows, ["<think>x</think>yes", "no", "", "", ""]
    )
    assert [e.outcome for e in report.examples] == [
        "correct",
        "incorrect",
        "incorrect",
        "scorer_error",
        "gold_error",
    ]
    assert report.examples[3].reason == "ValueError: bad row"
    assert report.accuracy == 0.25
    assert report.mean_score == pytest.approx(1.5 / 4)
    result = report.to_dict()
    assert result["metric"] == "custom_accuracy" and result["scorer"]["spec"].endswith(":score")


@pytest.mark.parametrize("value", ["'yes'", "0.7", "{'score': 1}", "{'correct': True, 'score': 2}"])
def test_bad_return_values_are_scorer_errors(tmp_path, value):
    path = write_scorer(tmp_path, f"def score(p, o, g):\n    return True if p == 'ok' else {value}\n")
    rows = [{"prompt": "ok", "completion": "x"}, {"prompt": "bad", "completion": "x"}]
    report = eval_custom.evaluate(eval_custom.load(str(path)), rows, ["x", "x"])
    assert report.examples[1].outcome == eval_custom.SCORER_ERROR


def test_a_scorer_that_always_fails_raises(tmp_path):
    path = write_scorer(tmp_path, "def score(p, o, g):\n    return 1 / 0\n")
    with pytest.raises(eval_custom.ScorerError, match="failed on every example.*ZeroDivisionError"):
        eval_custom.evaluate(eval_custom.load(str(path)), [{"prompt": "a", "completion": "b"}], ["b"])


def test_example_key_facts_scorer():
    scorer = eval_custom.load(str(KEY_FACTS))
    gold = "Small appliances carry a 2-year limited warranty from the date of delivery."
    right = scorer.function("q", "They have a 2-year limited warranty, counted from delivery date.", gold)
    wrong_number = scorer.function(
        "q", "Small appliances carry a 1-year limited warranty from delivery.", gold
    )
    assert right["correct"] and right["score"] > 0.6
    assert not wrong_number["correct"] and "2" in wrong_number["reason"]


def test_verify_prose_answers_with_a_scorer(trained_run_from, fake_generator):
    run_dir = trained_run_from(POLICY, goal="answer from our support policies")
    with pytest.raises(runs.RunError, match="--scorer"):
        evaluation.resolve_task(run_dir)  # prose has no automatic eval
    gold = {r["prompt"]: r["completion"] for r in evaluation.test_rows(run_dir)}

    def answer(prompt, adapter_path):
        return gold[prompt] if adapter_path else "Please check our website for details."

    generate = fake_generator(answer)
    result = verification.verify_run(run_dir, generate=generate, scorer=str(KEY_FACTS))
    assert result.verdict.outcome == verdict.IMPROVED
    assert result.verdict.metric == "custom_accuracy"
    assert result.verdict.secondary_label == "Mean score"

    record = runs.read_run_json(run_dir)
    assert record["task"]["type"] == "custom"
    assert record["task"]["scorer"] == f"{KEY_FACTS.resolve()}:score"
    report = (run_dir / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert "## Task metric: Custom scorer accuracy" in report
    assert f"- Scorer: `{KEY_FACTS.resolve()}:score`" in report
    assert "--scorer" in report and "missing:" in report

    # The recorded scorer is reused, so a second verify needs no --scorer ...
    again = verification.verify_run(run_dir, generate=generate)
    assert again.verdict.outcome == verdict.IMPROVED


def test_changing_the_scorer_invalidates_saved_evals(trained_run_from, fake_generator, tmp_path):
    run_dir = trained_run_from(POLICY)
    gold = {r["prompt"]: r["completion"] for r in evaluation.test_rows(run_dir)}
    path = write_scorer(tmp_path, "def score(p, o, g):\n    return o == g\n")
    calls = []

    def answer(prompt, adapter_path):
        calls.append(prompt)
        return gold[prompt]

    generate = fake_generator(answer)
    verification.verify_run(run_dir, generate=generate, scorer=str(path))
    first = len(calls)
    verification.verify_run(run_dir, generate=generate, scorer=str(path))
    assert len(calls) == first  # same scorer: saved evals reused
    path.write_text("def score(p, o, g):\n    return len(o) > 0\n", encoding="utf-8")
    verification.verify_run(run_dir, generate=generate, scorer=str(path))
    assert len(calls) > first  # edited scorer: re-scored


def test_scorer_only_applies_to_the_custom_task(trained_run_from):
    run_dir = trained_run_from(DEMO / "domains" / "education" / "question_tagging" / "data.jsonl")
    with pytest.raises(runs.RunError, match="only applies to --task custom"):
        evaluation.resolve_task(run_dir, "json", scorer=str(KEY_FACTS))
    with pytest.raises(runs.RunError, match="needs --scorer"):
        evaluation.resolve_task(run_dir, "custom")


def test_cli_eval_accepts_scorer(trained_run_from):
    run_dir = trained_run_from(POLICY)
    result = CliRunner().invoke(main, ["eval", str(run_dir), "--scorer", "missing_scorer.py"])
    assert result.exit_code != 0 and "not found" in result.output
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("task") is None


def test_builtin_example_scorer_loads_by_module_name():
    scorer = eval_custom.load("trainjudge.scorers.key_facts")
    assert scorer.spec == "trainjudge.scorers.key_facts:score" and scorer.source == KEY_FACTS
