import json
from pathlib import Path

import pytest

from trainjudge import eval_label, evaluation, tasks, verdict, verification

DEMO = Path(__file__).parent.parent / "demo"
INTENTS = DEMO / "domains" / "customer_support" / "intent_routing" / "data.jsonl"
LABELS = {"refund_request", "billing_dispute", "order_status"}


@pytest.mark.parametrize(
    "output, expected",
    [
        ("refund_request", "refund_request"),
        ("  Refund_Request.\n", "refund_request"),
        ("**refund_request**", "refund_request"),
        ("`billing_dispute`", "billing_dispute"),
        ("Intent: order_status", "order_status"),
        ("Label:order_status", "order_status"),
        ("The intent is billing_dispute.", "billing_dispute"),
        ("<think>\nmaybe order_status\n</think>\nrefund_request", "refund_request"),
        ("```\nrefund_request\n```", "refund_request"),
        # One label inside a sentence counts; hedging between two doesn't.
        ("This message is a billing_dispute, clearly.", "billing_dispute"),
        ("Either refund_request or billing_dispute.", "either refund_request or billing_dispute"),
        # A label only matches as a whole word.
        ("order_status_pending", "order_status_pending"),
        ("shipping", "shipping"),
        ("", None),
        ("<think>hmm</think>", None),
    ],
)
def test_extract_label(output, expected):
    assert eval_label.extract_label(output, LABELS) == expected


@pytest.mark.parametrize(
    "output, outcome",
    [
        ("refund_request", eval_label.CORRECT),
        ("billing_dispute", eval_label.WRONG_LABEL),
        ("shipping_issue", eval_label.UNKNOWN_LABEL),
        ("", eval_label.NO_ANSWER),
    ],
)
def test_score_example(output, outcome):
    assert eval_label.score_example("p", "refund_request", output, LABELS).outcome == outcome


def test_empty_gold_is_not_scored():
    report = eval_label.evaluate([{"prompt": "a", "completion": ""}, {"prompt": "b", "completion": "x"}],
                                 ["x", "x"])  # fmt: skip
    assert report.outcomes()[eval_label.GOLD_ERROR] == 1
    assert len(report.scored) == 1 and report.accuracy == 1.0


def test_report_metrics():
    rows = [{"prompt": str(i), "completion": gold} for i, gold in enumerate(["a", "a", "a", "b"])]
    report = eval_label.evaluate(rows, ["a", "a", "b", "b"])
    assert report.accuracy == 0.75
    stats = report.per_label()
    assert stats["a"] == {
        "precision": 1.0,
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(0.8),
        "support": 3,
    }
    assert stats["b"]["precision"] == 0.5 and stats["b"]["recall"] == 1.0
    assert report.macro_f1 == pytest.approx((0.8 + 2 / 3) / 2)
    assert report.confusions() == [{"gold": "a", "predicted": "b", "count": 1}]
    result = report.to_dict()
    assert result["metric"] == "label_accuracy"
    assert result["secondary"]["label"] == "Macro-F1"


def test_label_detection():
    labels = ["refund_request", "order_status", "complaint"] * 10
    assert tasks.detect([{"completion": c} for c in labels]) == tasks.LABEL
    # Unique short answers are not a label set.
    assert not tasks.looks_like_labels([f"answer {i}" for i in range(30)])
    # Neither is one repeated answer, nor sentences.
    assert not tasks.looks_like_labels(["yes"] * 30)
    assert not tasks.looks_like_labels(["The warranty lasts two years from delivery."] * 30)


def test_verify_the_intent_routing_demo(trained_run_from, fake_generator):
    run_dir = trained_run_from(
        INTENTS, goal="route customer messages to one of our support intents", mask_sensitive=True
    )
    gold = {r["prompt"]: r["completion"] for r in evaluation.test_rows(run_dir)}

    def answer(prompt, adapter_path):
        # The base model answers in a sentence with the wrong label; the adapter gets it right.
        return gold[prompt] if adapter_path else "I think this customer has a complaint."

    result = verification.verify_run(run_dir, generate=fake_generator(answer))
    assert result.verdict.outcome == verdict.IMPROVED
    assert result.verdict.metric == "label_accuracy"
    assert result.verdict.secondary_label == "Macro-F1"

    saved = json.loads((run_dir / "eval" / "finetuned.json").read_text(encoding="utf-8"))
    assert saved["task"] == "label" and saved["accuracy"] == 1.0
    assert "refund_request" in saved["per_label"]

    report = (run_dir / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert "## Task metric: Label accuracy" in report
    assert "| `refund_request` |" in report
    assert "Most common fine-tuned mistakes" not in report  # the adapter makes none
    assert "- Gold: `" in report and "Baseline (" in report
    card = (run_dir / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "text-classification" in card
