"""Label accuracy: did the model pick the gold label?

For classification-style tasks (intent routing, sentiment, triage, tagging)
where the gold completion is one short label. The label is pulled out of the
model's raw output, tolerating what a chatty base model adds around it:

- think blocks, code fences, quotes, bold markers and trailing punctuation
- a prefix such as "Label:", "Intent:" or "Category:"
- a sentence around the label, when exactly one known label appears in it

Labels are compared after trimming, collapsing whitespace and ignoring case.
Accuracy is the share of examples with the right label; macro-F1 (the mean
F1 over gold labels, so rare labels count as much as common ones) is reported
alongside it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

from trainjudge.textutil import strip_think

CORRECT = "correct"
WRONG_LABEL = "wrong_label"
UNKNOWN_LABEL = "unknown_label"  # an answer, but not one of the task's labels
NO_ANSWER = "no_answer"
GOLD_ERROR = "gold_error"
OUTCOMES = (CORRECT, WRONG_LABEL, UNKNOWN_LABEL, NO_ANSWER, GOLD_ERROR)

_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:label|intent|category|class|answer|sentiment|tag)(?:\s+is)?(?:\s*[:=-]\s*|\s+)",
    re.IGNORECASE,
)
_STRIP_CHARS = " \t`'\"*_.,;:!?()[]{}<>"
MAX_CONFUSIONS = 10


def normalize(label: str) -> str:
    return " ".join(label.split()).casefold()


@dataclass
class ExampleResult:
    prompt: str
    gold: str | None
    output: str
    predicted: str | None
    outcome: str


@dataclass
class EvalReport:
    examples: list[ExampleResult]

    @property
    def scored(self) -> list[ExampleResult]:
        return [e for e in self.examples if e.outcome != GOLD_ERROR]

    @property
    def accuracy(self) -> float:
        scored = self.scored
        return sum(e.outcome == CORRECT for e in scored) / len(scored) if scored else 0.0

    def outcomes(self) -> dict[str, int]:
        counts = Counter(e.outcome for e in self.examples)
        return {o: counts.get(o, 0) for o in OUTCOMES}

    def per_label(self) -> dict[str, dict[str, float | int]]:
        """Precision, recall, F1 and support for every gold label."""
        scored = self.scored  # every scored example has a gold label
        gold = Counter(str(e.gold) for e in scored)
        predicted = Counter(e.predicted for e in scored if e.predicted is not None)
        right = Counter(str(e.gold) for e in scored if e.outcome == CORRECT)
        stats: dict[str, dict[str, float | int]] = {}
        for label in sorted(gold):
            precision = right[label] / predicted[label] if predicted[label] else 0.0
            recall = right[label] / gold[label]
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            stats[label] = {"precision": precision, "recall": recall, "f1": f1, "support": gold[label]}
        return stats

    @property
    def macro_f1(self) -> float:
        stats = self.per_label()
        return sum(float(s["f1"]) for s in stats.values()) / len(stats) if stats else 0.0

    def confusions(self) -> list[dict]:
        """The most common (gold → predicted) mistakes."""
        pairs = Counter(
            (e.gold, e.predicted or "(no answer)") for e in self.scored if e.outcome != CORRECT
        ).most_common(MAX_CONFUSIONS)
        return [{"gold": g, "predicted": p, "count": n} for (g, p), n in pairs]

    def to_dict(self) -> dict:
        return {
            "metric": "label_accuracy",
            "accuracy": self.accuracy,
            "secondary": {"label": "Macro-F1", "value": self.macro_f1},
            "per_label": self.per_label(),
            "confusions": self.confusions(),
            "scored": len(self.scored),
            "outcomes": self.outcomes(),
            "examples": [asdict(e) for e in self.examples],
        }


def _clean(text: str) -> str:
    text = _PREFIX_RE.sub("", text.strip(_STRIP_CHARS))
    return text.strip(_STRIP_CHARS)


def extract_label(output: str, labels: set[str]) -> str | None:
    """The label the output answers with (normalized), or its first line if it names none.

    `labels` are the task's normalized labels. Returns None for an empty answer.
    """
    text = strip_think(output).replace("```", "\n")
    lines = [line for line in (_clean(raw) for raw in text.splitlines()) if line]
    if not lines:
        return None
    first = normalize(lines[0])
    if first in labels:
        return first
    # A sentence around the label ("This is a billing_dispute ticket."): accept it only
    # when exactly one known label appears, so hedging between labels isn't rewarded.
    lowered = normalize(text)
    found = {label for label in labels if re.search(rf"(?<![\w-]){re.escape(label)}(?![\w-])", lowered)}
    if len(found) == 1:
        return found.pop()
    return first


def score_example(prompt: str, gold_text: object, output: str, labels: set[str]) -> ExampleResult:
    if not isinstance(gold_text, str) or not gold_text.strip():
        return ExampleResult(prompt, None, output, None, GOLD_ERROR)
    gold = normalize(gold_text)
    predicted = extract_label(output, labels)
    if predicted is None:
        outcome = NO_ANSWER
    elif predicted == gold:
        outcome = CORRECT
    elif predicted in labels:
        outcome = WRONG_LABEL
    else:
        outcome = UNKNOWN_LABEL
    return ExampleResult(prompt, gold, output, predicted, outcome)


def evaluate(rows: list[dict], outputs: list[str], labels: set[str] | None = None) -> EvalReport:
    """Score outputs against rows whose completion is a gold label.

    `labels` is the task's label set (e.g. every label in the training data); the gold
    labels of `rows` are always included.
    """
    if len(rows) != len(outputs):
        raise ValueError(f"{len(rows)} rows but {len(outputs)} outputs")
    known = {normalize(label) for label in labels or ()}
    known |= {normalize(c) for r in rows if isinstance(c := r.get("completion"), str) and c.strip()}
    return EvalReport(
        [score_example(r["prompt"], r.get("completion"), o, known) for r, o in zip(rows, outputs)]
    )
