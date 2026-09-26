"""Task types `eval` and `verify` can score, and how to tell them apart."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SQL = "sql"
JSON = "json"
LABEL = "label"
CUSTOM = "custom"

_SQL_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# A classification task: every gold answer is a short label from a set of at most this many.
MAX_LABELS = 50
MAX_LABEL_CHARS = 60


@dataclass(frozen=True)
class TaskInfo:
    name: str
    metric: str
    label: str  # the main metric, e.g. "SQL execution accuracy"
    short_label: str  # the table row label, e.g. "Execution accuracy"
    secondary_label: str
    needs_db: bool
    card_task_type: str  # Hugging Face model-card task type
    card_metric_type: str
    caveat: str


TASKS = {
    SQL: TaskInfo(
        name=SQL,
        metric="sql_execution_accuracy",
        label="SQL execution accuracy",
        short_label="Execution accuracy",
        secondary_label="Lenient (extra columns allowed)",
        needs_db=True,
        card_task_type="text-to-sql",
        card_metric_type="execution_accuracy",
        caveat=(
            "Execution accuracy checks result rows on one database. A query can match by "
            "coincidence on this data and still be wrong in general."
        ),
    ),
    JSON: TaskInfo(
        name=JSON,
        metric="json_exact_match",
        label="JSON exact-match accuracy",
        short_label="Exact match (every field right)",
        secondary_label="Field-level accuracy",
        needs_db=False,
        card_task_type="structured-extraction",
        card_metric_type="exact_match",
        caveat=(
            "Exact match compares normalized field values (case, whitespace and number "
            "formatting are ignored). A value that means the same thing but is worded "
            "differently still counts as wrong."
        ),
    ),
    LABEL: TaskInfo(
        name=LABEL,
        metric="label_accuracy",
        label="Label accuracy",
        short_label="Accuracy (right label)",
        secondary_label="Macro-F1",
        needs_db=False,
        card_task_type="text-classification",
        card_metric_type="accuracy",
        caveat=(
            "Label accuracy compares the predicted label with the gold one after normalizing case "
            "and whitespace. Labels are only as good as the gold data: a mislabelled test row "
            "counts against a model that got it right."
        ),
    ),
    CUSTOM: TaskInfo(
        name=CUSTOM,
        metric="custom_accuracy",
        label="Custom scorer accuracy",
        short_label="Accuracy (scorer says correct)",
        secondary_label="Mean score",
        needs_db=False,
        card_task_type="text-generation",
        card_metric_type="accuracy",
        caveat=(
            "Scored by a user-supplied function; the verdict is only as meaningful as that "
            "function. Check its code before trusting the result."
        ),
    ),
}
METRIC_LABELS = {t.metric: t.label for t in TASKS.values()}


class UnknownTask(ValueError):
    pass


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False


def _looks_like_label(text: str) -> bool:
    text = text.strip()
    return bool(text) and "\n" not in text and len(text) <= MAX_LABEL_CHARS and len(text.split()) <= 4


def looks_like_labels(completions: list[str]) -> bool:
    """Short single-line answers drawn from a small, repeating set: a classification task."""
    if not completions or sum(map(_looks_like_label, completions)) / len(completions) < 0.9:
        return False
    distinct = {" ".join(c.split()).casefold() for c in completions}
    return 2 <= len(distinct) <= min(MAX_LABELS, len(completions) // 2)


def detect(rows: list[dict]) -> str:
    """Pick the task type from the gold completions of a test split."""
    completions = [c for r in rows if isinstance(c := r.get("completion"), str)]
    if not completions:
        raise UnknownTask("the test split has no prompt/completion rows to score")
    json_share = sum(_is_json_object(c) for c in completions) / len(completions)
    sql_share = sum(bool(_SQL_START_RE.match(c)) for c in completions) / len(completions)
    if json_share >= 0.8:
        return JSON
    if sql_share >= 0.8:
        return SQL
    if looks_like_labels(completions):
        return LABEL
    raise UnknownTask(
        "no automatic eval for this task: `eval`/`verify` detect SQL (execution accuracy), JSON "
        "objects (exact match) and labels (classification accuracy). For anything else, write a "
        "scoring function and pass --scorer path/to/scorer.py:score."
    )
