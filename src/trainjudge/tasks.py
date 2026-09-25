"""Task types `eval` and `verify` can score, and how to tell them apart."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SQL = "sql"
JSON = "json"

_SQL_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


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
}
METRIC_LABELS = {t.metric: t.label for t in TASKS.values()}


class UnknownTask(ValueError):
    pass


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False


def detect(rows: list[dict]) -> str:
    """Pick the task type from the gold completions of a test split."""
    completions = [r.get("completion") for r in rows if isinstance(r.get("completion"), str)]
    if not completions:
        raise UnknownTask("the test split has no prompt/completion rows to score")
    json_share = sum(_is_json_object(c) for c in completions) / len(completions)
    sql_share = sum(bool(_SQL_START_RE.match(c)) for c in completions) / len(completions)
    if json_share >= 0.8:
        return JSON
    if sql_share >= 0.8:
        return SQL
    raise UnknownTask(
        "no automatic eval for this task yet: `eval`/`verify` score SQL (execution accuracy) and "
        "JSON objects (exact match). Pass --task to force one."
    )
