"""SQL execution accuracy: does the generated query return the same rows as the gold query?

The generated SQL is pulled out of the model's raw output (think blocks, code
fences and surrounding prose are tolerated, so a chatty base model isn't
penalized for formatting alone), executed read-only against a SQLite
database, and its result compared with the gold query's result:

- rows are compared as a multiset, or in order when the gold query ends with
  ORDER BY (a top-level check that only looks at the gold query's tail)
- column names are ignored; column order and values must match
- floats are compared after rounding to 6 decimal places

Accuracy is strict: selecting extra columns is wrong, because returning exactly
what was asked is part of the task. A lenient score that allows extra columns
is reported alongside it, to show how much of a change is conventions versus
getting the underlying query right.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

from trainjudge.textutil import strip_think

CORRECT = "correct"
WRONG_RESULT = "wrong_result"
EXECUTION_ERROR = "execution_error"
NO_SQL = "no_sql"
TIMEOUT = "timeout"
GOLD_ERROR = "gold_error"
OUTCOMES = (CORRECT, WRONG_RESULT, EXECUTION_ERROR, NO_SQL, TIMEOUT, GOLD_ERROR)

QUERY_TIMEOUT_S = 5.0
MAX_ROWS = 10_000

_FENCE_RE = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
# SQL as conventionally written (uppercase keywords) is preferred over a
# lowercase "select" or "with" that is more likely part of a sentence.
_SQL_START_RES = [
    re.compile(r"\bSELECT\b|\bWITH\s+(?:RECURSIVE\s+)?\w+\s+AS\s*\("),
    re.compile(r"\bselect\b|\bwith\s+(?:recursive\s+)?\w+\s+as\s*\(", re.IGNORECASE),
]
MAX_LENIENT_COLUMNS = 8
_TRAILING_ORDER_BY_RE = re.compile(r"\border\s+by\b[^()]*$", re.IGNORECASE)


class QueryTimeout(Exception):
    pass


@dataclass
class ExampleResult:
    prompt: str
    gold_sql: str
    output: str
    sql: str | None
    outcome: str
    error: str | None = None
    lenient_correct: bool = False


@dataclass
class EvalReport:
    examples: list[ExampleResult]

    @property
    def scored(self) -> list[ExampleResult]:
        """Examples whose gold query ran; only these count toward accuracy."""
        return [e for e in self.examples if e.outcome != GOLD_ERROR]

    @property
    def accuracy(self) -> float:
        scored = self.scored
        return sum(e.outcome == CORRECT for e in scored) / len(scored) if scored else 0.0

    @property
    def lenient_accuracy(self) -> float:
        scored = self.scored
        return sum(e.lenient_correct for e in scored) / len(scored) if scored else 0.0

    def outcomes(self) -> dict[str, int]:
        counts = Counter(e.outcome for e in self.examples)
        return {o: counts.get(o, 0) for o in OUTCOMES}

    def to_dict(self) -> dict:
        return {
            "metric": "sql_execution_accuracy",
            "accuracy": self.accuracy,
            "lenient_accuracy": self.lenient_accuracy,
            "secondary": {"label": "Lenient (extra columns allowed)", "value": self.lenient_accuracy},
            "scored": len(self.scored),
            "outcomes": self.outcomes(),
            "examples": [asdict(e) for e in self.examples],
        }


def extract_sql(output: str) -> str | None:
    """Best-effort extraction of one SQL statement from raw model output."""
    text = strip_think(output)
    for block in _FENCE_RE.findall(text):
        if any(r.search(block) for r in _SQL_START_RES):
            text = block
            break
    m = next((m for r in _SQL_START_RES if (m := r.search(text))), None)
    if not m:
        return None
    sql = text[m.start() :]
    sql = re.split(r"\n\s*\n", sql, maxsplit=1)[0]  # stop at the first blank line
    if ";" in sql:
        sql = sql[: sql.index(";") + 1]
    sql = sql.strip().rstrip("`").strip()
    return sql or None


class Database:
    """A read-only, in-memory copy of a SQLite database."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(":memory:")
        if self.path.suffix == ".sql":
            self.conn.executescript(self.path.read_text(encoding="utf-8"))
        else:
            source = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            source.backup(self.conn)
            source.close()
        self.conn.execute("PRAGMA query_only = ON")

    def execute(self, sql: str, timeout_s: float = QUERY_TIMEOUT_S) -> list[tuple]:
        deadline = time.monotonic() + timeout_s
        timed_out = False

        def check() -> int:
            nonlocal timed_out
            timed_out = time.monotonic() > deadline
            return 1 if timed_out else 0

        self.conn.set_progress_handler(check, 10_000)
        try:
            cursor = self.conn.execute(sql)
            rows = cursor.fetchmany(MAX_ROWS + 1)
        except sqlite3.OperationalError:
            if timed_out:
                raise QueryTimeout(f"query took longer than {timeout_s:g}s") from None
            raise
        finally:
            self.conn.set_progress_handler(None, 0)
        if len(rows) > MAX_ROWS:
            raise sqlite3.OperationalError(f"query returned more than {MAX_ROWS:,} rows")
        return rows


def results_match(gold: list[tuple], predicted: list[tuple], ordered: bool) -> bool:
    gold_rows = [_normalize_row(r) for r in gold]
    predicted_rows = [_normalize_row(r) for r in predicted]
    if ordered:
        return gold_rows == predicted_rows
    return Counter(gold_rows) == Counter(predicted_rows)


def results_match_lenient(gold: list[tuple], predicted: list[tuple]) -> bool:
    """True if some choice of predicted columns, in some order, reproduces the gold result."""
    if len(gold) != len(predicted):
        return False
    if not gold:
        return True
    k, m = len(gold[0]), len(predicted[0])
    if m < k or m > MAX_LENIENT_COLUMNS:
        return False
    target = Counter(_normalize_row(r) for r in gold)
    return any(
        Counter(_normalize_row(tuple(row[i] for i in cols)) for row in predicted) == target
        for cols in permutations(range(m), k)
    )


def score_example(db: Database, prompt: str, gold_sql: str, output: str) -> ExampleResult:
    try:
        gold = db.execute(gold_sql)
    except (sqlite3.Error, QueryTimeout) as e:
        return ExampleResult(prompt, gold_sql, output, None, GOLD_ERROR, str(e))

    sql = extract_sql(output)
    if sql is None:
        return ExampleResult(prompt, gold_sql, output, None, NO_SQL)
    try:
        predicted = db.execute(sql)
    except QueryTimeout as e:
        return ExampleResult(prompt, gold_sql, output, sql, TIMEOUT, str(e))
    except (sqlite3.Error, sqlite3.Warning) as e:
        return ExampleResult(prompt, gold_sql, output, sql, EXECUTION_ERROR, str(e))

    ordered = bool(_TRAILING_ORDER_BY_RE.search(gold_sql.rstrip("; \n")))
    if results_match(gold, predicted, ordered):
        return ExampleResult(prompt, gold_sql, output, sql, CORRECT, lenient_correct=True)
    lenient = results_match_lenient(gold, predicted)
    return ExampleResult(prompt, gold_sql, output, sql, WRONG_RESULT, lenient_correct=lenient)


def evaluate(db: Database, rows: list[dict], outputs: list[str]) -> EvalReport:
    """Score model outputs against prompt/completion rows whose completion is gold SQL."""
    if len(rows) != len(outputs):
        raise ValueError(f"{len(rows)} rows but {len(outputs)} outputs")
    return EvalReport([score_example(db, r["prompt"], r["completion"], o) for r, o in zip(rows, outputs)])


def _normalize_row(row: tuple) -> tuple:
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)
