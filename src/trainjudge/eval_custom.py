"""Custom scorers: score any task with a Python function you write.

A scorer is a function in a .py file (or an importable module):

    def score(prompt: str, output: str, gold: str) -> bool | dict:
        ...

`output` is the model's answer with any think block removed; `gold` is the
row's completion. Add a fourth parameter named `row` to also get the whole
test row (for extra fields such as metadata or reference answers). Return:

- True / False: whether the answer is right, or
- a dict: {"correct": bool, "score": float in [0, 1] (optional, partial
  credit), "reason": str (optional, shown in reports)}

Accuracy is the share of answers marked correct; the mean score (1/0 for
plain booleans) is reported alongside it. Pass it as `--scorer path.py:func`
(`func` defaults to `score`) or `--scorer package.module:func`.

The scorer runs in-process with your permissions, like any script you run.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from trainjudge.textutil import strip_think

CORRECT = "correct"
INCORRECT = "incorrect"
SCORER_ERROR = "scorer_error"
GOLD_ERROR = "gold_error"
OUTCOMES = (CORRECT, INCORRECT, SCORER_ERROR, GOLD_ERROR)
DEFAULT_FUNCTION = "score"


class ScorerError(ValueError):
    pass


@dataclass(frozen=True)
class Scorer:
    spec: str  # as recorded: "/abs/path/scorer.py:score" or "package.module:score"
    source: Path | None  # the file the function lives in, if known
    sha256: str | None  # of that file, so changing the scorer invalidates saved evals
    function: Callable

    @property
    def wants_row(self) -> bool:
        try:
            return "row" in inspect.signature(self.function).parameters
        except (TypeError, ValueError):
            return False

    def describe(self) -> dict:
        return {"spec": self.spec, "sha256": self.sha256}


def _split(spec: str) -> tuple[str, str]:
    # "C:\\x\\s.py:score" has a drive colon, so split on the last colon only when what
    # follows looks like a function name.
    head, sep, tail = spec.rpartition(":")
    if sep and tail.isidentifier() and head:
        return head, tail
    return spec, DEFAULT_FUNCTION


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(spec: str, base: Path | None = None) -> Scorer:
    """Load `path.py[:func]` or `module[:func]`; relative paths resolve against `base`."""
    target, name = _split(spec.strip())
    if target.endswith(".py"):
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = (base or Path.cwd()) / path
        path = path.resolve()
        if not path.is_file():
            raise ScorerError(f"scorer file {path} not found")
        module_name = f"trainjudge_scorer_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ScorerError(f"can't load {path} as Python")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception as e:
            raise ScorerError(f"{path} failed to import: {type(e).__name__}: {e}") from e
        source: Path | None = path
        recorded = f"{path}:{name}"
    else:
        try:
            module = importlib.import_module(target)
        except Exception as e:
            raise ScorerError(f"can't import scorer module {target!r}: {type(e).__name__}: {e}") from e
        file = getattr(module, "__file__", None)
        source = Path(file) if file else None
        recorded = f"{target}:{name}"
    function = getattr(module, name, None)
    if not callable(function):
        raise ScorerError(f"{recorded} is not a function (define `def {name}(prompt, output, gold)`)")
    return Scorer(recorded, source, _sha256(source) if source else None, function)


@dataclass
class ExampleResult:
    prompt: str
    gold: str | None
    output: str
    outcome: str
    score: float = 0.0
    reason: str | None = None


def _interpret(value: object) -> tuple[bool, float, str | None]:
    if isinstance(value, bool):
        return value, float(value), None
    if isinstance(value, dict) and isinstance(value.get("correct"), bool):
        correct = value["correct"]
        raw = value.get("score", float(correct))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 0 <= raw <= 1:
            raise ScorerError(f"'score' must be a number from 0 to 1, got {raw!r}")
        reason = value.get("reason")
        return correct, float(raw), str(reason) if reason is not None else None
    raise ScorerError(
        f"a scorer must return True/False or a dict with a boolean 'correct', got {type(value).__name__}"
    )


@dataclass
class EvalReport:
    examples: list[ExampleResult]
    scorer: Scorer

    @property
    def scored(self) -> list[ExampleResult]:
        return [e for e in self.examples if e.outcome != GOLD_ERROR]

    @property
    def accuracy(self) -> float:
        scored = self.scored
        return sum(e.outcome == CORRECT for e in scored) / len(scored) if scored else 0.0

    @property
    def mean_score(self) -> float:
        scored = self.scored
        return sum(e.score for e in scored) / len(scored) if scored else 0.0

    def outcomes(self) -> dict[str, int]:
        counts = Counter(e.outcome for e in self.examples)
        return {o: counts.get(o, 0) for o in OUTCOMES}

    def to_dict(self) -> dict:
        return {
            "metric": "custom_accuracy",
            "scorer": self.scorer.describe(),
            "accuracy": self.accuracy,
            "secondary": {"label": "Mean score", "value": self.mean_score},
            "scored": len(self.scored),
            "outcomes": self.outcomes(),
            "examples": [asdict(e) for e in self.examples],
        }


def evaluate(scorer: Scorer, rows: list[dict], outputs: list[str]) -> EvalReport:
    """Score every output with the scorer.

    An exception (or a bad return value) marks that example `scorer_error`, which counts
    as wrong. If every example errors, the scorer is broken, so this raises instead of
    reporting 0%.
    """
    if len(rows) != len(outputs):
        raise ValueError(f"{len(rows)} rows but {len(outputs)} outputs")
    results = []
    first_error = None
    for row, output in zip(rows, outputs):
        prompt, gold = row["prompt"], row.get("completion")
        if not isinstance(gold, str):
            results.append(ExampleResult(prompt, None, output, GOLD_ERROR))
            continue
        answer = strip_think(output)
        try:
            args: dict = {"row": row} if scorer.wants_row else {}
            correct, score, reason = _interpret(scorer.function(prompt, answer, gold, **args))
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
            first_error = first_error or message
            results.append(ExampleResult(prompt, gold, output, SCORER_ERROR, reason=message))
            continue
        results.append(ExampleResult(prompt, gold, output, CORRECT if correct else INCORRECT, score, reason))
    report = EvalReport(results, scorer)
    if report.scored and all(e.outcome == SCORER_ERROR for e in report.scored):
        raise ScorerError(f"{scorer.spec} failed on every example; first error: {first_error}")
    return report
