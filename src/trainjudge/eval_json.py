"""JSON extraction accuracy: does the model's JSON object match the gold object?

The JSON object is pulled out of the model's raw output (think blocks, code
fences and surrounding prose are tolerated, so a chatty base model isn't
penalized for formatting alone) and compared with the gold completion field
by field:

- text is compared after trimming, collapsing whitespace and ignoring case
- numbers are compared by value (3 == 3.0 == "3"), booleans and null exactly
- extra fields in the prediction are reported but don't make it wrong

Accuracy is exact match: every gold field must be right. Field-level accuracy
(the share of gold fields right, averaged over examples) is reported alongside
it, to show how close the misses were.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

CORRECT = "correct"
WRONG_FIELDS = "wrong_fields"
MISSING_FIELDS = "missing_fields"
NOT_AN_OBJECT = "not_an_object"
NO_JSON = "no_json"
GOLD_ERROR = "gold_error"
OUTCOMES = (CORRECT, WRONG_FIELDS, MISSING_FIELDS, NOT_AN_OBJECT, NO_JSON, GOLD_ERROR)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_DECODER = json.JSONDecoder()


@dataclass
class ExampleResult:
    prompt: str
    gold: dict | None
    output: str
    predicted: object
    outcome: str
    wrong_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    extra_fields: list[str] = field(default_factory=list)
    field_score: float = 0.0
    error: str | None = None


@dataclass
class EvalReport:
    examples: list[ExampleResult]

    @property
    def scored(self) -> list[ExampleResult]:
        """Examples whose gold completion is a JSON object; only these count."""
        return [e for e in self.examples if e.outcome != GOLD_ERROR]

    @property
    def accuracy(self) -> float:
        scored = self.scored
        return sum(e.outcome == CORRECT for e in scored) / len(scored) if scored else 0.0

    @property
    def field_accuracy(self) -> float:
        scored = self.scored
        return sum(e.field_score for e in scored) / len(scored) if scored else 0.0

    def outcomes(self) -> dict[str, int]:
        counts = Counter(e.outcome for e in self.examples)
        return {o: counts.get(o, 0) for o in OUTCOMES}

    def per_field(self) -> dict[str, float]:
        """Share of examples that got each gold field right."""
        right: dict[str, int] = defaultdict(int)
        seen: dict[str, int] = defaultdict(int)
        for e in self.scored:
            for key in e.gold:
                seen[key] += 1
                right[key] += key not in e.wrong_fields and key not in e.missing_fields
        return {k: right[k] / seen[k] for k in seen}

    def to_dict(self) -> dict:
        return {
            "metric": "json_exact_match",
            "accuracy": self.accuracy,
            "secondary": {"label": "Field-level accuracy", "value": self.field_accuracy},
            "per_field": self.per_field(),
            "scored": len(self.scored),
            "outcomes": self.outcomes(),
            "examples": [asdict(e) for e in self.examples],
        }


def extract_json(output: str) -> object | None:
    """The first JSON value in the output that decodes, preferring objects."""
    text = _THINK_RE.sub("", output).strip()
    candidates = [block.strip() for block in _FENCE_RE.findall(text)] + [text]
    first_value = None
    for candidate in candidates:
        for i, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                value, _ = _DECODER.raw_decode(candidate, i)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
            if first_value is None:
                first_value = value
    return first_value


def normalize_value(value: object) -> object:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = " ".join(value.split()).casefold()
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    return value


def score_example(prompt: str, gold_text: str, output: str) -> ExampleResult:
    try:
        gold = json.loads(gold_text)
    except (TypeError, ValueError) as e:
        return ExampleResult(prompt, None, output, None, GOLD_ERROR, error=str(e))
    if not isinstance(gold, dict) or not gold:
        return ExampleResult(prompt, None, output, None, GOLD_ERROR, error="gold is not a JSON object")

    predicted = extract_json(output)
    # With no object to compare, every gold field counts as missing.
    if predicted is None:
        return ExampleResult(prompt, gold, output, None, NO_JSON, missing_fields=list(gold))
    if not isinstance(predicted, dict):
        return ExampleResult(prompt, gold, output, predicted, NOT_AN_OBJECT, missing_fields=list(gold))

    missing = [k for k in gold if k not in predicted]
    wrong = [k for k in gold if k in predicted and normalize_value(predicted[k]) != normalize_value(gold[k])]
    extra = [k for k in predicted if k not in gold]
    score = (len(gold) - len(missing) - len(wrong)) / len(gold)
    outcome = CORRECT if not missing and not wrong else MISSING_FIELDS if missing else WRONG_FIELDS
    return ExampleResult(prompt, gold, output, predicted, outcome, wrong, missing, extra, score)


def evaluate(rows: list[dict], outputs: list[str]) -> EvalReport:
    """Score model outputs against prompt/completion rows whose completion is a gold JSON object."""
    if len(rows) != len(outputs):
        raise ValueError(f"{len(rows)} rows but {len(outputs)} outputs")
    return EvalReport([score_example(r["prompt"], r["completion"], o) for r, o in zip(rows, outputs)])
