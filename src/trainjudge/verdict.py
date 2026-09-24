"""Verdict: compare baseline and fine-tuned on the task metric and the regression suite.

- IMPROVED:  the task metric rose by at least --min-improvement points, the gain
             is statistically significant (paired McNemar test on the same test
             examples), and no regression category dropped beyond tolerance
- REGRESSED: the task improved, but general capability got worse; don't deploy
             as-is
- REJECTED:  the task metric didn't improve meaningfully, whatever the loss did
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

from trainjudge.regression_check import CATEGORIES, CATEGORY_LABELS

IMPROVED = "IMPROVED"
REGRESSED = "REGRESSED"
REJECTED = "REJECTED"

DEFAULT_MIN_IMPROVEMENT = 3.0  # accuracy points
DEFAULT_REGRESSION_TOLERANCE = 5.0  # pass-rate points
SIGNIFICANCE = 0.05


@dataclass
class RegressionResult:
    category: str
    baseline: float
    finetuned: float
    total: int
    tolerance: float

    @property
    def delta_points(self) -> float:
        return (self.finetuned - self.baseline) * 100

    @property
    def regressed(self) -> bool:
        return self.delta_points < -self.tolerance


@dataclass
class Verdict:
    outcome: str
    metric: str
    baseline: float
    finetuned: float
    scored: int
    gained: int  # wrong at baseline, right after fine-tuning
    lost: int  # right at baseline, wrong after fine-tuning
    p_value: float
    baseline_lenient: float | None
    finetuned_lenient: float | None
    regressions: list[RegressionResult]
    train_loss_drop_pct: float | None
    min_improvement: float
    reasons: list[str] = field(default_factory=list)

    @property
    def improvement_points(self) -> float:
        return (self.finetuned - self.baseline) * 100

    @property
    def significant(self) -> bool:
        return self.p_value < SIGNIFICANCE

    def to_dict(self) -> dict:
        return {
            "verdict": self.outcome,
            "reasons": self.reasons,
            "task": {
                "metric": self.metric,
                "baseline": self.baseline,
                "finetuned": self.finetuned,
                "improvement_points": self.improvement_points,
                "baseline_lenient": self.baseline_lenient,
                "finetuned_lenient": self.finetuned_lenient,
                "scored": self.scored,
                "gained": self.gained,
                "lost": self.lost,
                "p_value": self.p_value,
                "significant": self.significant,
            },
            "regression": {
                r.category: {
                    "baseline": r.baseline,
                    "finetuned": r.finetuned,
                    "delta_points": r.delta_points,
                    "total": r.total,
                    "regressed": r.regressed,
                }
                for r in self.regressions
            },
            "training": {"train_loss_drop_pct": self.train_loss_drop_pct},
            "thresholds": {
                "min_improvement_points": self.min_improvement,
                "regression_tolerance_points": (
                    self.regressions[0].tolerance if self.regressions else None
                ),
                "significance": SIGNIFICANCE,
            },
        }


def mcnemar_p(gained: int, lost: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def decide(
    baseline_eval: dict,
    finetuned_eval: dict,
    baseline_regression: dict,
    finetuned_regression: dict,
    train_loss_drop_pct: float | None = None,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    regression_tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> Verdict:
    base_examples = {e["prompt"]: e for e in baseline_eval["examples"]}
    tuned_examples = {e["prompt"]: e for e in finetuned_eval["examples"]}
    if set(base_examples) != set(tuned_examples):
        raise ValueError("baseline and fine-tuned evals cover different test examples")

    gained = lost = 0
    for prompt, base in base_examples.items():
        tuned = tuned_examples[prompt]
        if "gold_error" in (base["outcome"], tuned["outcome"]):
            continue
        b, t = base["outcome"] == "correct", tuned["outcome"] == "correct"
        gained += t and not b
        lost += b and not t

    regressions = [
        RegressionResult(
            category=c,
            baseline=baseline_regression["summary"][c]["rate"],
            finetuned=finetuned_regression["summary"][c]["rate"],
            total=finetuned_regression["summary"][c]["total"],
            tolerance=regression_tolerance,
        )
        for c in CATEGORIES
    ]
    verdict = Verdict(
        outcome=REJECTED,
        metric=baseline_eval.get("metric", "accuracy"),
        baseline=baseline_eval["accuracy"],
        finetuned=finetuned_eval["accuracy"],
        scored=finetuned_eval["scored"],
        gained=gained,
        lost=lost,
        p_value=mcnemar_p(gained, lost),
        baseline_lenient=baseline_eval.get("lenient_accuracy"),
        finetuned_lenient=finetuned_eval.get("lenient_accuracy"),
        regressions=regressions,
        train_loss_drop_pct=train_loss_drop_pct,
        min_improvement=min_improvement,
    )

    improved = verdict.improvement_points >= min_improvement and verdict.significant
    regressed = [r for r in regressions if r.regressed]
    loss = (f"training loss dropped {train_loss_drop_pct:.0f}%, but "
            if train_loss_drop_pct and train_loss_drop_pct > 0 else "")
    if not improved:
        verdict.outcome = REJECTED
        if verdict.improvement_points < min_improvement:
            verdict.reasons.append(
                f"{loss}task accuracy changed {verdict.improvement_points:+.1f} points "
                f"(below the {min_improvement:g}-point bar)"
            )
        else:
            verdict.reasons.append(
                f"{loss}the {verdict.improvement_points:+.1f}-point gain isn't statistically "
                f"significant ({format_p(verdict.p_value)}; {gained} gained, {lost} lost)"
            )
    elif regressed:
        verdict.outcome = REGRESSED
        verdict.reasons.append(
            f"task accuracy improved {verdict.improvement_points:+.1f} points"
        )
    else:
        verdict.outcome = IMPROVED
        verdict.reasons.append(
            f"task accuracy improved {verdict.improvement_points:+.1f} points on held-out "
            f"examples ({format_p(verdict.p_value)}), with no regressions"
        )
    for r in regressed:
        verdict.reasons.append(
            f"{CATEGORY_LABELS[r.category].lower()} regressed {r.delta_points:+.1f} points"
        )
    return verdict


def format_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


METRIC_LABELS = {"sql_execution_accuracy": "SQL execution accuracy"}


def format_box(v: Verdict) -> str:
    rows: list[str | None] = ["TRAINJUDGE VERDICT", None]
    rows.append(f"Metric: {METRIC_LABELS.get(v.metric, v.metric.replace('_', ' '))}")
    rows.append(f"Baseline        {v.baseline:.1%}")
    rows.append(f"Fine-tuned      {v.finetuned:.1%}")
    rows.append(f"Improvement     {v.improvement_points:+.1f} pts")
    rows.append(f"Held-out n={v.scored} · {format_p(v.p_value)}")
    rows.append(None)
    rows.append("Regression check (held-out tasks):")
    for r in v.regressions:
        mark = "⚠" if r.regressed else "✓"
        rates = f"{r.baseline:.0%} → {r.finetuned:.0%}"
        rows.append(f"  {CATEGORY_LABELS[r.category] + ':':<31} {mark} {rates}")
    rows.append(None)

    width = max(38, *(len(r) for r in rows if r)) + 2
    rows += _wrap_verdict(v, width - 2)
    out = ["╔" + "═" * width + "╗"]
    for i, row in enumerate(rows):
        if row is None:
            out.append("╠" + "═" * width + "╣")
        elif i == 0:
            out.append("║" + row.center(width) + "║")
        else:
            out.append("║ " + row.ljust(width - 1) + "║")
    out.append("╚" + "═" * width + "╝")
    return "\n".join(out)


def _wrap_verdict(v: Verdict, width: int) -> list[str]:
    head = {IMPROVED: "✓ IMPROVED", REGRESSED: "⚠ REGRESSED", REJECTED: "✗ REJECTED"}[v.outcome]
    tail = {
        IMPROVED: "Verified on held-out data, not just lower training loss.",
        REGRESSED: "Don't deploy as-is.",
        REJECTED: "Do not deploy this.",
    }[v.outcome]
    text = f"{head}: {'; '.join(v.reasons)}. {tail}"
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = "  " + word
        else:
            current = f"{current} {word}" if current else word
    lines.append(current)
    return lines
