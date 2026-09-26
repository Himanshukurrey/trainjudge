"""`trainjudge verify`: gather evals, decide the verdict, write the reports."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from trainjudge import evaluation, reports, runs, verdict
from trainjudge.status import StatusTracker


@dataclass
class VerifyResult:
    verdict: verdict.Verdict
    artifacts: list[Path]


def _usable(result: dict | None) -> bool:
    return result is not None and not result.get("test_split", {}).get("limited", False)


def verify_run(
    run_dir: Path,
    db_path: Path | None = None,
    min_improvement: float = verdict.DEFAULT_MIN_IMPROVEMENT,
    regression_tolerance: float = verdict.DEFAULT_REGRESSION_TOLERANCE,
    rerun: bool = False,
    log: Callable[[str], None] = lambda msg: None,
    generate: Callable[..., list[str]] | None = None,
    tracker: StatusTracker | None = None,
    task: str | None = None,
    scorer: str | None = None,
) -> VerifyResult:
    run = runs.read_run_json(run_dir)
    if run.get("status") != "trained":
        raise runs.RunError(f"{run_dir} hasn't finished training (status: {run.get('status')})")
    spec = evaluation.resolve_task(run_dir, task, db_path, scorer)

    def progress(done: int, total: int) -> None:
        log(f"  generated {done:,}/{total:,}")
        if tracker:
            tracker.progress(done, total)

    def stage(name: str, message: str) -> None:
        log(message)
        if tracker:
            tracker.stage(name)

    task_evals: dict[str, dict] = {}
    regression_evals: dict[str, dict] = {}
    for target in evaluation.TARGETS:
        label = "baseline" if target == evaluation.BASELINE else "fine-tuned"
        cached = None if rerun else evaluation.load_eval(run_dir, target)
        if cached is not None and _usable(cached) and evaluation.matches_task(cached, spec):
            log(f"Using saved {label} task eval ({cached['accuracy']:.1%}).")
            task_evals[target] = cached
        else:
            stage(f"eval:{target}", f"Evaluating {label} model on the held-out test split...")
            task_evals[target] = evaluation.evaluate_target(
                run_dir,
                target,
                on_progress=progress,
                generate=generate,
                task=spec,
                use_cache=not rerun,
            )
            if task_evals[target].get("cached_from"):
                log(
                    f"  reused the baseline from {task_evals[target]['cached_from']} "
                    "(same model and test split)"
                )
        cached = None if rerun else evaluation.load_eval(run_dir, target, regression=True)
        if cached is not None:
            log(f"Using saved {label} regression check.")
            regression_evals[target] = cached
        else:
            stage(f"regression:{target}", f"Running the regression check on the {label} model...")
            regression_evals[target] = evaluation.evaluate_regression(
                run_dir, target, on_progress=progress, generate=generate, use_cache=not rerun
            )
            if regression_evals[target].get("cached_from"):
                log(f"  reused the baseline regression check from {regression_evals[target]['cached_from']}")

    if tracker:
        tracker.stage("verdict")
    training = run.get("training") or {}
    v = verdict.decide(
        task_evals[evaluation.BASELINE],
        task_evals[evaluation.FINETUNED],
        regression_evals[evaluation.BASELINE],
        regression_evals[evaluation.FINETUNED],
        train_loss_drop_pct=training.get("train_loss_drop_pct"),
        min_improvement=min_improvement,
        regression_tolerance=regression_tolerance,
    )

    run = runs.read_run_json(run_dir)  # evals may have updated it
    results = {
        **v.to_dict(),
        "run": str(run_dir),
        "model": run["model"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evals": {t: str(evaluation.eval_path(run_dir, t)) for t in evaluation.TARGETS}
        | {
            f"regression_{t}": str(evaluation.eval_path(run_dir, t, regression=True))
            for t in evaluation.TARGETS
        },
    }
    artifacts = [
        run_dir / "MODEL_CARD.md",
        run_dir / "EXPERIMENT_REPORT.md",
        run_dir / "eval_results.json",
    ]
    artifacts[0].write_text(reports.model_card(run, v, str(run_dir), spec.name), encoding="utf-8")
    artifacts[1].write_text(
        reports.experiment_report(
            run,
            v,
            task_evals[evaluation.BASELINE],
            task_evals[evaluation.FINETUNED],
            regression_evals[evaluation.BASELINE],
            regression_evals[evaluation.FINETUNED],
            str(run_dir),
        ),
        encoding="utf-8",
    )
    artifacts[2].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    run["verdict"] = {
        "outcome": v.outcome,
        "reasons": v.reasons,
        "created_at": results["created_at"],
    }
    runs.write_run_json(run_dir, run)
    return VerifyResult(v, artifacts)
