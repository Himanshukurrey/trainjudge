"""Evaluate a run's base model and fine-tuned adapter on its held-out test split.

Results go to <run>/eval/<target>.json (task metric) and
<run>/eval/regression_<target>.json (general-capability suite): every prompt,
raw output and outcome, so a verdict can always be traced back to examples.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from trainjudge import eval_sql, mlx_backend, regression_check, runs

BASELINE = "baseline"
FINETUNED = "finetuned"
TARGETS = (BASELINE, FINETUNED)


def test_rows(run_dir: Path, limit: int | None = None) -> list[dict]:
    path = run_dir / "data" / "test.jsonl"
    if not path.exists():
        raise runs.RunError(f"{path} not found; is {run_dir} a trainjudge run directory?")
    rows = [json.loads(line) for line in path.open()]
    if rows and "completion" not in rows[0]:
        raise runs.RunError("SQL eval needs prompt/completion rows with gold SQL completions")
    return rows[:limit] if limit else rows


def evaluate_target(
    run_dir: Path,
    target: str,
    db_path: Path,
    decoding: mlx_backend.DecodingConfig | None = None,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    generate: Callable[..., list[str]] = mlx_backend.generate_outputs,
) -> dict:
    """Generate and score one target; write and return its eval record."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    record = runs.read_run_json(run_dir)
    adapter = run_dir / "adapters" if target == FINETUNED else None
    if adapter is not None and not (adapter / "adapters.safetensors").exists():
        raise runs.RunError(f"no trained adapter in {adapter}; run `trainjudge train` first")

    rows = test_rows(run_dir, limit)
    db = eval_sql.Database(db_path)
    decoding = decoding or mlx_backend.DecodingConfig()

    start = time.monotonic()
    outputs = generate(record["model"], [r["prompt"] for r in rows], adapter_path=adapter,
                       decoding=decoding, on_progress=on_progress)
    report = eval_sql.evaluate(db, rows, outputs)
    result = {
        "target": target,
        "model": record["model"],
        "adapter_path": str(adapter) if adapter else None,
        "task": "sql",
        "database": {"path": str(db_path), "sha256": runs.file_sha256(db_path)},
        "test_split": {"path": str(run_dir / "data" / "test.jsonl"), "rows": len(rows),
                       "limited": limit is not None},
        "decoding": asdict(decoding),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - start, 1),
        **report.to_dict(),
    }

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / f"{target}.json").write_text(json.dumps(result, indent=2) + "\n")

    record["task"] = {"type": "sql", "database": str(db_path)}
    record.setdefault("evals", {})[target] = {
        "accuracy": result["accuracy"],
        "scored": result["scored"],
        "path": str(eval_dir / f"{target}.json"),
    }
    runs.write_run_json(run_dir, record)
    return result


def eval_path(run_dir: Path, target: str, regression: bool = False) -> Path:
    return run_dir / "eval" / (f"regression_{target}.json" if regression else f"{target}.json")


def load_eval(run_dir: Path, target: str, regression: bool = False) -> dict | None:
    path = eval_path(run_dir, target, regression)
    return json.loads(path.read_text()) if path.exists() else None


def evaluate_regression(
    run_dir: Path,
    target: str,
    decoding: mlx_backend.DecodingConfig | None = None,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    generate: Callable[..., list[str]] = mlx_backend.generate_outputs,
) -> dict:
    """Run the general-capability suite for one target; write and return its record."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    record = runs.read_run_json(run_dir)
    adapter = run_dir / "adapters" if target == FINETUNED else None
    if adapter is not None and not (adapter / "adapters.safetensors").exists():
        raise runs.RunError(f"no trained adapter in {adapter}; run `trainjudge train` first")

    items = regression_check.build_suite()
    decoding = decoding or mlx_backend.DecodingConfig(max_tokens=256)
    start = time.monotonic()
    outputs = generate(record["model"], [i.prompt for i in items], adapter_path=adapter,
                       decoding=decoding, on_progress=on_progress)
    result = {
        "target": target,
        "model": record["model"],
        "adapter_path": str(adapter) if adapter else None,
        "suite": "trainjudge-regression-v1",
        "decoding": asdict(decoding),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - start, 1),
        **regression_check.to_dict(regression_check.score(items, outputs)),
    }
    path = eval_path(run_dir, target, regression=True)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result
