"""Evaluate a run's base model and fine-tuned adapter on its held-out test split.

Results go to <run>/eval/<target>.json (task metric) and
<run>/eval/regression_<target>.json (general-capability suite): every prompt,
raw output and outcome, so a verdict can always be traced back to examples.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from trainjudge import backends, eval_json, eval_sql, regression_check, runs, tasks
from trainjudge.backend_base import DecodingConfig

CACHE_DIR = ".baseline-cache"

BASELINE = "baseline"
FINETUNED = "finetuned"
TARGETS = (BASELINE, FINETUNED)


def test_rows(run_dir: Path, limit: int | None = None) -> list[dict]:
    path = run_dir / "data" / "test.jsonl"
    if not path.exists():
        raise runs.RunError(f"{path} not found; is {run_dir} a trainjudge run directory?")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if rows and "completion" not in rows[0]:
        raise runs.RunError("evals need prompt/completion rows with gold completions")
    return rows[:limit] if limit else rows


def _cache_file(run_dir: Path, kind: str, parts: dict) -> Path:
    """Where a baseline result is cached, keyed by everything that determines it.

    The base model's answers depend only on the model, the prompts, decoding and (for
    SQL) the database, so every run in the same runs folder that shares them can reuse
    one baseline instead of regenerating it.
    """
    key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]
    return run_dir.parent / CACHE_DIR / f"{kind}-{key}.json"


def _load_cached(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_cached(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def resolve_task(
    run_dir: Path, task: str | None = None, db_path: Path | None = None
) -> tuple[str, Path | None]:
    """The task to score (given, recorded by an earlier eval, or detected) and its database."""
    recorded = runs.read_run_json(run_dir).get("task") or {}
    if task in (None, "auto"):
        task = recorded.get("type")
    if task in (None, "auto"):
        try:
            task = tasks.detect(test_rows(run_dir))
        except tasks.UnknownTask as e:
            raise runs.RunError(str(e)) from e
    if task not in tasks.TASKS:
        raise runs.RunError(f"unknown task {task!r}; choose from {', '.join(tasks.TASKS)}")
    if db_path is None and recorded.get("database"):
        db_path = Path(recorded["database"])
    if tasks.TASKS[task].needs_db and (db_path is None or not db_path.is_file()):
        raise runs.RunError(
            "the SQL eval needs --db: the database (or .sql script) queries should run against"
        )
    return task, db_path if tasks.TASKS[task].needs_db else None


def evaluate_target(
    run_dir: Path,
    target: str,
    db_path: Path | None = None,
    decoding: DecodingConfig | None = None,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    generate: Callable[..., list[str]] | None = None,
    task: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Generate and score one target; write and return its eval record.

    Baselines are reused from other runs with the same model, test split, decoding and
    database unless `use_cache` is False.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    record = runs.read_run_json(run_dir)
    adapter = run_dir / "adapters" if target == FINETUNED else None
    if adapter is not None and not backends.adapter_ready(adapter):
        raise runs.RunError(f"no trained adapter in {adapter}; run `trainjudge train` first")
    generate = generate or backends.generator(record)

    task, db_path = resolve_task(run_dir, task, db_path)
    rows = test_rows(run_dir, limit)
    decoding = decoding or DecodingConfig()

    cache = None
    if target == BASELINE:
        cache = _cache_file(run_dir, "task", {
            "model": record["model"],
            "task": task,
            "test_split": runs.file_sha256(run_dir / "data" / "test.jsonl"),
            "limit": limit,
            "decoding": asdict(decoding),
            "database": runs.file_sha256(db_path) if db_path else None,
        })  # fmt: skip
        cached = _load_cached(cache) if use_cache else None
        if cached is not None:
            result = {
                **cached,
                "run": str(run_dir),
                "cached_from": cached.get("cached_from") or cached.get("run"),
            }
            result["test_split"] = {**cached["test_split"], "path": str(run_dir / "data" / "test.jsonl")}
            return _record_task_eval(run_dir, record, target, task, db_path, result)

    start = time.monotonic()
    outputs = generate(
        record["model"],
        [r["prompt"] for r in rows],
        adapter_path=adapter,
        decoding=decoding,
        on_progress=on_progress,
    )
    report: eval_sql.EvalReport | eval_json.EvalReport
    if task == tasks.SQL:
        assert db_path is not None  # resolve_task requires one for SQL
        report = eval_sql.evaluate(eval_sql.Database(db_path), rows, outputs)
    else:
        report = eval_json.evaluate(rows, outputs)
    result = {
        "target": target,
        "model": record["model"],
        "adapter_path": str(adapter) if adapter else None,
        "run": str(run_dir),
        "task": task,
        "database": {"path": str(db_path), "sha256": runs.file_sha256(db_path)} if db_path else None,
        "test_split": {
            "path": str(run_dir / "data" / "test.jsonl"),
            "rows": len(rows),
            "limited": limit is not None,
        },
        "decoding": asdict(decoding),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - start, 1),
        **report.to_dict(),
    }

    if cache is not None:
        _save_cached(cache, result)
    return _record_task_eval(run_dir, record, target, task, db_path, result)


def _record_task_eval(run_dir: Path, record: dict, target: str, task: str, db_path, result: dict) -> dict:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / f"{target}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    record["task"] = {"type": task, "database": str(db_path) if db_path else None}
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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def evaluate_regression(
    run_dir: Path,
    target: str,
    decoding: DecodingConfig | None = None,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    generate: Callable[..., list[str]] | None = None,
    use_cache: bool = True,
) -> dict:
    """Run the general-capability suite for one target; write and return its record."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    record = runs.read_run_json(run_dir)
    adapter = run_dir / "adapters" if target == FINETUNED else None
    if adapter is not None and not backends.adapter_ready(adapter):
        raise runs.RunError(f"no trained adapter in {adapter}; run `trainjudge train` first")
    generate = generate or backends.generator(record)

    items = regression_check.build_suite()
    decoding = decoding or DecodingConfig(max_tokens=256)
    path = eval_path(run_dir, target, regression=True)
    cache = None
    if target == BASELINE:
        cache = _cache_file(run_dir, "regression", {
            "model": record["model"],
            "prompts": [i.prompt for i in items],
            "decoding": asdict(decoding),
        })  # fmt: skip
        cached = _load_cached(cache) if use_cache else None
        if cached is not None:
            result = {
                **cached,
                "run": str(run_dir),
                "cached_from": cached.get("cached_from") or cached.get("run"),
            }
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
    start = time.monotonic()
    outputs = generate(
        record["model"],
        [i.prompt for i in items],
        adapter_path=adapter,
        decoding=decoding,
        on_progress=on_progress,
    )
    result = {
        "target": target,
        "model": record["model"],
        "adapter_path": str(adapter) if adapter else None,
        "suite": "trainjudge-regression-v1",
        "run": str(run_dir),
        "decoding": asdict(decoding),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - start, 1),
        **regression_check.to_dict(regression_check.score(items, outputs)),
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if cache is not None:
        _save_cached(cache, result)
    return result


def accuracy_line(result: dict) -> str:
    """One-line summary of an eval result: accuracy, the secondary metric and failure counts."""
    outcomes = result["outcomes"]
    correct = outcomes["correct"]
    issues = [f"{n:,} {name.replace('_', ' ')}" for name, n in outcomes.items() if n and name != "correct"]
    secondary = result.get("secondary") or {"label": "lenient", "value": result.get("lenient_accuracy", 0)}
    line = (
        f"Accuracy {result['accuracy']:.1%} ({correct:,}/{result['scored']:,}) · "
        f"{secondary['label'].lower()} {secondary['value']:.1%}"
    )
    return line + (f" · {' · '.join(issues)}" if issues else "")
