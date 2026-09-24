"""Prepare and run a fine-tuning run: audit, filter, split, train, record."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from trainjudge import __version__, mlx_backend, runs
from trainjudge.dataset_audit import LOW_QUALITY, AuditReport, audit_dataset


class SensitiveDataError(runs.RunError):
    """The dataset contains sensitive identifiers and the caller didn't allow them."""


@dataclass
class PreparedRun:
    run_dir: Path
    audit: AuditReport
    splits: runs.Splits
    config: mlx_backend.LoraConfig
    command: list[str]
    dropped: dict[str, int]
    epochs: float
    record: dict


def prepare_run(
    dataset: Path,
    model: str,
    runs_dir: Path = runs.DEFAULT_RUNS_DIR,
    *,
    epochs: float = 2.0,
    iters: int | None = None,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    rank: int = 16,
    num_layers: int = 16,
    max_seq_length: int = 2048,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
    keep_low_quality: bool = False,
    allow_sensitive_data: bool = False,
    goal: str | None = None,
) -> PreparedRun:
    audit = audit_dataset(dataset)
    if audit.sensitive and not allow_sensitive_data:
        kinds = ", ".join(f"{k} ({len(v)} rows)" for k, v in audit.sensitive.items())
        raise SensitiveDataError(
            f"the dataset contains sensitive identifiers: {kinds}. Mask them before training "
            "(see `trainjudge audit` for line numbers), or pass --allow-sensitive-data."
        )

    rows = runs.training_rows(audit, keep_low_quality=keep_low_quality)
    splits = runs.split_rows(rows, valid_fraction, test_fraction, seed)
    counts = audit.counts()
    dropped = {"duplicate": counts["duplicate"], "malformed": counts["malformed"]}
    if not keep_low_quality:
        dropped[LOW_QUALITY] = counts[LOW_QUALITY]

    if iters is None:
        iters = max(1, math.ceil(epochs * len(splits.train) / batch_size))
    else:
        epochs = round(iters * batch_size / len(splits.train), 2)
    config = mlx_backend.LoraConfig(
        model=runs.resolve_model(model),
        iters=iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rank=rank,
        num_layers=num_layers,
        max_seq_length=max_seq_length,
        seed=seed,
        steps_per_eval=max(10, iters // 10),
    )

    run_dir = runs.new_run_dir(Path(runs_dir), Path(dataset))
    run_dir.mkdir(parents=True)
    runs.write_splits(run_dir, splits)
    config_path = mlx_backend.write_config(config, run_dir)
    command = mlx_backend.build_command(config, config_path)

    record = {
        "trainjudge_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "prepared",
        "goal": goal,
        "model": config.model,
        "method": "lora",
        "backend": {"name": "mlx", "mlx_lm_version": mlx_backend.mlx_lm_version()},
        "dataset": {
            "path": str(dataset),
            "sha256": runs.file_sha256(Path(dataset)),
            "format": audit.format,
            "rows": audit.total,
            "audit_counts": counts,
        },
        "prep": {
            "dropped": dropped,
            "kept": len(rows),
            "splits": splits.sizes(),
            "split_method": "grouped by normalized completion",
            "valid_fraction": valid_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
        },
        "config": {**asdict(config), "epochs": epochs},
        "command": command,
        "training": None,
    }
    runs.write_run_json(run_dir, record)
    return PreparedRun(run_dir, audit, splits, config, command, dropped, epochs, record)


def run_training(prepared: PreparedRun, on_event=lambda e: None) -> mlx_backend.TrainingSummary:
    record = prepared.record
    try:
        summary = mlx_backend.train(prepared.command, prepared.run_dir, on_event)
    except mlx_backend.TrainingFailed as e:
        record["status"] = "failed"
        record["error"] = str(e)
        runs.write_run_json(prepared.run_dir, record)
        raise
    record["status"] = "trained"
    record["training"] = summary.to_dict()
    runs.write_run_json(prepared.run_dir, record)
    return summary
