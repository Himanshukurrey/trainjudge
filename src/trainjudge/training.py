"""Prepare and run a fine-tuning run: audit, filter, split, train, record."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from trainjudge import __version__, backends, pii, replay, runs, torch_backend
from trainjudge.backend_base import DecodingConfig, LoraConfig, TrainingFailed, TrainingSummary
from trainjudge.dataset_audit import LOW_QUALITY, AuditReport, audit_dataset


class SensitiveDataError(runs.RunError):
    """The dataset contains sensitive identifiers and the caller didn't allow them."""


@dataclass
class PreparedRun:
    run_dir: Path
    audit: AuditReport
    splits: runs.Splits
    config: LoraConfig
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
    mask_sensitive: bool = False,
    goal: str | None = None,
    replay_count: int = 0,
    backend: str = "auto",
    device: str = "auto",
) -> PreparedRun:
    audit = audit_dataset(dataset)
    if audit.sensitive and not (allow_sensitive_data or mask_sensitive):
        kinds = ", ".join(f"{k} ({len(v)} rows)" for k, v in audit.sensitive.items())
        raise SensitiveDataError(
            f"the dataset contains sensitive identifiers: {kinds}. Pass --mask-sensitive to "
            "replace them with placeholders like [EMAIL] before training, mask them yourself "
            "(see `trainjudge audit` for line numbers), or pass --allow-sensitive-data."
        )

    rows = runs.training_rows(audit, keep_low_quality=keep_low_quality)
    masked: Counter = Counter()
    if mask_sensitive:
        masked_rows = []
        for key, row in rows:
            row, counts = pii.mask_value(row)
            masked += counts
            masked_rows.append((key, row))
        rows = masked_rows
        leftover = set().union(*(pii.scan_value(row) for _, row in rows)) if rows else set()
        if leftover and not allow_sensitive_data:
            raise SensitiveDataError(
                f"masking left identifiers the placeholders didn't cover: {', '.join(sorted(leftover))}"
            )
    splits = runs.split_rows(rows, valid_fraction, test_fraction, seed)
    counts = audit.counts()
    dropped = {"duplicate": counts["duplicate"], "malformed": counts["malformed"]}
    if not keep_low_quality:
        dropped[LOW_QUALITY] = counts[LOW_QUALITY]

    if not 0 <= replay_count <= replay.max_replay():
        raise runs.RunError(f"--replay must be between 0 and {replay.max_replay()}")
    train_size = len(splits.train) + replay_count
    if iters is None:
        iters = max(1, math.ceil(epochs * train_size / batch_size))
    else:
        epochs = round(iters * batch_size / train_size, 2)
    backend_name = backends.resolve(backend)
    config = LoraConfig(
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
    module = backends.get(backend_name)
    if backend_name == backends.TORCH:
        config_path = torch_backend.write_config(config, run_dir, device)
    else:
        config_path = module.write_config(config, run_dir)
    command = module.build_command(config, config_path)
    backend_record = {"name": backend_name, **module.versions()}
    if backend_name == backends.TORCH:
        backend_record["device"] = device

    record = {
        "trainjudge_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "prepared",
        "goal": goal,
        "model": config.model,
        "method": "lora",
        "backend": backend_record,
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
            "split_method": splits.method,
            "valid_fraction": valid_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "replay": {"requested": replay_count, "added": 0},
            "masked": dict(masked),
            "mask_sensitive": mask_sensitive,
            "allow_sensitive_data": allow_sensitive_data,
            "keep_low_quality": keep_low_quality,
        },
        "config": {**asdict(config), "epochs": epochs},
        "command": command,
        "training": None,
    }
    runs.write_run_json(run_dir, record)
    return PreparedRun(run_dir, audit, splits, config, command, dropped, epochs, record)


def add_replay(
    prepared: PreparedRun,
    generate=None,
    on_progress=lambda done, total: None,
) -> int:
    """Append base-model answers to general prompts to the training split."""
    count = prepared.record["prep"]["replay"]["requested"]
    if not count:
        return 0
    prompts = replay.replay_prompts(count)
    generate = generate or backends.generator(prepared.record)
    outputs = generate(
        prepared.config.model,
        prompts,
        decoding=DecodingConfig(max_tokens=256),
        on_progress=on_progress,
    )
    rows = [{"prompt": p, "completion": o.strip()} for p, o in zip(prompts, outputs) if o.strip()]
    with (prepared.run_dir / "data" / "train.jsonl").open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    prepared.record["prep"]["replay"] = {
        "requested": count,
        "added": len(rows),
        "source": "base-model answers to trainjudge's general replay prompts",
    }
    runs.write_run_json(prepared.run_dir, prepared.record)
    return len(rows)


def run_training(prepared: PreparedRun, on_event=lambda e: None) -> TrainingSummary:
    record = prepared.record
    module = backends.get(backends.backend_of(record))
    try:
        summary = module.train(prepared.command, prepared.run_dir, on_event)
    except TrainingFailed as e:
        record["status"] = "failed"
        record["error"] = str(e)
        runs.write_run_json(prepared.run_dir, record)
        raise
    record["status"] = "trained"
    record["training"] = summary.to_dict()
    device_file = prepared.run_dir / "logs" / "device.json"
    if device_file.exists():
        record["backend"].update(json.loads(device_file.read_text(encoding="utf-8")))
    runs.write_run_json(prepared.run_dir, record)
    return summary
