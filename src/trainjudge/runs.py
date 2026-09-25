"""Run directories: one folder per training run with its data splits, config and logs.

trainjudge-runs/2026-09-24-sql_generation/
  run.json            what was trained, on what, and how it went
  data/train.jsonl    training split
  data/valid.jsonl    validation split (loss during training)
  data/test.jsonl     held out; only `trainjudge verify` reads it
  adapters/           LoRA weights
  logs/               raw backend output and the parsed loss curve
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from trainjudge.dataset_audit import CLEAN, LOW_QUALITY, AuditReport, normalize

DEFAULT_RUNS_DIR = Path("trainjudge-runs")
MODEL_ALIASES = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
}
# Only the fields mlx-lm reads, per dataset format.
FORMAT_FIELDS = {"completions": ("prompt", "completion"), "chat": ("messages",), "text": ("text",)}
GROUPED = "grouped by normalized completion"
PER_ROW = "per row (completions are labels shared by many rows)"
# Grouping keeps paraphrases of one answer out of both train and test. It only makes
# sense when answers are paraphrase-style: many small groups. Label-like answers
# (a few values shared by many rows) are split per row, or whole labels would be
# missing from training.
MIN_GROUPS_FOR_GROUPING = 50
MAX_MEDIAN_GROUP_SIZE = 10
GENERIC_STEMS = {"data", "train", "dataset", "examples", "all"}


class RunError(Exception):
    """The dataset can't be turned into a training run."""


@dataclass
class Splits:
    train: list[dict]
    valid: list[dict]
    test: list[dict]
    method: str = GROUPED

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "valid": len(self.valid), "test": len(self.test)}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def training_rows(report: AuditReport, keep_low_quality: bool = False) -> list[tuple[str, dict]]:
    """(group key, mlx-ready row) for every row that should be trained on.

    Duplicates and malformed rows are always dropped. The group key is the
    normalized completion, so paraphrases of one answer stay in one split.
    """
    keep = {CLEAN, LOW_QUALITY} if keep_low_quality else {CLEAN}
    fields = FORMAT_FIELDS.get(report.format)
    if fields is None:
        raise RunError("dataset format not recognized; nothing to train on")
    rows = []
    for r in report.rows:
        if r.status in keep:
            obj = json.loads(r.raw)
            rows.append((normalize(r.example.completion), {k: obj[k] for k in fields}))
    return rows


def split_rows(
    rows: list[tuple[str, dict]], valid_fraction: float, test_fraction: float, seed: int
) -> Splits:
    """Deterministic split: test first, then valid, the rest train.

    Rows are grouped by answer when answers are paraphrase-style, so paraphrases of
    one answer never land in both train and test; otherwise each row is its own group.
    """
    if valid_fraction <= 0 or test_fraction <= 0 or valid_fraction + test_fraction >= 1:
        raise RunError("valid and test fractions must be positive and sum to less than 1")
    groups: dict[str, list[dict]] = defaultdict(list)
    for key, row in rows:
        groups[key].append(row)
    if len(groups) < 3:
        raise RunError(f"need at least 3 distinct completions to split, found {len(groups)}")

    sizes = sorted(len(g) for g in groups.values())
    method = GROUPED
    if len(groups) < MIN_GROUPS_FOR_GROUPING or sizes[len(sizes) // 2] > MAX_MEDIAN_GROUP_SIZE:
        method = PER_ROW
        groups = {f"{i:08d}": [row] for i, (_, row) in enumerate(rows)}

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    test_target = max(1, round(len(rows) * test_fraction))
    valid_target = max(1, round(len(rows) * valid_fraction))
    splits = Splits([], [], [], method)
    for key in keys:
        if len(splits.test) < test_target:
            splits.test.extend(groups[key])
        elif len(splits.valid) < valid_target:
            splits.valid.extend(groups[key])
        else:
            splits.train.extend(groups[key])
    if not splits.train:
        raise RunError("dataset too small: nothing left for training after the held-out splits")
    return splits


def new_run_dir(runs_dir: Path, dataset: Path, today: date | None = None) -> Path:
    stem = dataset.stem if dataset.stem.lower() not in GENERIC_STEMS else dataset.parent.name
    base = f"{(today or datetime.now().astimezone().date()).isoformat()}-{stem or 'run'}"
    candidate = runs_dir / base
    n = 2
    while candidate.exists():
        candidate = runs_dir / f"{base}-{n}"
        n += 1
    return candidate


def write_splits(run_dir: Path, splits: Splits) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", splits.train), ("valid", splits.valid), ("test", splits.test)):
        (data_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_json(run_dir: Path, data: dict) -> None:
    (run_dir / "run.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_run_json(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
