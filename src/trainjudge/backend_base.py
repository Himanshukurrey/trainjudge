"""What every training backend shares: config, progress format, process runner, summary.

A backend trains in a subprocess that prints progress lines in mlx-lm's format

    Iter 10: Train loss 2.314, Learning Rate 5.000e-05, It/sec 2.500, Tokens/sec 300.0,
    Trained Tokens 1200, Peak mem 1.250 GB
    Iter 10: Val loss 2.100, Val took 1.5s

so one parser drives the live progress, the loss curve, `trainjudge status` and
the training summary, whichever backend trained the run.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

_TRAIN_RE = re.compile(
    r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+), It/sec ([\d.]+), "
    r"Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB"
)
_VAL_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+), Val took ([\d.]+)s")


class TrainingFailed(Exception):
    def __init__(self, message: str, log_tail: str = "", log_path: Path | None = None):
        super().__init__(message)
        self.log_tail = log_tail
        self.log_path = log_path


@dataclass
class LoraConfig:
    model: str
    iters: int
    batch_size: int = 4
    learning_rate: float = 5e-5
    rank: int = 16
    scale: float = 20.0
    dropout: float = 0.0
    num_layers: int = 16
    max_seq_length: int = 2048
    seed: int = 0
    mask_prompt: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 50
    val_batches: int = -1


@dataclass
class TrainingSummary:
    iters_completed: int
    duration_s: float
    first_train_loss: float | None
    final_train_loss: float | None
    first_val_loss: float | None
    final_val_loss: float | None
    best_val_loss: float | None
    peak_mem_gb: float | None
    trained_tokens: int | None

    @property
    def train_loss_drop_pct(self) -> float | None:
        if not self.first_train_loss or self.final_train_loss is None:
            return None
        return (self.first_train_loss - self.final_train_loss) / self.first_train_loss * 100

    def to_dict(self) -> dict:
        return {**asdict(self), "train_loss_drop_pct": self.train_loss_drop_pct}


def parse_line(line: str) -> dict | None:
    if m := _TRAIN_RE.search(line):
        return {
            "type": "train",
            "iter": int(m[1]),
            "loss": float(m[2]),
            "learning_rate": float(m[3]),
            "it_per_sec": float(m[4]),
            "tokens_per_sec": float(m[5]),
            "trained_tokens": int(m[6]),
            "peak_mem_gb": float(m[7]),
        }
    if m := _VAL_RE.search(line):
        return {"type": "val", "iter": int(m[1]), "loss": float(m[2]), "took_s": float(m[3])}
    return None


def train(
    command: list[str],
    run_dir: Path,
    on_event: Callable[[dict], None] = lambda e: None,
    log_name: str = "mlx.log",
    label: str = "mlx-lm",
) -> TrainingSummary:
    """Run the training command, streaming parsed events; raise TrainingFailed on error."""
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / log_name
    events: list[dict] = []
    start = time.monotonic()
    with (
        log_path.open("w", encoding="utf-8") as raw_log,
        (logs / "training_log.jsonl").open("w", encoding="utf-8") as event_log,
    ):
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            raw_log.write(line)
            raw_log.flush()  # so the log can be followed live
            event = parse_line(line)
            if event:
                events.append(event)
                event_log.write(json.dumps(event) + "\n")
                event_log.flush()
                on_event(event)
        returncode = proc.wait()
    duration = time.monotonic() - start

    if returncode != 0:
        tail = "".join(log_path.read_text(encoding="utf-8").splitlines(keepends=True)[-20:])
        raise TrainingFailed(f"{label} exited with status {returncode}", tail, log_path)
    return summarize(events, duration)


def summarize(events: list[dict], duration_s: float) -> TrainingSummary:
    train_events = [e for e in events if e["type"] == "train"]
    val_events = [e for e in events if e["type"] == "val"]
    return TrainingSummary(
        iters_completed=max((e["iter"] for e in events), default=0),
        duration_s=round(duration_s, 1),
        first_train_loss=train_events[0]["loss"] if train_events else None,
        final_train_loss=train_events[-1]["loss"] if train_events else None,
        first_val_loss=val_events[0]["loss"] if val_events else None,
        final_val_loss=val_events[-1]["loss"] if val_events else None,
        best_val_loss=min((e["loss"] for e in val_events), default=None),
        peak_mem_gb=max((e["peak_mem_gb"] for e in train_events), default=None),
        trained_tokens=train_events[-1]["trained_tokens"] if train_events else None,
    )


@dataclass
class DecodingConfig:
    max_tokens: int = 512
    batch_size: int = 16
    enable_thinking: bool = False
    system_prompt: str | None = None
