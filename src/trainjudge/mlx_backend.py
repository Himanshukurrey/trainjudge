"""Local LoRA fine-tuning through mlx-lm (Apple Silicon).

Training runs mlx-lm as a subprocess (`python -m mlx_lm lora`), which keeps
TrainJudge independent of mlx-lm's training internals. Its progress lines are
parsed into a loss curve as they stream. Generation for evals uses mlx-lm's
Python API, imported only when needed.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

_TRAIN_RE = re.compile(
    r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+), It/sec ([\d.]+), "
    r"Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB"
)
_VAL_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+), Val took ([\d.]+)s")


class TrainingFailed(Exception):
    def __init__(self, message: str, log_tail: str = ""):
        super().__init__(message)
        self.log_tail = log_tail


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


def unavailable_reason() -> str | None:
    """Why MLX training can't run here, or None if it can."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "MLX training needs an Apple Silicon Mac"
    try:
        metadata.version("mlx-lm")
    except metadata.PackageNotFoundError:
        return "mlx-lm is not installed (pip install 'trainjudge[mlx]')"
    return None


def mlx_lm_version() -> str | None:
    try:
        return metadata.version("mlx-lm")
    except metadata.PackageNotFoundError:
        return None


def write_config(cfg: LoraConfig, run_dir: Path) -> Path:
    """mlx-lm's YAML config. JSON is valid YAML, so no YAML dependency is needed."""
    config = {
        "model": cfg.model,
        "data": str(run_dir / "data"),
        "adapter_path": str(run_dir / "adapters"),
        "fine_tune_type": "lora",
        "iters": cfg.iters,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "num_layers": cfg.num_layers,
        "max_seq_length": cfg.max_seq_length,
        "seed": cfg.seed,
        "steps_per_report": cfg.steps_per_report,
        "steps_per_eval": cfg.steps_per_eval,
        "val_batches": cfg.val_batches,
        "save_every": cfg.iters + 1,  # final weights only, no intermediate checkpoints
        "lora_parameters": {"rank": cfg.rank, "scale": cfg.scale, "dropout": cfg.dropout},
    }
    path = run_dir / "mlx_config.yaml"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def build_command(cfg: LoraConfig, config_path: Path, python: str = sys.executable) -> list[str]:
    # Boolean switches must be on the command line: mlx-lm only takes config
    # values for options left unset, and store_true flags default to False.
    cmd = [python, "-m", "mlx_lm", "lora", "--train", "-c", str(config_path)]
    if cfg.mask_prompt:
        cmd.append("--mask-prompt")
    return cmd


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
) -> TrainingSummary:
    """Run the training command, streaming parsed events; raise TrainingFailed on error."""
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    start = time.monotonic()
    with (
        (logs / "mlx.log").open("w") as raw_log,
        (logs / "training_log.jsonl").open("w") as event_log,
    ):
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            raw_log.write(line)
            event = parse_line(line)
            if event:
                events.append(event)
                event_log.write(json.dumps(event) + "\n")
                event_log.flush()
                on_event(event)
        returncode = proc.wait()
    duration = time.monotonic() - start

    if returncode != 0:
        tail = "".join((logs / "mlx.log").read_text().splitlines(keepends=True)[-20:])
        raise TrainingFailed(f"mlx-lm exited with status {returncode}", tail)
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


def generate_outputs(
    model: str,
    prompts: list[str],
    adapter_path: Path | None = None,
    decoding: DecodingConfig | None = None,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
) -> list[str]:
    """Greedy completions for each prompt, rendered with the model's chat template.

    Thinking is disabled by default. For Qwen3 this renders the same empty
    think block the training data contains, so the fine-tuned model sees
    exactly the prompt format it was trained on.
    """
    import importlib

    from mlx_lm import load

    batch_generate = importlib.import_module("mlx_lm.generate").batch_generate
    decoding = decoding or DecodingConfig()
    llm, tokenizer = load(model, adapter_path=str(adapter_path) if adapter_path else None)

    def render(prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        if decoding.system_prompt:
            messages.insert(0, {"role": "system", "content": decoding.system_prompt})
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=decoding.enable_thinking
        )

    outputs: list[str] = []
    for start in range(0, len(prompts), decoding.batch_size):
        batch = [render(p) for p in prompts[start : start + decoding.batch_size]]
        response = batch_generate(llm, tokenizer, batch, max_tokens=decoding.max_tokens)
        outputs.extend(response.texts)
        on_progress(len(outputs), len(prompts))
    return outputs
