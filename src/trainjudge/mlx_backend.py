"""Local LoRA fine-tuning through mlx-lm (Apple Silicon).

Training runs mlx-lm as a subprocess (`python -m mlx_lm lora`), which keeps
TrainJudge independent of mlx-lm's training internals. Its progress lines are
parsed into a loss curve as they stream. Generation for evals uses mlx-lm's
Python API, imported only when needed.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

from trainjudge.backend_base import (  # noqa: F401  (re-exported for callers and tests)
    DecodingConfig,
    LoraConfig,
    TrainingFailed,
    TrainingSummary,
    parse_line,
    summarize,
)
from trainjudge.backend_base import train as _train

NAME = "mlx"
LOG_NAME = "mlx.log"


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
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def build_command(cfg: LoraConfig, config_path: Path, python: str = sys.executable) -> list[str]:
    # Boolean switches must be on the command line: mlx-lm only takes config
    # values for options left unset, and store_true flags default to False.
    cmd = [python, "-m", "mlx_lm", "lora", "--train", "-c", str(config_path)]
    if cfg.mask_prompt:
        cmd.append("--mask-prompt")
    return cmd


def train(command: list[str], run_dir: Path, on_event: Callable[[dict], None] = lambda e: None):
    return _train(command, run_dir, on_event, log_name=LOG_NAME, label="mlx-lm")


def versions() -> dict:
    return {"mlx_lm_version": mlx_lm_version()}


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
