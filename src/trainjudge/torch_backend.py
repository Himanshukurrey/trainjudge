"""PyTorch backend: LoRA fine-tuning and generation with transformers + peft.

Runs on NVIDIA GPUs (CUDA, on Windows or Linux), Apple GPUs (MPS) or the CPU,
picked automatically. Training runs `python -m trainjudge.torch_train` as a
subprocess that prints progress in the shared format (see backend_base).
Generation for evals uses transformers directly, rendering prompts exactly like
training does, so the fine-tuned model sees the format it was trained on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

from trainjudge.backend_base import DecodingConfig, LoraConfig
from trainjudge.backend_base import train as _train

NAME = "torch"
LOG_NAME = "torch.log"
REQUIRED = ("torch", "transformers", "peft")
DEVICES = ("auto", "cuda", "mps", "cpu")


def unavailable_reason() -> str | None:
    """Why PyTorch training can't run here, or None if it can."""
    missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
    if missing:
        return f"{', '.join(missing)} not installed (pip install 'trainjudge[cuda]')"
    return None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def versions() -> dict:
    return {f"{p}_version": _version(p) for p in REQUIRED}


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_config(cfg: LoraConfig, run_dir: Path, device: str = "auto") -> Path:
    config = {
        "model": cfg.model,
        "data": str(run_dir / "data"),
        "adapter_path": str(run_dir / "adapters"),
        "device": device,
        "iters": cfg.iters,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "rank": cfg.rank,
        "scale": cfg.scale,
        "dropout": cfg.dropout,
        "num_layers": cfg.num_layers,
        "max_seq_length": cfg.max_seq_length,
        "seed": cfg.seed,
        "mask_prompt": cfg.mask_prompt,
        "steps_per_report": cfg.steps_per_report,
        "steps_per_eval": cfg.steps_per_eval,
        "val_batches": cfg.val_batches,
    }
    path = run_dir / "torch_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def build_command(cfg: LoraConfig, config_path: Path, python: str = sys.executable) -> list[str]:
    return [python, "-m", "trainjudge.torch_train", "--config", str(config_path)]


def train(command: list[str], run_dir: Path, on_event: Callable[[dict], None] = lambda e: None):
    return _train(command, run_dir, on_event, log_name=LOG_NAME, label="PyTorch training")


def generate_outputs(
    model: str,
    prompts: list[str],
    adapter_path: Path | None = None,
    decoding: DecodingConfig | None = None,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    device: str = "auto",
) -> list[str]:
    """Greedy completions for each prompt, rendered with the model's chat template."""
    import torch
    from transformers import AutoTokenizer

    from trainjudge.torch_train import load_model, pick_dtype

    decoding = decoding or DecodingConfig()
    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    llm = load_model(model, pick_dtype(device, torch), device)
    if adapter_path is not None:
        from peft import PeftModel

        llm = PeftModel.from_pretrained(llm, str(adapter_path))
    llm.eval()
    eos = llm.generation_config.eos_token_id or tokenizer.eos_token_id

    def render(prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        if decoding.system_prompt:
            messages.insert(0, {"role": "system", "content": decoding.system_prompt})
        return list(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=decoding.enable_thinking,
                tokenize=True,
                return_dict=False,
            )
        )

    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), decoding.batch_size):
            batch = [render(p) for p in prompts[start : start + decoding.batch_size]]
            width = max(len(ids) for ids in batch)
            # Left-pad so every prompt ends right where generation starts.
            input_ids = torch.tensor([[pad_id] * (width - len(ids)) + ids for ids in batch], device=device)
            attention = torch.tensor(
                [[0] * (width - len(ids)) + [1] * len(ids) for ids in batch], device=device
            )
            with torch.no_grad():
                generated = llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention,
                    max_new_tokens=decoding.max_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos,
                )
            for row in generated[:, width:]:
                outputs.append(tokenizer.decode(row, skip_special_tokens=True))
            on_progress(len(outputs), len(prompts))
    finally:
        # Hand the GPU memory back: PyTorch's caching allocator otherwise keeps it reserved
        # in this process, and the next step (the training subprocess, or the next eval pass)
        # would then run out of memory loading its own copy of the model.
        del llm
        release_memory(device)
    return outputs


def release_memory(device: str) -> None:
    """Free cached accelerator memory held by this process."""
    import gc

    import torch

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
