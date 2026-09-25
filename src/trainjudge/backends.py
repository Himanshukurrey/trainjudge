"""Training/inference backends and how a run picks one.

- mlx:   mlx-lm on Apple Silicon (fastest on a Mac)
- torch: transformers + peft on NVIDIA GPUs (Windows/Linux), Apple MPS or CPU

`auto` picks MLX when it's usable and PyTorch otherwise. The backend that
trained a run is recorded in run.json, and evals of that run use the same one,
because the two write different adapter formats.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from trainjudge import mlx_backend, torch_backend

MLX = mlx_backend.NAME
TORCH = torch_backend.NAME
BACKENDS = {MLX: mlx_backend, TORCH: torch_backend}
CHOICES = ("auto", MLX, TORCH)


def resolve(name: str = "auto") -> str:
    """The backend to use: the one asked for, or the best one available here."""
    if name != "auto":
        if name not in BACKENDS:
            raise ValueError(f"unknown backend {name!r}; choose from {', '.join(CHOICES)}")
        return name
    if mlx_backend.unavailable_reason() is None:
        return MLX
    if torch_backend.unavailable_reason() is None:
        return TORCH
    # Nothing installed yet: suggest what fits this machine.
    return MLX if sys.platform == "darwin" and platform.machine() == "arm64" else TORCH


def get(name: str):
    return BACKENDS[name]


def backend_of(record: dict) -> str:
    return (record.get("backend") or {}).get("name", MLX)


def unavailable_reason(name: str) -> str | None:
    return BACKENDS[name].unavailable_reason()


def adapter_ready(adapter_dir: Path) -> bool:
    """A trained adapter exists (mlx-lm and peft use different file names)."""
    return adapter_dir.is_dir() and any(adapter_dir.glob("*.safetensors"))


def generator(record: dict) -> Callable[..., list[str]]:
    """The generate function for a run, on the backend (and device) that trained it."""
    backend = record.get("backend") or {}
    name = backend.get("name", MLX)
    if name == TORCH:
        return partial(torch_backend.generate_outputs, device=backend.get("device", "auto"))
    return mlx_backend.generate_outputs
