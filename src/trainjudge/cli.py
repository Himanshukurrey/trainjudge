"""TrainJudge command-line interface."""

from __future__ import annotations

import click

from trainjudge import __version__


def _not_yet(feature: str, day: int) -> None:
    raise click.ClickException(f"`{feature}` is not implemented yet (planned for Day {day}).")


@click.group()
@click.version_option(__version__, prog_name="trainjudge")
def main() -> None:
    """Decide whether to fine-tune, then verify it actually worked."""


@main.command()
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--model", required=True, help="Base model name, e.g. Qwen/Qwen3-0.6B.")
@click.option("--goal", required=True, help="What you want fine-tuning to achieve.")
def diagnose(dataset: str, model: str, goal: str) -> None:
    """Classify the goal: knowledge / format / cost / prompt gap."""
    _not_yet("diagnose", 3)


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def audit(path: str) -> None:
    """Audit a JSONL dataset for duplicates, malformed rows and low quality."""
    _not_yet("audit", 2)


@main.command()
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--model", required=True, help="Base model name, e.g. Qwen/Qwen3-0.6B.")
@click.option("--method", type=click.Choice(["lora"]), default="lora", show_default=True)
def train(dataset: str, model: str, method: str) -> None:
    """Fine-tune locally via MLX LoRA."""
    _not_yet("train", 4)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def verify(run_dir: str) -> None:
    """Compare baseline vs fine-tuned on the task metric and issue a verdict."""
    _not_yet("verify", 6)


if __name__ == "__main__":
    main()
