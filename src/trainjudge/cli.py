"""TrainJudge command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from trainjudge import __version__, dataset_audit, diagnosis


def _not_yet(feature: str) -> None:
    raise click.ClickException(f"`{feature}` is not implemented yet.")


@click.group()
@click.version_option(__version__, prog_name="trainjudge")
def main() -> None:
    """Decide whether to fine-tune, then verify it actually worked."""


@main.command()
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--model", required=True, help="Base model name, e.g. Qwen/Qwen3-0.6B.")
@click.option("--goal", required=True, help="What you want fine-tuning to achieve.")
@click.option(
    "--tried-prompting/--not-tried-prompting",
    default=None,
    help="Whether few-shot prompting has already been tried (default: unknown).",
)
@click.option("--json", "as_json", is_flag=True, help="Print the diagnosis as JSON.")
def diagnose(
    dataset: str, model: str, goal: str, tried_prompting: bool | None, as_json: bool
) -> None:
    """Classify the goal: knowledge / format / cost / prompt gap."""
    report = dataset_audit.audit_dataset(dataset)
    result = diagnosis.diagnose(goal, report, model=model, tried_prompting=tried_prompting)
    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(diagnosis.format_diagnosis(result))


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
@click.option(
    "--write-clean",
    type=click.Path(dir_okay=False),
    help="Write a copy without duplicate and malformed rows to this path.",
)
@click.option(
    "--drop-low-quality", is_flag=True, help="With --write-clean, also drop low-quality rows."
)
def audit(path: str, as_json: bool, write_clean: str | None, drop_low_quality: bool) -> None:
    """Audit a JSONL dataset for duplicates, malformed rows and low quality."""
    if drop_low_quality and not write_clean:
        raise click.UsageError("--drop-low-quality only applies with --write-clean.")
    if write_clean and Path(write_clean).resolve() == Path(path).resolve():
        raise click.UsageError("--write-clean must not overwrite the input dataset.")

    report = dataset_audit.audit_dataset(path)
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(dataset_audit.format_report(report))

    if write_clean:
        kept = dataset_audit.write_clean_dataset(report, write_clean, drop_low_quality)
        click.echo(f"\nWrote {kept:,} rows to {write_clean}.", err=as_json)
    elif not as_json:
        click.echo(
            "\nNothing was removed. Use --write-clean <path> to save a copy "
            "without duplicate and malformed rows."
        )


@main.command()
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--model", required=True, help="Base model name, e.g. Qwen/Qwen3-0.6B.")
@click.option("--method", type=click.Choice(["lora"]), default="lora", show_default=True)
def train(dataset: str, model: str, method: str) -> None:
    """Fine-tune locally via MLX LoRA."""
    _not_yet("train")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def verify(run_dir: str) -> None:
    """Compare baseline vs fine-tuned on the task metric and issue a verdict."""
    _not_yet("verify")


if __name__ == "__main__":
    main()
