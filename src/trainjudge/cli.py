"""TrainJudge command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from trainjudge import (
    __version__,
    dataset_audit,
    diagnosis,
    evaluation,
    mlx_backend,
    runs,
    training,
)


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
@click.option("--model", required=True, help="Base model, e.g. Qwen3-0.6B or a Hugging Face repo.")
@click.option("--method", type=click.Choice(["lora"]), default="lora", show_default=True)
@click.option("--epochs", type=float, default=2.0, show_default=True)
@click.option("--iters", type=int, help="Training steps (overrides --epochs).")
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--learning-rate", type=float, default=5e-5, show_default=True)
@click.option("--rank", type=int, default=16, show_default=True, help="LoRA rank.")
@click.option("--num-layers", type=int, default=16, show_default=True,
              help="Layers to adapt (-1 for all).")
@click.option("--max-seq-length", type=int, default=2048, show_default=True)
@click.option("--valid-fraction", type=float, default=0.1, show_default=True)
@click.option("--test-fraction", type=float, default=0.1, show_default=True,
              help="Held out for `trainjudge verify`.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--keep-low-quality", is_flag=True, help="Train on rows the audit flags as low-quality.")
@click.option("--allow-sensitive-data", is_flag=True,
              help="Train even if the audit finds card numbers, Aadhaar, PAN, etc.")
@click.option("--goal", help="What the fine-tune is for (recorded in run.json).")
@click.option("--runs-dir", type=click.Path(file_okay=False), default=str(runs.DEFAULT_RUNS_DIR),
              show_default=True)
@click.option("--dry-run", is_flag=True, help="Prepare the run directory but don't train.")
def train(dataset: str, model: str, method: str, runs_dir: str, dry_run: bool, **options) -> None:
    """Fine-tune locally via MLX LoRA."""
    if not dry_run and (reason := mlx_backend.unavailable_reason()):
        raise click.ClickException(f"{reason}. Use --dry-run to prepare the run anyway.")
    try:
        prepared = training.prepare_run(Path(dataset), model, Path(runs_dir), **options)
    except runs.RunError as e:
        raise click.ClickException(str(e)) from e

    cfg = prepared.config
    sizes = prepared.splits.sizes()
    dropped = ", ".join(f"{n:,} {k.replace('_', '-')}" for k, n in prepared.dropped.items() if n)
    click.echo(f"Prepared run {prepared.run_dir}/")
    click.echo(f"  Dataset:  {prepared.audit.total:,} rows → {prepared.record['prep']['kept']:,} "
               f"usable" + (f" (dropped {dropped})" if dropped else ""))
    click.echo(f"  Split:    {sizes['train']:,} train · {sizes['valid']:,} valid · "
               f"{sizes['test']:,} test (held out for verify)")
    click.echo("")
    click.echo("Training via MLX LoRA (local, Apple Silicon)...")
    click.echo(f"  Base model:  {cfg.model}")
    click.echo(f"  Method:      LoRA (rank {cfg.rank}, {cfg.num_layers} layers, "
               f"lr {cfg.learning_rate:g})")
    click.echo(f"  Examples:    {sizes['train']:,}")
    click.echo(f"  Steps:       {cfg.iters:,} ({prepared.epochs:g} epochs, batch {cfg.batch_size})")

    if dry_run:
        click.echo("\nDry run: not training. To train, run:\n  " + " ".join(prepared.command))
        return

    def on_event(event: dict) -> None:
        if event["type"] == "train":
            remaining = (cfg.iters - event["iter"]) / event["it_per_sec"]
            click.echo(f"  step {event['iter']:>5,}/{cfg.iters:,} · train loss "
                       f"{event['loss']:.3f} · {event['it_per_sec']:.2f} it/s · "
                       f"ETA {_duration(remaining)}")
        else:
            click.echo(f"  step {event['iter']:>5,}/{cfg.iters:,} · val loss {event['loss']:.3f}")

    try:
        summary = training.run_training(prepared, on_event)
    except mlx_backend.TrainingFailed as e:
        click.echo(e.log_tail, err=True)
        raise click.ClickException(f"training failed: {e}. Full log: "
                                   f"{prepared.run_dir / 'logs' / 'mlx.log'}") from e

    click.echo("")
    click.echo(f"Training finished in {_duration(summary.duration_s)}.")
    if summary.train_loss_drop_pct is not None:
        click.echo(f"  Train loss {summary.first_train_loss:.3f} → {summary.final_train_loss:.3f} "
                   f"({-summary.train_loss_drop_pct:+.1f}%)")
    if summary.first_val_loss is not None:
        click.echo(f"  Val loss   {summary.first_val_loss:.3f} → {summary.final_val_loss:.3f}")
    click.echo(f"  Adapters:  {prepared.run_dir / 'adapters'}")
    click.echo("\nA lower loss doesn't prove the model got better at the task. Check with:")
    click.echo(f"  trainjudge verify {prepared.run_dir}")


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@main.command("eval")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--db", "db_path", type=click.Path(exists=True, dir_okay=False),
              help="SQLite database (.sqlite/.db) or SQL script (.sql) to run queries against. "
                   "Defaults to the one recorded by a previous eval of this run.")
@click.option("--target", type=click.Choice(["baseline", "finetuned", "both"]), default="both",
              show_default=True)
@click.option("--limit", type=int, help="Only evaluate the first N test examples.")
@click.option("--max-tokens", type=int, default=512, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--system-prompt", help="System prompt for both targets (default: none).")
@click.option("--json", "as_json", is_flag=True, help="Print results as JSON.")
def eval_command(run_dir: str, db_path: str | None, target: str, limit: int | None,
                 max_tokens: int, batch_size: int, system_prompt: str | None,
                 as_json: bool) -> None:
    """Measure SQL execution accuracy of the base model and the fine-tuned adapter."""
    run = Path(run_dir)
    try:
        record = runs.read_run_json(run)
    except FileNotFoundError as e:
        raise click.ClickException(f"{run} has no run.json; is it a trainjudge run?") from e
    db_path = db_path or (record.get("task") or {}).get("database")
    if not db_path:
        raise click.UsageError("--db is required the first time a run is evaluated.")
    if not as_json and (reason := mlx_backend.unavailable_reason()):
        raise click.ClickException(reason)

    decoding = mlx_backend.DecodingConfig(max_tokens=max_tokens, batch_size=batch_size,
                                          system_prompt=system_prompt)
    targets = evaluation.TARGETS if target == "both" else (target,)
    labels = {"baseline": "Baseline (no fine-tuning)", "finetuned": "Fine-tuned"}
    log = (lambda *a, **k: None) if as_json else click.echo
    rows = len(evaluation.test_rows(run, limit))
    log(f"SQL execution accuracy on {rows:,} held-out test examples")
    log(f"  Database: {db_path}\n")

    results = {}
    for t in targets:
        log(f"{labels[t]}: {record['model']}" + (" + LoRA adapter" if t == "finetuned" else ""))
        try:
            result = evaluation.evaluate_target(
                run, t, Path(db_path), decoding, limit,
                on_progress=lambda done, total: log(f"  generated {done:,}/{total:,}"),
            )
        except runs.RunError as e:
            raise click.ClickException(str(e)) from e
        results[t] = result
        log(f"  {_accuracy_line(result)}\n")

    if as_json:
        click.echo(json.dumps({t: {k: v for k, v in r.items() if k != "examples"}
                               for t, r in results.items()}, indent=2))
        return
    if len(results) == 2:
        delta = (results["finetuned"]["accuracy"] - results["baseline"]["accuracy"]) * 100
        click.echo(f"Change: {delta:+.1f} points. Run `trainjudge verify {run}` for the verdict.")
    click.echo(f"Per-example results: {run / 'eval'}/")


def _accuracy_line(result: dict) -> str:
    outcomes = result["outcomes"]
    correct = outcomes["correct"]
    issues = [f"{n:,} {name.replace('_', ' ')}" for name, n in outcomes.items()
              if n and name != "correct"]
    line = (f"Accuracy {result['accuracy']:.1%} ({correct:,}/{result['scored']:,}) · "
            f"lenient {result['lenient_accuracy']:.1%}")
    return line + (f" · {' · '.join(issues)}" if issues else "")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def verify(run_dir: str) -> None:
    """Compare baseline vs fine-tuned on the task metric and issue a verdict."""
    _not_yet("verify")


if __name__ == "__main__":
    main()
