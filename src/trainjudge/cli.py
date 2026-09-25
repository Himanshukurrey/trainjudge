"""TrainJudge command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from trainjudge import (
    __version__,
    backends,
    dataset_audit,
    diagnosis,
    domains,
    evaluation,
    runs,
    tasks,
    training,
    verdict,
    verification,
)
from trainjudge.backend_base import DecodingConfig, TrainingFailed
from trainjudge.status import StatusTracker

NOTIFY_HELP = "Show a desktop notification (macOS) when the command finishes or fails."


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
@click.option(
    "--domain",
    type=click.Choice(["auto", "none", *sorted(domains.PACKS)]),
    default="auto",
    show_default=True,
    help="Domain pack for industry-specific checks: detect it, turn it off, or force one.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the diagnosis as JSON.")
def diagnose(
    dataset: str, model: str, goal: str, tried_prompting: bool | None, domain: str, as_json: bool
) -> None:
    """Classify the goal: knowledge / format / cost / prompt gap."""
    report = dataset_audit.audit_dataset(dataset)
    result = diagnosis.diagnose(goal, report, model=model, tried_prompting=tried_prompting, domain=domain)
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
@click.option("--drop-low-quality", is_flag=True, help="With --write-clean, also drop low-quality rows.")
@click.option(
    "--mask-sensitive",
    is_flag=True,
    help="With --write-clean, replace card numbers, IDs, emails, phones etc. with placeholders.",
)
def audit(
    path: str, as_json: bool, write_clean: str | None, drop_low_quality: bool, mask_sensitive: bool
) -> None:
    """Audit a JSONL dataset for duplicates, malformed rows and low quality."""
    if (drop_low_quality or mask_sensitive) and not write_clean:
        raise click.UsageError("--drop-low-quality and --mask-sensitive only apply with --write-clean.")
    if write_clean and Path(write_clean).resolve() == Path(path).resolve():
        raise click.UsageError("--write-clean must not overwrite the input dataset.")

    report = dataset_audit.audit_dataset(path)
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(dataset_audit.format_report(report))

    if write_clean:
        kept = dataset_audit.write_clean_dataset(report, write_clean, drop_low_quality, mask_sensitive)
        click.echo(f"\nWrote {kept:,} rows to {write_clean}.", err=as_json)
        if mask_sensitive:
            _, masked = dataset_audit.clean_rows(report, drop_low_quality, mask_sensitive=True)
            summary = ", ".join(f"{k} ({n})" for k, n in masked.items()) or "nothing to mask"
            click.echo(f"Masked: {summary}.", err=as_json)
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
@click.option("--num-layers", type=int, default=16, show_default=True, help="Layers to adapt (-1 for all).")
@click.option("--max-seq-length", type=int, default=2048, show_default=True)
@click.option("--valid-fraction", type=float, default=0.1, show_default=True)
@click.option(
    "--test-fraction",
    type=float,
    default=0.1,
    show_default=True,
    help="Held out for `trainjudge verify`.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--keep-low-quality", is_flag=True, help="Train on rows the audit flags as low-quality.")
@click.option(
    "--mask-sensitive",
    is_flag=True,
    help="Replace card numbers, IDs, emails, phones etc. with placeholders like [EMAIL] before training.",
)
@click.option(
    "--allow-sensitive-data",
    is_flag=True,
    help="Train even if the audit finds card numbers, Aadhaar, PAN, etc.",
)
@click.option(
    "--replay",
    "replay_count",
    type=int,
    default=0,
    show_default=True,
    help="Mix in N base-model answers to general prompts to limit regressions.",
)
@click.option("--goal", help="What the fine-tune is for (recorded in run.json).")
@click.option(
    "--runs-dir",
    type=click.Path(file_okay=False),
    default=str(runs.DEFAULT_RUNS_DIR),
    show_default=True,
)
@click.option(
    "--backend",
    type=click.Choice(list(backends.CHOICES)),
    default="auto",
    show_default=True,
    help="mlx (Apple Silicon) or torch (NVIDIA CUDA, Apple MPS or CPU). auto picks MLX when it's usable.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    show_default=True,
    help="torch backend only: where to train and run evals.",
)
@click.option("--dry-run", is_flag=True, help="Prepare the run directory but don't train.")
@click.option("--notify", is_flag=True, help=NOTIFY_HELP)
def train(
    dataset: str,
    model: str,
    method: str,
    runs_dir: str,
    backend: str,
    device: str,
    dry_run: bool,
    notify: bool,
    **options,
) -> None:
    """Fine-tune with LoRA: MLX on Apple Silicon, or PyTorch on NVIDIA GPUs, MPS or CPU."""
    backend_name = backends.resolve(backend)
    if not dry_run and (reason := backends.unavailable_reason(backend_name)):
        raise click.ClickException(f"{reason}. Use --dry-run to prepare the run anyway.")
    if backend_name == backends.TORCH and device == "cuda" and not dry_run:
        import torch

        if not torch.cuda.is_available():
            raise click.ClickException("--device cuda was requested, but PyTorch can't see an NVIDIA GPU.")
    try:
        prepared = training.prepare_run(
            Path(dataset), model, Path(runs_dir), backend=backend_name, device=device, **options
        )
    except runs.RunError as e:
        raise click.ClickException(str(e)) from e

    cfg = prepared.config
    sizes = prepared.splits.sizes()
    dropped = ", ".join(f"{n:,} {k.replace('_', '-')}" for k, n in prepared.dropped.items() if n)
    click.echo(f"Prepared run {prepared.run_dir}/")
    click.echo(
        f"  Dataset:  {prepared.audit.total:,} rows → {prepared.record['prep']['kept']:,} "
        f"usable" + (f" (dropped {dropped})" if dropped else "")
    )
    click.echo(
        f"  Split:    {sizes['train']:,} train · {sizes['valid']:,} valid · "
        f"{sizes['test']:,} test (held out for verify)"
    )
    if masked := prepared.record["prep"].get("masked"):
        click.echo("  Masked:   " + ", ".join(f"{n:,} {k}" for k, n in masked.items()))
    click.echo("")
    if backend_name == backends.MLX:
        click.echo("Training via MLX LoRA (local, Apple Silicon)...")
    else:
        click.echo(f"Training via PyTorch LoRA (transformers + peft, device: {device})...")
    click.echo(f"  Base model:  {cfg.model}")
    click.echo(f"  Method:      LoRA (rank {cfg.rank}, {cfg.num_layers} layers, lr {cfg.learning_rate:g})")
    replay_count = prepared.record["prep"]["replay"]["requested"]
    click.echo(f"  Examples:    {sizes['train']:,}" + (f" + {replay_count:,} replay" if replay_count else ""))
    click.echo(f"  Steps:       {cfg.iters:,} ({prepared.epochs:g} epochs, batch {cfg.batch_size})")

    if dry_run:
        click.echo("\nDry run: not training. To train, run:\n  " + " ".join(prepared.command))
        return
    click.echo(f"\nProgress: trainjudge status {prepared.run_dir}")

    with StatusTracker(prepared.run_dir, "train", notify_on_finish=notify) as tracker:
        if replay_count:
            click.echo(f"\nGenerating {replay_count:,} replay examples with the base model...")
            tracker.stage("replay", total=replay_count)

            def on_replay(done: int, total: int) -> None:
                click.echo(f"  {done:,}/{total:,}")
                tracker.progress(done, total)

            added = training.add_replay(prepared, on_progress=on_replay)
            click.echo(f"  Added {added:,} replay examples to the training split.\n")

        tracker.stage("train", total=cfg.iters)

        def on_event(event: dict) -> None:
            if event["type"] == "train":
                remaining = (cfg.iters - event["iter"]) / event["it_per_sec"]
                click.echo(
                    f"  step {event['iter']:>5,}/{cfg.iters:,} · train loss "
                    f"{event['loss']:.3f} · {event['it_per_sec']:.2f} it/s · "
                    f"ETA {_duration(remaining)}"
                )
                tracker.progress(event["iter"], eta_s=remaining, message=f"train loss {event['loss']:.3f}")
            else:
                click.echo(f"  step {event['iter']:>5,}/{cfg.iters:,} · val loss {event['loss']:.3f}")
                tracker.progress(event["iter"], message=f"val loss {event['loss']:.3f}")

        try:
            summary = training.run_training(prepared, on_event)
        except TrainingFailed as e:
            click.echo(e.log_tail, err=True)
            raise click.ClickException(f"training failed: {e}. Full log: {e.log_path}") from e
        tracker.finish(
            f"trained in {_duration(summary.duration_s)}; next: trainjudge verify {prepared.run_dir}",
            result="trained",
        )

    click.echo("")
    click.echo(f"Training finished in {_duration(summary.duration_s)}.")
    if summary.train_loss_drop_pct is not None:
        click.echo(
            f"  Train loss {summary.first_train_loss:.3f} → {summary.final_train_loss:.3f} "
            f"({-summary.train_loss_drop_pct:+.1f}%)"
        )
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
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False),
    help="SQL tasks only: SQLite database (.sqlite/.db) or SQL script (.sql) to run queries "
    "against. Defaults to the one recorded by a previous eval of this run.",
)
@click.option(
    "--task",
    type=click.Choice(["auto", "sql", "json"]),
    default="auto",
    show_default=True,
    help="What to score: SQL execution accuracy or JSON exact match (auto: detect from the test split).",
)
@click.option(
    "--target",
    type=click.Choice(["baseline", "finetuned", "both"]),
    default="both",
    show_default=True,
)
@click.option("--limit", type=int, help="Only evaluate the first N test examples.")
@click.option("--max-tokens", type=int, default=512, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--system-prompt", help="System prompt for both targets (default: none).")
@click.option("--json", "as_json", is_flag=True, help="Print results as JSON.")
@click.option("--notify", is_flag=True, help=NOTIFY_HELP)
def eval_command(
    run_dir: str,
    db_path: str | None,
    task: str,
    target: str,
    limit: int | None,
    max_tokens: int,
    batch_size: int,
    system_prompt: str | None,
    as_json: bool,
    notify: bool,
) -> None:
    """Score the base model and the fine-tuned adapter on the held-out test split."""
    run = Path(run_dir)
    try:
        record = runs.read_run_json(run)
    except FileNotFoundError as e:
        raise click.ClickException(f"{run} has no run.json; is it a trainjudge run?") from e
    try:
        task, db = evaluation.resolve_task(run, task, Path(db_path) if db_path else None)
    except runs.RunError as e:
        raise click.UsageError(str(e)) from e
    if reason := backends.unavailable_reason(backends.backend_of(record)):
        raise click.ClickException(reason)

    decoding = DecodingConfig(max_tokens=max_tokens, batch_size=batch_size, system_prompt=system_prompt)
    targets = evaluation.TARGETS if target == "both" else (target,)
    labels = {"baseline": "Baseline (no fine-tuning)", "finetuned": "Fine-tuned"}
    log = (lambda *a, **k: None) if as_json else click.echo
    rows = len(evaluation.test_rows(run, limit))
    log(f"{tasks.TASKS[task].label} on {rows:,} held-out test examples")
    log(f"  Database: {db}\n" if db else "")

    results = {}
    with StatusTracker(run, "eval", notify_on_finish=notify) as tracker:
        for t in targets:
            log(f"{labels[t]}: {record['model']}" + (" + LoRA adapter" if t == "finetuned" else ""))
            tracker.stage(f"eval:{t}", total=rows)

            def on_progress(done: int, total: int) -> None:
                log(f"  generated {done:,}/{total:,}")
                tracker.progress(done, total)

            try:
                result = evaluation.evaluate_target(run, t, db, decoding, limit, on_progress, task=task)
            except runs.RunError as e:
                raise click.ClickException(str(e)) from e
            results[t] = result
            log(f"  {_accuracy_line(result)}\n")
        tracker.finish(" · ".join(f"{t} {r['accuracy']:.1%}" for t, r in results.items()), result="evaluated")

    if as_json:
        click.echo(
            json.dumps(
                {t: {k: v for k, v in r.items() if k != "examples"} for t, r in results.items()},
                indent=2,
            )
        )
        return
    if len(results) == 2:
        delta = (results["finetuned"]["accuracy"] - results["baseline"]["accuracy"]) * 100
        click.echo(f"Change: {delta:+.1f} points. Run `trainjudge verify {run}` for the verdict.")
    click.echo(f"Per-example results: {run / 'eval'}/")


def _accuracy_line(result: dict) -> str:
    outcomes = result["outcomes"]
    correct = outcomes["correct"]
    issues = [f"{n:,} {name.replace('_', ' ')}" for name, n in outcomes.items() if n and name != "correct"]
    secondary = result.get("secondary") or {"label": "lenient", "value": result.get("lenient_accuracy", 0)}
    line = (
        f"Accuracy {result['accuracy']:.1%} ({correct:,}/{result['scored']:,}) · "
        f"{secondary['label'].lower()} {secondary['value']:.1%}"
    )
    return line + (f" · {' · '.join(issues)}" if issues else "")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False),
    help="SQL tasks only: the database queries run against (defaults to the one recorded for this run).",
)
@click.option(
    "--task",
    type=click.Choice(["auto", "sql", "json"]),
    default="auto",
    show_default=True,
    help="What to score: SQL execution accuracy or JSON exact match (auto: detect from the test split).",
)
@click.option(
    "--min-improvement",
    type=float,
    default=verdict.DEFAULT_MIN_IMPROVEMENT,
    show_default=True,
    help="Accuracy points needed to count as improved.",
)
@click.option(
    "--regression-tolerance",
    type=float,
    default=verdict.DEFAULT_REGRESSION_TOLERANCE,
    show_default=True,
    help="Pass-rate points a regression category may drop before it's flagged.",
)
@click.option("--rerun", is_flag=True, help="Re-run all evals instead of reusing saved ones.")
@click.option("--strict", is_flag=True, help="Exit with status 1 unless the verdict is IMPROVED.")
@click.option("--notify", is_flag=True, help=NOTIFY_HELP)
def verify(
    run_dir: str,
    db_path: str | None,
    task: str,
    min_improvement: float,
    regression_tolerance: float,
    rerun: bool,
    strict: bool,
    notify: bool,
) -> None:
    """Compare baseline vs fine-tuned on the task metric and issue a verdict."""
    run = Path(run_dir)
    if not (run / "run.json").exists():
        raise click.ClickException(f"{run} has no run.json; is it a trainjudge run?")
    click.echo(f"Progress: trainjudge status {run}\n")
    with StatusTracker(run, "verify", notify_on_finish=notify) as tracker:
        try:
            result = verification.verify_run(
                run,
                Path(db_path) if db_path else None,
                min_improvement,
                regression_tolerance,
                rerun,
                log=click.echo,
                tracker=tracker,
                task=task,
            )
        except runs.RunError as e:
            raise click.ClickException(str(e)) from e
        except ImportError as e:
            raise click.ClickException(f"evals need mlx-lm: {e}") from e
        v = result.verdict
        tracker.finish(f"{v.outcome}: {'; '.join(v.reasons)}", result=v.outcome)

    click.echo("")
    click.echo(verdict.format_box(result.verdict))
    click.echo(f"\nArtifacts written to {run}/")
    click.echo("  " + " · ".join(p.name for p in result.artifacts))
    if strict and result.verdict.outcome != verdict.IMPROVED:
        raise SystemExit(1)


@main.command("status")
@click.argument("run_dir", required=False, type=click.Path(file_okay=False))
@click.option(
    "--runs-dir",
    type=click.Path(file_okay=False),
    default=str(runs.DEFAULT_RUNS_DIR),
    show_default=True,
)
@click.option("--all", "show_all", is_flag=True, help="List every run instead of the latest one.")
@click.option("--watch", is_flag=True, help="Keep printing progress until the job finishes.")
@click.option(
    "--milestones",
    is_flag=True,
    help="With --watch, print one line per milestone only (stage start, 25/50/75%, stage done, "
    "finish or failure). Made for agents: each line can become a chat notification.",
)
@click.option(
    "--interval", type=float, default=5.0, show_default=True, help="Seconds between --watch updates."
)
@click.option("--json", "as_json", is_flag=True, help="Print the status as JSON (for agents).")
def status_command(
    run_dir: str | None,
    runs_dir: str,
    show_all: bool,
    watch: bool,
    milestones: bool,
    interval: float,
    as_json: bool,
) -> None:
    """Show what a train/eval/verify job is doing: stage, progress, ETA, finished or not."""
    import time

    from trainjudge import status as st

    if show_all:
        found = st.find_runs(Path(runs_dir))
        if as_json:
            click.echo(json.dumps([{**s, "state": st.effective_state(s)} for _, s in found], indent=2))
        elif not found:
            click.echo(f"No runs with status in {runs_dir}/.")
        else:
            for path, s in found:
                click.echo(st.one_line(path, s))
        return

    if run_dir:
        path = Path(run_dir)
    else:
        found = st.find_runs(Path(runs_dir))
        if not found:
            raise click.ClickException(f"No runs with status in {runs_dir}/.")
        path = found[0][0]

    if milestones:
        # A job that was just launched may not have written its status yet.
        deadline = time.monotonic() + 60
        while st.read_status(path) is None and time.monotonic() < deadline:
            time.sleep(1)
    status = st.read_status(path)
    if status is None:
        raise click.ClickException(
            f"{path} has no {st.STATUS_FILE} (no train/eval/verify has run there yet)."
        )
    if as_json:
        click.echo(json.dumps({**status, "state": st.effective_state(status)}, indent=2))
        return
    if milestones:
        raise SystemExit(_watch_milestones(path, status, interval))
    click.echo(st.format_status(path, status))
    if not watch:
        return

    last = None
    while st.effective_state(status) == st.RUNNING:
        line = st.progress_text(status)
        if line != last:
            click.echo(f"  … {line}")
            last = line
        time.sleep(interval)
        status = st.read_status(path) or status
    click.echo("")
    click.echo(st.format_status(path, status))
    if st.effective_state(status) != st.DONE:
        raise SystemExit(1)


def _watch_milestones(path: Path, status: dict, interval: float) -> int:
    """Print one flushed line per milestone until the job ends; return the exit code."""
    import time

    from trainjudge import status as st

    def say(text: str) -> None:
        click.echo(f"[trainjudge {status['command']}] {text}")

    announced: set[tuple] = set()
    seen_stages = 0
    while True:
        for s in status.get("stages", [])[seen_stages:]:
            say(f"✓ {s['label']} done ({st._duration(s['duration_s'])})")
        seen_stages = len(status.get("stages", []))

        state = st.effective_state(status)
        if state != st.RUNNING:
            mark = {st.DONE: "✓ finished", st.FAILED: "✗ failed", st.INTERRUPTED: "■ interrupted"}
            say(f"{mark.get(state, '✗ stopped (process gone)')}: {status.get('message') or ''}".rstrip(": "))
            return 0 if state == st.DONE else 1

        stage, step, total = status.get("stage"), status.get("step") or 0, status.get("total")
        if stage and (stage, "start") not in announced:
            announced.add((stage, "start"))
            say(f"▶ {status['stage_label']} started" + (f" ({total:,} items)" if total else ""))
        if stage and total:
            quarter = min(3, int(step / total * 4))
            if quarter and (stage, quarter) not in announced:
                announced.update((stage, q) for q in range(1, quarter + 1))
                say(f"… {st.progress_text(status)}")
        time.sleep(interval)
        status = st.read_status(path) or status


if __name__ == "__main__":
    main()
