"""EXPERIMENT_REPORT.md and MODEL_CARD.md for a verified run."""

from __future__ import annotations

import json
import shlex

from trainjudge import tasks
from trainjudge.regression_check import CATEGORY_LABELS
from trainjudge.verdict import IMPROVED, REGRESSED, REJECTED, Verdict, format_p

VERDICT_BADGES = {IMPROVED: "✓ IMPROVED", REGRESSED: "⚠ REGRESSED", REJECTED: "✗ REJECTED"}
MAX_EXAMPLES = 5


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _regression_table(v: Verdict) -> list[str]:
    lines = ["| Category | Baseline | Fine-tuned | Change | |", "|---|---|---|---|---|"]
    for r in v.regressions:
        mark = "⚠ regressed" if r.regressed else "✓"
        lines.append(
            f"| {CATEGORY_LABELS[r.category]} ({r.total}) | {r.baseline:.0%} | "
            f"{r.finetuned:.0%} | {r.delta_points:+.1f} pts | {mark} |"
        )
    return lines


def _question(prompt: str) -> str:
    """The part of the prompt that varies: after "Question:" or "Input:", if present."""
    for marker in ("Question: ", "Input: "):
        if marker in prompt:
            return prompt.split(marker, 1)[-1].strip()
    return prompt.strip()


def _backend_label(backend: dict) -> str:
    if backend.get("name", "mlx") == "mlx":
        return f"mlx-lm {backend.get('mlx_lm_version') or '?'} (Apple Silicon)"
    libs = ", ".join(
        f"{k.removesuffix('_version')} {v}" for k, v in backend.items() if k.endswith("_version") and v
    )
    if backend.get("device_name"):
        where = f"{backend['device_name']} ({backend.get('device_used', '?')}, {backend.get('dtype', '?')})"
    else:
        where = backend.get("device", "auto")
    return f"PyTorch ({libs}) on {where}"


def reproduce_command(run: dict) -> str:
    """The `trainjudge train` command that recreates this run's data prep and training."""
    cfg, prep, backend = run["config"], run["prep"], run.get("backend") or {}
    parts = [
        "trainjudge train",
        f"--dataset {shlex.quote(run['dataset']['path'])}",
        f"--model {shlex.quote(run['model'])}",
        f"--backend {backend.get('name', 'mlx')}",
    ]
    if backend.get("name") == "torch" and backend.get("device", "auto") != "auto":
        parts.append(f"--device {backend['device']}")
    parts += [
        f"--iters {cfg['iters']}",
        f"--batch-size {cfg['batch_size']}",
        f"--learning-rate {cfg['learning_rate']:g}",
        f"--rank {cfg['rank']}",
        f"--num-layers {cfg['num_layers']}",
        f"--seed {cfg['seed']}",
    ]
    if cfg.get("max_seq_length", 2048) != 2048:
        parts.append(f"--max-seq-length {cfg['max_seq_length']}")
    for key, flag in (("valid_fraction", "--valid-fraction"), ("test_fraction", "--test-fraction")):
        if prep.get(key, 0.1) != 0.1:
            parts.append(f"{flag} {prep[key]:g}")
    if (prep.get("replay") or {}).get("requested"):
        parts.append(f"--replay {prep['replay']['requested']}")
    # Older runs predate the explicit flag; masked counts imply masking was on.
    if prep.get("mask_sensitive", bool(prep.get("masked"))):
        parts.append("--mask-sensitive")
    if prep.get("allow_sensitive_data"):
        parts.append("--allow-sensitive-data")
    if prep.get("keep_low_quality", "low_quality" not in prep.get("dropped", {"low_quality": 0})):
        parts.append("--keep-low-quality")
    if run.get("goal"):
        parts.append(f"--goal {shlex.quote(run['goal'])}")
    return " \\\n    ".join(parts)


def _task_of(evaluation: dict) -> tasks.TaskInfo:
    return tasks.TASKS[evaluation.get("task", tasks.SQL)]


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False) if value is not None else "_no JSON_"


def _json_answer(e: dict) -> str:
    text = f"`{_compact(e['predicted'])}`" if e["predicted"] is not None else "_no JSON_"
    problems = [f"wrong {k}" for k in e.get("wrong_fields", [])] + [
        f"missing {k}" for k in e.get("missing_fields", [])
    ]
    return text + (f" ({', '.join(problems)})" if problems else "")


def _short(text: str | None, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _label_answer(e: dict) -> str:
    return f"`{e['predicted']}`" if e.get("predicted") else "_no answer_"


def _custom_answer(e: dict) -> str:
    text = f"`{_short(e['output'])}`" if e["output"].strip() else "_empty_"
    return text + (f" ({_short(e['reason'], 120)})" if e.get("reason") else "")


def _flipped(baseline_eval: dict, finetuned_eval: dict) -> tuple[list, list]:
    base = {e["prompt"]: e for e in baseline_eval["examples"]}
    gained, lost = [], []
    for tuned in finetuned_eval["examples"]:
        b = base[tuned["prompt"]]
        if tuned["outcome"] == "correct" and b["outcome"] != "correct":
            gained.append((b, tuned))
        elif b["outcome"] == "correct" and tuned["outcome"] != "correct":
            lost.append((b, tuned))
    return gained, lost


def _example_block(pairs: list, title: str, task: str = tasks.SQL) -> list[str]:
    if not pairs:
        return []
    lines = [f"### {title} ({len(pairs)})", ""]
    for base, tuned in pairs[:MAX_EXAMPLES]:
        if task == tasks.SQL:
            gold = f"`{base['gold_sql']}`"
            answers = [f"`{' '.join(e['sql'].split())}`" if e["sql"] else "_no SQL_" for e in (base, tuned)]
        elif task == tasks.LABEL:
            gold = f"`{base['gold']}`"
            answers = [_label_answer(e) for e in (base, tuned)]
        elif task == tasks.CUSTOM:
            gold = f"`{_short(base['gold'])}`"
            answers = [_custom_answer(e) for e in (base, tuned)]
        else:
            gold = f"`{_compact(base['gold'])}`"
            answers = [_json_answer(e) for e in (base, tuned)]
        lines += [
            f"**Input:** {_question(base['prompt'])}",
            "",
            f"- Gold: {gold}",
            f"- Baseline ({base['outcome'].replace('_', ' ')}): {answers[0]}",
            f"- Fine-tuned ({tuned['outcome'].replace('_', ' ')}): {answers[1]}",
            "",
        ]
    if len(pairs) > MAX_EXAMPLES:
        lines += [f"_…and {len(pairs) - MAX_EXAMPLES} more in `eval/*.json`._", ""]
    return lines


def _regression_examples(baseline_reg: dict, finetuned_reg: dict) -> list[str]:
    base = {i["id"]: i for i in baseline_reg["items"]}
    broken = [i for i in finetuned_reg["items"] if base[i["id"]]["passed"] and not i["passed"]]
    if not broken:
        return []
    lines = [f"### General-capability items that broke ({len(broken)})", ""]
    for item in broken[:MAX_EXAMPLES]:
        output = " ".join(item["output"].split())
        output = output[:200] + ("…" if len(output) > 200 else "")
        lines += [f"- **{item['prompt']}**", f"  - Fine-tuned: `{output}`"]
    if len(broken) > MAX_EXAMPLES:
        lines.append(f"- _…and {len(broken) - MAX_EXAMPLES} more in `eval/regression_finetuned.json`._")
    return lines + [""]


def _verify_command(evaluation: dict) -> str:
    command = "trainjudge verify <run-dir>"
    if evaluation.get("database"):
        command += f" --db {shlex.quote(evaluation['database']['path'])}"
    if evaluation.get("scorer"):
        command += f" --scorer {shlex.quote(evaluation['scorer']['spec'])}"
    return command


def experiment_report(
    run: dict,
    v: Verdict,
    baseline_eval: dict,
    finetuned_eval: dict,
    baseline_reg: dict,
    finetuned_reg: dict,
    run_dir: str,
) -> str:
    cfg = run["config"]
    prep = run["prep"]
    training = run.get("training") or {}
    dataset = run["dataset"]
    gained, lost = _flipped(baseline_eval, finetuned_eval)
    task = _task_of(finetuned_eval)
    lines = [
        f"# Experiment report: {run_dir.rstrip('/').split('/')[-1]}",
        "",
        f"**Verdict: {VERDICT_BADGES[v.outcome]}.** {'; '.join(v.reasons).capitalize()}.",
        "",
    ]
    if run.get("goal"):
        lines += [f"**Goal:** {run['goal']}", ""]
    lines += [
        f"## Task metric: {task.label} (held-out test split)",
        "",
        "| | Baseline | Fine-tuned | Change |",
        "|---|---|---|---|",
        (
            f"| {task.short_label} | {_pct(v.baseline)} | {_pct(v.finetuned)} | "
            f"{v.improvement_points:+.1f} pts |"
        ),
        f"| {task.secondary_label} | {_pct(v.baseline_lenient)} | "
        f"{_pct(v.finetuned_lenient)} | "
        + (
            f"{(v.finetuned_lenient - v.baseline_lenient) * 100:+.1f} pts |"
            if v.baseline_lenient is not None and v.finetuned_lenient is not None
            else "n/a |"
        ),
        "",
        (
            f"{v.scored} held-out examples. {v.gained} went from wrong to right and {v.lost} from "
            f"right to wrong (McNemar exact test, {format_p(v.p_value)}). The bar for improvement is "
            f"{v.min_improvement:g} points and p < 0.05."
        ),
        "",
        "| Outcome | Baseline | Fine-tuned |",
        "|---|---|---|",
    ]
    for outcome, n in baseline_eval["outcomes"].items():
        m = finetuned_eval["outcomes"].get(outcome, 0)
        if n or m:
            lines.append(f"| {outcome.replace('_', ' ')} | {n} | {m} |")
    if finetuned_eval.get("per_field"):
        base_fields = baseline_eval.get("per_field", {})
        lines += ["", "| Field | Baseline | Fine-tuned |", "|---|---|---|"]
        for name, value in finetuned_eval["per_field"].items():
            lines.append(f"| `{name}` | {_pct(base_fields.get(name))} | {_pct(value)} |")
    if finetuned_eval.get("per_label"):
        base_labels = baseline_eval.get("per_label", {})
        lines += ["", "| Label | Test rows | Baseline F1 | Fine-tuned F1 |", "|---|---|---|---|"]
        for name, stats in finetuned_eval["per_label"].items():
            base_f1 = (base_labels.get(name) or {}).get("f1")
            lines.append(f"| `{name}` | {stats['support']} | {_pct(base_f1)} | {_pct(stats['f1'])} |")
        if finetuned_eval.get("confusions"):
            lines += ["", "Most common fine-tuned mistakes (gold → predicted): " + ", ".join(
                f"`{c['gold']}` → `{c['predicted']}` ×{c['count']}" for c in finetuned_eval["confusions"][:5]
            ) + "."]  # fmt: skip
    lines += [
        "",
        "## Regression check (held-out general tasks)",
        "",
        *_regression_table(v),
        "",
        (
            "A category counts as regressed when its pass rate drops by more than "
            f"{v.regressions[0].tolerance:g} points and it loses at least 2 items net. "
            "The suite is small (60 prompts), so "
            "treat changes of a few points as noise."
        ),
        "",
    ]
    lines += [
        "## Examples",
        "",
        *_example_block(gained, "Fixed by fine-tuning", task.name),
        *_example_block(lost, "Broken by fine-tuning", task.name),
        *_regression_examples(baseline_reg, finetuned_reg),
    ]

    drop = training.get("train_loss_drop_pct")
    lines += [
        "## Training",
        "",
        "| | |",
        "|---|---|",
        f"| Base model | `{run['model']}` |",
        f"| Method | LoRA, rank {cfg['rank']}, {cfg['num_layers']} layers, scale {cfg['scale']:g} |",
        (
            f"| Steps | {cfg['iters']} ({cfg['epochs']:g} epochs, batch {cfg['batch_size']}, "
            f"lr {cfg['learning_rate']:g}) |"
        ),
        f"| Backend | {_backend_label(run.get('backend') or {})} |",
        f"| Duration | {training.get('duration_s', 0) / 60:.1f} min |",
        f"| Train loss | {training.get('first_train_loss')} → {training.get('final_train_loss')}"
        + (f" ({-drop:+.1f}%)" if drop is not None else "")
        + " |",
        (
            f"| Validation loss | {training.get('first_val_loss')} → "
            f"{training.get('final_val_loss')} (best {training.get('best_val_loss')}) |"
        ),
        "",
        "## Data",
        "",
        (f"- Dataset: `{dataset['path']}` ({dataset['rows']:,} rows, sha256 `{dataset['sha256'][:12]}…`)"),
        "- Dropped before training: "
        + ", ".join(f"{n:,} {k.replace('_', '-')}" for k, n in prep["dropped"].items()),
        (
            f"- Splits: {prep['splits']['train']:,} train · {prep['splits']['valid']:,} valid · "
            f"{prep['splits']['test']:,} test ({prep['split_method']}, seed {prep['seed']})"
        ),
        *(
            [
                f"- Database: `{baseline_eval['database']['path']}` (sha256 "
                f"`{baseline_eval['database']['sha256'][:12]}…`)"
            ]
            if baseline_eval.get("database")
            else []
        ),
        *(
            [
                f"- Scorer: `{baseline_eval['scorer']['spec']}` (sha256 "
                f"`{(baseline_eval['scorer']['sha256'] or '')[:12]}…`)"
            ]
            if baseline_eval.get("scorer")
            else []
        ),
        "",
        "## Reproduce",
        "",
        "```bash",
        reproduce_command(run),
        _verify_command(baseline_eval),
        "```",
        "",
        "## Caveats",
        "",
        f"- {task.caveat}",
        "- The regression suite is a small heuristic check, not a full benchmark.",
        (
            "- Both models use greedy decoding with the same prompt, so results are deterministic "
            "for this setup but may differ with sampling or other prompts."
        ),
        "",
    ]
    return "\n".join(lines)


def model_card(run: dict, v: Verdict, run_dir: str, task_name: str = tasks.SQL) -> str:
    task = tasks.TASKS[task_name]
    cfg = run["config"]
    prep = run["prep"]
    torch_run = (run.get("backend") or {}).get("name") == "torch"
    status = {
        IMPROVED: "Verified: improved on held-out data with no regressions.",
        REGRESSED: "⚠ Not recommended for deployment: the task improved, but general capability regressed.",
        REJECTED: "✗ Not recommended for deployment: no meaningful improvement on the task.",
    }[v.outcome]
    lines = [
        "---",
        f"base_model: {run['model']}",
        f"library_name: {'peft' if torch_run else 'mlx'}",
        "tags:",
        "  - lora",
        f"  - {'peft' if torch_run else 'mlx'}",
        f"  - {task.card_task_type}",
        "  - trainjudge",
        "model-index:",
        f"  - name: {run_dir.rstrip('/').split('/')[-1]}",
        "    results:",
        "      - task:",
        f"          type: {task.card_task_type}",
        "        metrics:",
        f"          - type: {task.card_metric_type}",
        f"            value: {v.finetuned:.4f}",
        f"            name: {task.label} (held-out)",
        "---",
        "",
        f"# LoRA adapter for {run['model']}",
        "",
        f"**TrainJudge verdict: {VERDICT_BADGES[v.outcome]}.** {status}",
        "",
        "## Intended use",
        "",
        run.get("goal") or f"The task in the training data, measured by {task.label.lower()}.",
        "",
        "## Evaluation",
        "",
        "| Metric | Base model | This adapter |",
        "|---|---|---|",
        f"| {task.label} (n={v.scored}) | {_pct(v.baseline)} | {_pct(v.finetuned)} |",
    ]
    for r in v.regressions:
        lines.append(
            f"| {CATEGORY_LABELS[r.category]} (n={r.total}) | {r.baseline:.0%} | "
            f"{r.finetuned:.0%}{' ⚠' if r.regressed else ''} |"
        )
    lines += [
        "",
        "See `EXPERIMENT_REPORT.md` for the full comparison and examples.",
        "",
        "## Training",
        "",
        (
            f"LoRA (rank {cfg['rank']}, {cfg['num_layers']} layers) for {cfg['iters']} steps on "
            f"{prep['splits']['train']:,} examples from `{run['dataset']['path']}`, with the "
            "loss on completions only."
        ),
        "",
        "## Usage",
        "",
        "```python",
        *(
            [
                "from peft import PeftModel",
                "from transformers import AutoModelForCausalLM, AutoTokenizer",
                "",
                f'tokenizer = AutoTokenizer.from_pretrained("{run["model"]}")',
                f'model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{run["model"]}"), '
                f'"{run_dir}/adapters")',
            ]
            if torch_run
            else [
                "from mlx_lm import load, generate",
                "",
                f'model, tokenizer = load("{run["model"]}", adapter_path="{run_dir}/adapters")',
            ]
        ),
        "```",
        "",
        "## Limitations",
        "",
        (
            "- Trained and evaluated on one schema. Expect it to fail on other databases."
            if task.name == tasks.SQL
            else "- Trained and evaluated on one output schema and one data source."
        ),
        (
            "- Evaluated on a held-out split of the same dataset, which may share templates or "
            "phrasing with the training data."
        ),
        "",
    ]
    return "\n".join(lines)
