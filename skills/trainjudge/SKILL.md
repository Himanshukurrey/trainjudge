---
name: trainjudge
description: Decide whether fine-tuning is the right fix before training, and verify whether a fine-tune actually worked afterwards, using held-out task metrics rather than training loss. Use this whenever the user asks to fine-tune, LoRA-train or distill a model on a dataset, asks whether they should fine-tune, or asks whether a fine-tuned model or adapter is better than the base model.
allowed-tools: Bash(trainjudge *)
---

# TrainJudge: diagnose before training, verify after

A lower training loss doesn't prove a model got better at the task. It can
memorize its training data and still fail on new examples, or get better at
the task while breaking everything else. Many fine-tuning requests shouldn't
be fine-tuned at all: facts belong in retrieval, and unclear instructions
belong in the prompt. TrainJudge checks both ends.

Check `trainjudge` is installed before using this skill: `trainjudge --version`.
If it isn't, tell the user (`pip install "trainjudge[mlx] @ git+https://github.com/Himanshukurrey/trainjudge"`)
and proceed without it rather than blocking on it. Training and evals need an
Apple Silicon Mac; `diagnose` and `audit` run anywhere.

## Workflow

**1. Diagnose first, before any training:**

```
trainjudge diagnose --dataset <path.jsonl> --model <model> --goal "<the user's goal, in their words>"
```

Show the user the classification, the evidence and any BFSI / sensitive-data
warnings. Then:

- **KNOWLEDGE GAP** or **PROMPT-ENGINEERING GAP**: tell the user TrainJudge
  recommends *not* fine-tuning and why. Only continue if the user explicitly
  says to fine-tune anyway.
- **FORMAT/BEHAVIOR GAP** or **COST/LATENCY GAP**: fine-tuning is a reasonable
  fit. Ask before starting training, because it takes minutes to hours.
- **UNCLEAR** or low confidence: ask the user what should change in the
  model's output before going further.

Pass `--tried-prompting` / `--not-tried-prompting` if the user has said whether
they've tried few-shot prompting. Use `--json` if you want every signal and
score; you may disagree with the classification, but say so and explain why
rather than silently overriding it.

**2. If the dataset has sensitive data** (card numbers, Aadhaar, PAN, account
numbers, UPI IDs, phones, emails), `train` refuses to run. Tell the user and
suggest masking. Never pass `--allow-sensitive-data` unless the user explicitly
asks for it.

**3. Train** (after the user confirms):

```
trainjudge train --dataset <path.jsonl> --model <model> --goal "<goal>"
```

It drops duplicate, malformed and low-quality rows, holds out a test split and
prints the run directory. For narrow tasks where general skills matter, add
`--replay 200` to mix in the base model's own answers to general prompts,
which limits regressions. `--dry-run` prepares everything without training.

**4. Verify before saying the model improved:**

```
trainjudge verify <run-dir> --db <database>
```

For SQL tasks, `--db` is the SQLite database (or `.sql` script) the queries
should run against. It's only needed the first time.

**5. Report the verdict honestly. Don't override it with the loss curve:**

- **IMPROVED**: the task metric rose significantly on held-out data and no
  general capability regressed. Safe to tell the user it worked.
- **REGRESSED**: the task improved but general capability got worse. Tell the
  user it isn't ready to deploy, and which categories broke
  (`EXPERIMENT_REPORT.md` lists examples). Fewer steps, a lower learning rate
  or `--replay` usually help.
- **REJECTED**: no meaningful improvement on the task. Say so plainly, even if
  training loss dropped a lot.

Point the user to `EXPERIMENT_REPORT.md` and `MODEL_CARD.md` in the run folder.

## Keep the user informed during long jobs

`train` takes minutes to hours and `verify` several minutes. Don't go silent:

- Run `train` and `verify` as background commands, not blocking calls.
- Before starting, tell the user what's about to run and roughly how long it
  will take. Both commands print a `Progress: trainjudge status <run>` line
  near the start.
- While it runs, check `trainjudge status <run-dir> --json` every minute or
  two. It reports `state` (`running`, `done`, `failed`, `interrupted`, or
  `stopped` if the process died without finishing), the current `stage_label`,
  `step`/`total`, `eta_s` and the finished `stages`. Tell the user when each
  stage finishes and what's next, with the ETA. For example: "Training done
  (7m 41s). Now running the baseline eval, about 4 minutes left."
- When `state` is `done`, report the result. If it's `failed` or `stopped`,
  say so right away with the `message`, and check `<run>/logs/mlx.log` for
  training failures.
- `trainjudge status --all` lists every run, if the user asks what's running.

Suggest `--notify` if the user wants a desktop notification when a job
finishes, and `trainjudge status --watch` if they want to follow it in their
own terminal.

## Limitations to keep in mind

The diagnosis is a transparent heuristic, not a guarantee, and mixed goals
(part knowledge, part format) are common. The task eval in v0.1 covers
text-to-SQL (execution accuracy). The regression suite is small (60 prompts)
and catches broken instruction-following and formatting, not subtle capability
loss. Run folders contain examples from the test split, so treat them as
being as sensitive as the dataset.
