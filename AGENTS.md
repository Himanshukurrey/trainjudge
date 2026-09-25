# AGENTS.md

Instructions for coding agents (Codex and others that read `AGENTS.md`).

## Using TrainJudge

When the user asks to fine-tune, LoRA-train or distill a model, or asks whether
a fine-tuned model is better than its base model, use the `trainjudge` CLI.
Install it with `pip install "trainjudge[mlx] @ git+https://github.com/Himanshukurrey/trainjudge"`.
Training and evals need an Apple Silicon Mac; `diagnose` and `audit` run anywhere.

1. **Diagnose before training.**
   `trainjudge diagnose --dataset <path> --model <model> --goal "<goal>"`.
   If the result is a KNOWLEDGE GAP (use retrieval) or a PROMPT-ENGINEERING GAP
   (fix the prompt), tell the user TrainJudge recommends not fine-tuning and
   only continue if they explicitly ask to. For a FORMAT/BEHAVIOR or
   COST/LATENCY gap, ask before training, because it takes a while.
2. **Respect sensitive-data refusals.** If `train` refuses because the audit
   found card numbers, Aadhaar, PAN or similar identifiers, tell the user and
   offer `--mask-sensitive`, which replaces them with placeholders like `[EMAIL]`
   and trains on the masked rows. Never add `--allow-sensitive-data` unless the
   user asks for it.
3. **Train.** `trainjudge train --dataset <path> --model <model> --goal "<goal>"`.
   Add `--replay 200` when general skills matter.
4. **Verify.** `trainjudge verify <run-dir>` (add `--db <database>` for SQL tasks).
5. **Report the verdict as-is.** IMPROVED means it worked. REGRESSED means the
   task improved but general capability broke, so don't call it ready to
   deploy. REJECTED means it didn't really improve, whatever the loss did.
   Never claim improvement from training loss alone.

**Long jobs: keep the user informed.** `train` and `verify` take minutes, and
command output often isn't shown live. Run them in the background, then run
`trainjudge status <run-dir> --watch --milestones`, which prints one line per
milestone (stage started, 25/50/75%, stage done, finished or failed) and exits
when the job ends. Relay each milestone to the user with what's next and the
ETA, and report failures right away. Never go quiet until the result is in.

Full details: [skills/trainjudge/SKILL.md](skills/trainjudge/SKILL.md).

## Working on this repository

- Setup: `pip install -e ".[dev]"` (add `mlx` on an Apple Silicon Mac for real training).
- Before finishing a change, run `pytest`, `ruff check .` and `ruff format --check .`.
- Tests must not download models or need `mlx-lm`. Use the fake backends in
  `tests/test_training.py` and `tests/test_verify.py`.
- `demo/*/data.jsonl`, `demo/*/docs/` and `demo/sql_generation/shop.sql` are
  generated. Edit the matching `generate.py`, rerun it and commit both.
- Always pass `encoding="utf-8"` when reading or writing text files; CI runs on Windows.
