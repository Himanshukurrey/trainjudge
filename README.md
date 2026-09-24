# TrainJudge

Before you fine-tune, TrainJudge tells you whether fine-tuning is even the right move.
After you fine-tune, it tells you whether it actually worked — on the task metric, not training loss.

> Status: pre-alpha (v0.1 in progress). Some commands are not implemented yet.

## Install (dev)

```bash
pip install -e ".[dev,mlx]"   # mlx extra is Apple Silicon only
```

## Commands

```
trainjudge diagnose --dataset <path> --model <name> --goal "<text>"
trainjudge audit <path>
trainjudge train --dataset <path> --model <name> --method lora
trainjudge verify <run-dir>
```

## Dataset audit

`trainjudge audit` reads JSONL in any of mlx-lm's formats (`prompt`/`completion`,
`messages`, or `text`) and gives each row exactly one status:

- **malformed:** invalid JSON or UTF-8, or missing, empty or non-string fields
- **duplicate:** same prompt and completion as an earlier row, ignoring case, whitespace
  and trailing punctuation
- **low-quality:** refusals, placeholders (`TODO`, `N/A`), completions that repeat the
  prompt, and degenerate repetition
- **clean:** everything else

It also warns when the same prompt has conflicting completions.

The audit only flags rows. `--write-clean <path>` saves a copy without duplicate and
malformed rows; low-quality rows stay in the copy unless you add `--drop-low-quality`,
because the heuristics can flag legitimate rows (for example, intended refusals in
safety data).

```
$ trainjudge audit demo/sql_generation/data.jsonl
TRAINJUDGE DATASET AUDIT

Dataset: demo/sql_generation/data.jsonl
Format:  prompt/completion
  1,830 examples
  71% clean · 18% duplicates · 7% low-quality · 4% malformed
...
```

## Known limitations

- The diagnosis step is a heuristic classifier, not a guarantee. It can misclassify
  mixed-goal tasks (partly knowledge, partly format).
- The audit's low-quality checks are heuristics too. They can miss subtly wrong answers
  and can flag legitimate ones.

## License

MIT
