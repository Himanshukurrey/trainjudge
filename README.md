# TrainJudge

Before you fine-tune, TrainJudge tells you whether fine-tuning is even the right move.
After you fine-tune, it tells you whether it actually worked — on the task metric, not training loss.

> Status: pre-alpha (v0.1 in progress).

## Install (dev)

```bash
pip install -e ".[dev,mlx]"   # mlx extra is Apple Silicon only
```

## Commands

```
trainjudge diagnose --dataset <path> --model <name> --goal "<text>"
trainjudge audit <path>
trainjudge train --dataset <path> --model <name> --method lora
trainjudge eval <run-dir> --db <database>
trainjudge verify <run-dir>
```

## Diagnosis: should you fine-tune at all?

`trainjudge diagnose` runs before any training. It reads the goal and the dataset and
sorts the request into one of four gaps:

| Gap | What it means | Recommendation |
|---|---|---|
| Knowledge | The model lacks facts (policies, rates, product details) | ❌ Use retrieval, not fine-tuning |
| Format/behavior | The model needs to learn an output format or convention (SQL, JSON, labels, tone) | ✓ Fine-tune |
| Cost/latency | A big model already does the task; you want it cheaper or faster | ✓ Distill into a small model |
| Prompt engineering | Few-shot prompting or clear instructions haven't been tried yet | ❌ Fix the prompt first |

Signals come from the goal text and from the dataset itself: the output shape (SQL,
JSON, code, labels or prose), whether answers state numbers that aren't in the question,
how many prompts share the same answer, whether answers are grounded in context given in
the prompt, whether rows cite a source document, and the dataset size. Every point a
bucket earns is shown as evidence. When two buckets score close together, the output
says the goal is mixed.

```
$ trainjudge diagnose --dataset demo/policy_docs/data.jsonl --model Qwen3-0.6B \
    --goal "make it answer from our internal support policy documents"
TRAINJUDGE DIAGNOSIS

Goal: make it answer from our internal support policy documents
Model: Qwen3-0.6B
Dataset: 236 examples (question → prose answer pairs citing source documents)

Classification: KNOWLEDGE GAP   (confidence: high)
...
Evidence:
  • goal mentions "documents", "policy", "internal" +1 more
  • completions are free-form prose answers
  • 78% of answers state numbers, dates or amounts that aren't in the
    question
  • 216 prompts map to only 36 distinct answers, so the dataset teaches
    recall of fixed facts
  • only 23% of answer words appear in the prompt; the facts have to come
    from the model's weights
  • 100% of rows cite a source document

Recommendation: ❌ Do not fine-tune for this goal.
```

Pass `--tried-prompting` or `--not-tried-prompting` if you know, and `--json` for
machine-readable output. The JSON output includes every signal and score so a coding
agent can review the call.

## BFSI checks

For banking, financial services and insurance datasets, diagnosis adds:

- **Sensitive data:** card numbers (Luhn-checked), Aadhaar numbers (Verhoeff-checked),
  PANs, account numbers, UPI IDs, Indian mobile numbers and email addresses. Fine-tuned
  models can memorize and repeat training data, so mask or tokenize these first. The scan
  runs on every dataset, BFSI or not, and reports line numbers only, never the values.
- **Regulated facts:** goals about interest rates, charges, KYC rules or regulator
  circulars lean towards retrieval over versioned documents with effective dates.
- **Automated decisions:** goals like approving or rejecting loans or claims get a
  warning to keep a human in the loop and check outcomes for bias.

Two BFSI demos show both outcomes: [bfsi_transactions](demo/bfsi_transactions/)
(transaction categorization: fine-tune, but mask the PII first) and
[bfsi_loan_faq](demo/bfsi_loan_faq/) (rates and charges: don't fine-tune, use retrieval).

## Dataset audit

`trainjudge audit` reads JSONL in any of mlx-lm's formats (`prompt`/`completion`,
`messages`, or `text`) and gives each row exactly one status:

- **malformed:** invalid JSON or UTF-8, or missing, empty or non-string fields
- **duplicate:** same prompt and completion as an earlier row, ignoring case, whitespace
  and trailing punctuation
- **low-quality:** refusals, placeholders (`TODO`, `N/A`), completions that repeat the
  prompt, and degenerate repetition
- **clean:** everything else

It also warns when the same prompt has conflicting completions, and when rows contain
sensitive identifiers (see [BFSI checks](#bfsi-checks)).

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

## Training (local, MLX)

`trainjudge train` fine-tunes with LoRA through [mlx-lm](https://github.com/ml-explore/mlx-lm)
on an Apple Silicon Mac. No GPU or cloud account is needed.

```bash
trainjudge train --dataset demo/sql_generation/data.jsonl --model Qwen3-0.6B
```

Before training, it:

1. audits the dataset and drops duplicate and malformed rows. Low-quality rows are dropped
   too unless you pass `--keep-low-quality`.
2. refuses to train if the audit finds card numbers, Aadhaar, PAN or other sensitive
   identifiers, unless you pass `--allow-sensitive-data`.
3. splits the data into train, validation and **held-out test** sets (80/10/10 by
   default). Rows are grouped by answer, so paraphrases of one answer never land in both
   train and test. Only `trainjudge verify` reads the test split.

Each run gets its own folder under `trainjudge-runs/`, holding the splits, the mlx-lm
config, the raw log, the parsed loss curve (`logs/training_log.jsonl`), the adapters and
`run.json` (model, hyperparameters, dataset hash, audit counts and the training summary).
`--dry-run` prepares the folder without training. Defaults: rank 16, 16 layers,
learning rate 5e-5, batch 4, 2 epochs, loss on completions only. Run
`trainjudge train --help` for all options.

## Evaluation: SQL execution accuracy

`trainjudge eval` scores the base model and the fine-tuned adapter on the run's
held-out test split. Neither model sees these rows during training.

```bash
trainjudge eval trainjudge-runs/2026-09-24-sql_generation --db demo/sql_generation/shop.sql
```

Each generated query runs read-only against the database, with a 5-second timeout,
and counts as correct only if it returns the same rows as the gold query. Row order
matters only when the gold query ends with `ORDER BY`. A **lenient** score, which
allows extra columns, is reported alongside it. The gap between the two shows how
much of a change comes from learned conventions (selecting exactly what was asked)
rather than from getting the underlying query right.

The base model is scored fairly: SQL is extracted from code fences and surrounding
prose, both models use greedy decoding with the same prompt, and Qwen3's thinking
mode is off for both. With thinking off, the prompt ends in the same empty think
block the training data contains. Every example's prompt, raw output, extracted SQL
and outcome is saved to `<run>/eval/baseline.json` and `<run>/eval/finetuned.json`.

## Verdict

`trainjudge verify <run-dir>` compares the base model and the fine-tuned adapter, reusing
saved evals where possible, and issues one of three verdicts:

| Verdict | When |
|---|---|
| ✓ IMPROVED | Task accuracy rose by at least 3 points (`--min-improvement`), the gain is statistically significant (exact McNemar test on the same held-out examples, p < 0.05), and no regression category dropped more than 5 points (`--regression-tolerance`) while losing at least 2 items |
| ⚠ REGRESSED | The task improved, but general capability got worse. Don't deploy as-is. |
| ✗ REJECTED | The task didn't improve meaningfully, whatever the training loss did |

The **regression check** is a built-in, offline suite of 60 prompts with automatic
pass/fail checks:

- instruction-following: exact bullet counts, lowercase only, word limits, required
  endings, bare-number arithmetic
- format compliance: JSON objects with given keys, JSON string arrays, numbered lists
- hallucination resistance: questions about prizes, towns and novels that don't exist;
  the model passes by saying it doesn't know

It writes `EXPERIMENT_REPORT.md` (the comparison, outcome breakdown, examples fixed and
broken by fine-tuning, training details and reproduction commands), `MODEL_CARD.md`
(Hugging Face–style, with the verdict) and `eval_results.json` to the run folder.
`--strict` exits with status 1 unless the verdict is IMPROVED, which is useful in CI.

## Known limitations

- The diagnosis step is a heuristic classifier, not a guarantee. It can misclassify
  mixed-goal tasks (partly knowledge, partly format).
- The regression suite is small (60 prompts) and heuristic. It catches broken
  formatting and instruction-following, not subtle capability loss, and its
  hallucination check looks for explicit "I don't know" phrasing.
- The sensitive-data scan catches common Indian and payment identifiers in known
  formats. It won't find names, addresses or identifiers in unusual formats, so it
  doesn't replace a proper data-protection review.
- The audit's low-quality checks are heuristics too. They can miss subtly wrong answers
  and can flag legitimate ones.

## License

MIT
