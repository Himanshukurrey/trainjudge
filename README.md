# TrainJudge

Before you fine-tune, TrainJudge tells you whether fine-tuning is even the right move.
After you fine-tune, it tells you whether it actually worked — on the task metric, not training loss.

> Status: pre-alpha (v0.1 in progress). `verify` is not implemented yet.

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

## Known limitations

- The diagnosis step is a heuristic classifier, not a guarantee. It can misclassify
  mixed-goal tasks (partly knowledge, partly format).
- The sensitive-data scan catches common Indian and payment identifiers in known
  formats. It won't find names, addresses or identifiers in unusual formats, so it
  doesn't replace a proper data-protection review.
- The audit's low-quality checks are heuristics too. They can miss subtly wrong answers
  and can flag legitimate ones.

## License

MIT
