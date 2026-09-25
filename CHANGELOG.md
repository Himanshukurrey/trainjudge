# Changelog

## 0.2.0 (2026-09-25)

### Train and verify on NVIDIA GPUs, not just Macs

- **PyTorch backend** (`--backend torch`): LoRA with transformers + peft on NVIDIA GPUs via CUDA
  (Windows and Linux), Apple GPUs via MPS, or CPU. `--backend auto` keeps MLX on Apple Silicon and
  uses PyTorch elsewhere. Both backends render prompts the same way, train with the loss on
  completions only and report the same progress. Verified end to end on an NVIDIA T4.
- **Colab notebook** (`notebooks/trainjudge_colab.ipynb`): diagnose, mask, train and verify on a
  free GPU, for anyone without one.

### Every domain, not just one

- **Domain packs** for healthcare, legal, e-commerce/retail, customer support, HR/recruiting, BFSI
  and education: facts that change (pointing to retrieval), high-stakes decision warnings and
  domain notes. `--domain` forces or disables one.
- **Demos for every pack**: a fine-tune case and a retrieval case per domain, 600 rows each for
  the fine-tune cases so a real gain can be statistically significant.

### Evals and verdicts

- **JSON extraction eval**: exact match plus field-level and per-field accuracy. `eval` and
  `verify` detect SQL or JSON from the test split (`--task` overrides), so every domain's
  fine-tune demo can be verified.
- **Baseline cache**: base-model evals are reused across runs with the same model, test split,
  decoding and database. `verify --rerun` bypasses it.
- Reports include a complete reproduce command and the device training actually used.

### Data safety

- **PII masking**: `train --mask-sensitive` and `audit --write-clean --mask-sensitive` replace
  identifiers with placeholders (`[EMAIL]`, `[MRN]`, `[CARD]`…).
- **Region-aware detection**: US SSNs, UK National Insurance numbers, IBANs (mod-97),
  international and US phone numbers, labelled medical record numbers and dates of birth, on top
  of cards, Aadhaar, PAN, UPI IDs and emails. Compliance pointers follow what was found.

### Progress for people and agents

- `trainjudge status --watch --milestones` prints one line per milestone, so agents can relay
  progress even where command output isn't streamed (such as the Claude Code VS Code extension).

### Fixes

- Label-like answers are split per row, so no label is missing from training; paraphrase-style
  answers stay grouped.
- Status files survive Windows file locking; cleaned datasets always use LF line endings.

### Other

- Relicensed from MIT to Apache-2.0.

## 0.1.0 (2026-09-24)

First release: `diagnose` (knowledge / format / cost / prompt gap), `audit` (duplicates,
malformed, low-quality and sensitive rows), `train` (MLX LoRA with held-out splits and
`--replay`), `eval` (SQL execution accuracy), `verify` (IMPROVED / REGRESSED / REJECTED with a
regression suite and reports), `status`, and a Claude Code plugin plus `AGENTS.md`.
