# Security Policy

## Supported Versions

TrainJudge is pre-1.0. There's no version support matrix yet: only the latest release on `main` gets fixes.

## Reporting a Vulnerability

Please **don't open a public GitHub issue** for a security vulnerability. Instead, use
[GitHub's private vulnerability reporting](https://github.com/Himanshukurrey/trainjudge/security/advisories/new)
for this repository, or open a regular issue asking to be contacted privately if that isn't available to you.

Include what you'd include in any good bug report: the affected version or commit, steps to reproduce, and the
impact you think it has.

This is a solo-maintained, pre-1.0 project, so please allow a few days for a first response.

## What's in scope

Worth knowing before reporting:

- **Model-generated SQL.** `trainjudge eval` runs SQL written by a model against the database you pass with
  `--db`. It works on an in-memory copy opened with `PRAGMA query_only`, extracts only `SELECT`/`WITH`
  statements, and applies a time limit and row cap. A way for generated SQL to write to the original database
  file, escape that sandbox, or exhaust memory despite the caps is in scope.
- **Sensitive-data scan.** The audit reports card numbers, Aadhaar numbers, PANs and similar identifiers by
  line number only, and `train` refuses such datasets by default. A case where TrainJudge prints or stores a
  matched value, or where `--allow-sensitive-data` isn't required for a dataset the scan flags, is in scope.
  The scan missing an identifier in an unusual format is a bug, not a vulnerability. It's a heuristic, and
  the README says so.
- **Reports contain examples.** `eval/*.json`, `EXPERIMENT_REPORT.md` and `MODEL_CARD.md` include prompts and
  model outputs from your test split, by design, so verdicts can be traced to examples. Treat run folders as
  being as sensitive as the dataset they came from.
- **Models.** Models are loaded through `mlx-lm` from Hugging Face or a local path you choose. Loading an
  untrusted model is the same trust decision as loading it any other way.
