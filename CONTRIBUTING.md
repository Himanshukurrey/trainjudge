# Contributing to TrainJudge

Thanks for considering a contribution. Here's the workflow.

## How to contribute

1. **Fork the repo** and clone your fork locally.
2. **Create a branch** for your change: `git checkout -b fix/short-description`.
3. **Make your change**, with tests if you're changing behavior.
4. **Run the test suite and linter** locally before opening a PR:
   ```bash
   pip install -e ".[dev]"
   pytest
   ruff check .
   ruff format --check .
   ```
   CI runs the same checks on Linux, Windows and macOS, on Python 3.10 and 3.12. It also checks that the
   committed demo datasets match their generators. Running these locally first saves a round trip.
5. **Open a pull request** against `main`. Fill in the PR template: what the change does, why, and how you
   tested it.
6. A maintainer will review it. Please be patient; this is currently maintained part-time.

## Tests don't need a GPU

The test suite never downloads a model or needs `mlx-lm`. Training and generation go through fake backends
(see `fake_backend` in `tests/test_training.py` and `good_model` in `tests/test_verify.py`), so every test runs
on any OS in a few seconds. If you change something that only shows up with a real model, such as prompt
rendering, generation or `mlx-lm` flags, also run it on an Apple Silicon Mac and say so in the PR:

```bash
pip install -e ".[dev,mlx]"
trainjudge train --dataset demo/sql_generation/data.jsonl --model Qwen3-0.6B --iters 20
trainjudge verify trainjudge-runs/<run> --db demo/sql_generation/shop.sql
```

## Adding a domain pack

Domain packs add industry-specific checks to `diagnose` (see the README's "Domain packs"
section). To add one, say for healthcare:

1. Create `src/trainjudge/domains/healthcare.py` defining `PACK = DomainPack(...)`, using
   `domains/bfsi.py` as the template:
   - `terms`: words that identify the domain in a goal or dataset. Prefer specific terms;
     generic words like "claim" appear in many domains.
   - `changing_fact_terms`, `changing_facts` and `changing_fact_note`: facts that change
     and belong in retrieval.
   - `high_stakes_terms` and `high_stakes_note`: decisions about people that shouldn't be
     fully automated.
   - `closing_notes`: anything else worth saying for this domain.
2. Register it in `_load_packs()` in `src/trainjudge/domains/__init__.py`.
3. Add tests showing it's detected from a goal, isn't detected on the other demos, and
   produces its notes.
4. Add two demo datasets under `demo/`, each with a `generate.py`: one where fine-tuning
   fits and one where it doesn't.

Keep sensitive-data detectors in `src/trainjudge/pii.py` rather than in a pack, so every
dataset is scanned whatever its domain. Compliance pointers should name the rule, not
interpret it.

## Demo datasets

The files in `demo/*/` are generated, not hand-written. To change one, edit its `generate.py`, run it, and
commit both the script and its output. Tests pin the exact number of injected duplicate, low-quality,
malformed and PII rows, so update those counts too if you change them.

## Why PRs go through review

Every change lands through a reviewed pull request, so the project has a consistent, auditable history and a
second pair of eyes on every change.

## What makes a good PR

- **Small and focused.** One logical change per PR is much easier to review than five unrelated fixes bundled
  together.
- **Tested.** If you fixed a bug, a regression test that would have caught it is the strongest evidence the
  fix is real.
- **Explained.** A one-line "fixed the bug" isn't enough. Say what was broken and why your change addresses
  it.

## Reporting bugs

Open an issue describing what you expected vs. what happened, your OS, Python and `mlx-lm` versions, and the
exact command you ran. For problems with a run, attaching its `run.json` (and `eval_results.json`, if it got
that far) is the fastest way to get it looked at. Check them for anything private first.

## Code of conduct

Be respectful. Disagreements about code are fine; personal attacks aren't. See
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Releasing (maintainer only)

1. Bump `version` in `pyproject.toml`, `src/trainjudge/__init__.py`, `.claude-plugin/plugin.json` and the
   version badge at the top of `README.md`.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`
3. `gh release create vX.Y.Z --generate-notes`

Publishing that release triggers `.github/workflows/release.yml`, which builds the package, checks that the
wheel installs and `trainjudge --version` and `trainjudge audit` run, and publishes to PyPI via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no stored API token).

**One-time setup required before the first release**, done once on pypi.org by whoever owns the PyPI project:
add a "pending publisher" for a project named `trainjudge` under Account Settings → Publishing, with:

- Owner: `Himanshukurrey`
- Repository: `trainjudge`
- Workflow name: `release.yml`
- Environment name: `pypi`

This reserves the trust relationship before the package exists on PyPI, so the first `gh release create` can
publish without a manual upload.
