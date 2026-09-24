---
base_model: Qwen/Qwen3-0.6B
library_name: mlx
tags:
  - lora
  - mlx
  - text-to-sql
  - trainjudge
model-index:
  - name: 2026-09-24-sql_generation-5
    results:
      - task:
          type: text-to-sql
        metrics:
          - type: execution_accuracy
            value: 0.2462
            name: SQL execution accuracy (held-out)
---

# LoRA adapter for Qwen/Qwen3-0.6B

**TrainJudge verdict: ✗ REJECTED.** ✗ Not recommended for deployment: no meaningful improvement on the task.

## Intended use

improve SQL generation for our shop database

## Evaluation

| Metric | Base model | This adapter |
|---|---|---|
| SQL execution accuracy (n=130) | 30.8% | 24.6% |
| General instruction-following (n=30) | 77% | 77% |
| Format compliance (n=18) | 100% | 100% |
| Hallucination resistance (n=12) | 0% | 0% |

See `EXPERIMENT_REPORT.md` for the full comparison and examples.

## Training

LoRA (rank 4, 4 layers) for 20 steps on 1,039 examples from `demo/sql_generation/data.jsonl`, with the loss on completions only.

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen3-0.6B", adapter_path="trainjudge-runs/2026-09-24-sql_generation-5/adapters")
```

## Limitations

- Trained and evaluated on one schema. Expect it to fail on other databases.
- Evaluated on a held-out split of the same dataset, which may share templates or phrasing with the training data.
