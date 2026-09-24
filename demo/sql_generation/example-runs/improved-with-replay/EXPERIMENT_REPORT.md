# Experiment report: 2026-09-24-sql_generation-3

**Verdict: ✓ IMPROVED.** Task accuracy improved +67.7 points on held-out examples (p < 0.001), with no regressions.

**Goal:** improve SQL generation for our shop database

## Task metric: SQL execution accuracy (held-out test split)

| | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Execution accuracy | 30.8% | 98.5% | +67.7 pts |
| Lenient (extra columns allowed) | 32.3% | 98.5% | +66.2 pts |

130 held-out examples. 89 went from wrong to right and 1 from right to wrong (McNemar exact test, p < 0.001). The bar for improvement is 3 points and p < 0.05.

| Outcome | Baseline | Fine-tuned |
|---|---|---|
| correct | 40 | 128 |
| wrong result | 20 | 1 |
| execution error | 31 | 1 |
| no sql | 39 | 0 |

## Regression check (held-out general tasks)

| Category | Baseline | Fine-tuned | Change | |
|---|---|---|---|---|
| General instruction-following (30) | 77% | 77% | +0.0 pts | ✓ |
| Format compliance (18) | 100% | 94% | -5.6 pts | ✓ |
| Hallucination resistance (12) | 0% | 17% | +16.7 pts | ✓ |

A category counts as regressed when its pass rate drops by more than 5 points and it loses at least 2 items net. The suite is small (60 prompts), so treat changes of a few points as noise.

## Examples

### Fixed by fine-tuning (89)

**Q:** Show the top 14 products by price.

- Gold: `SELECT product_name FROM products ORDER BY unit_price_cents DESC LIMIT 14;`
- Baseline (wrong result): `SELECT p.product_id, p.product_name, p.category, p.unit_price_cents FROM products p JOIN order_items oi ON p.product_id = oi.product_id ORDER BY oi.unit_price_cents DESC LIMIT 14;`
- Fine-tuned (correct): `SELECT product_name FROM products ORDER BY unit_price_cents DESC LIMIT 14;`

**Q:** What are the 14 most expensive products

- Gold: `SELECT product_name FROM products ORDER BY unit_price_cents DESC LIMIT 14;`
- Baseline (wrong result): `SELECT product_id, product_name, unit_price_cents FROM products ORDER BY unit_price_cents DESC LIMIT 14;`
- Fine-tuned (correct): `SELECT product_name FROM products ORDER BY unit_price_cents DESC LIMIT 14;`

**Q:** What  is  the email address of Ivan Garcia?

- Gold: `SELECT email FROM customers WHERE full_name = 'Ivan Garcia';`
- Baseline (no sql): _no SQL_
- Fine-tuned (correct): `SELECT email FROM customers WHERE full_name = 'Ivan Garcia';`

**Q:** Give me Ivan Garcia's email.

- Gold: `SELECT email FROM customers WHERE full_name = 'Ivan Garcia';`
- Baseline (no sql): _no SQL_
- Fine-tuned (correct): `SELECT email FROM customers WHERE full_name = 'Ivan Garcia';`

**Q:** Count the orders placed by Mateo Fischer.

- Gold: `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Mateo Fischer';`
- Baseline (execution error): `SELECT COUNT(*) AS order_count FROM orders JOIN order_items ON orders.customer_id = order_items.customer_id WHERE orders.customer_id = 'Mateo Fischer';`
- Fine-tuned (correct): `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Mateo Fischer';`

_…and 84 more in `eval/*.json`._

### Broken by fine-tuning (1)

**Q:** How many orders has Nina Evans placed?

- Gold: `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Nina Evans';`
- Baseline (correct): `SELECT COUNT(*) AS order_count FROM orders WHERE customer_id = ( SELECT customer_id FROM customers WHERE full_name = 'Nina Evans' );`
- Fine-tuned (wrong result): `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Nina Evans' AND c.email = 'email@example.com';`

### General-capability items that broke (2)

- **Describe volcanoes in fewer than 15 words.**
  - Fine-tuned: `Volcanoes are large, active geological features that erupt magma, releasing gases and ash into the atmosphere.`
- **List 4 animals as a JSON array of strings. Output only the JSON.**
  - Fine-tuned: ````json [{"animal": "cat"}, {"animal": "dog"}, {"animal": "bird"}, {"animal": "fish"}] ````

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Method | LoRA, rank 8, 8 layers, scale 20 |
| Steps | 150 (0.48 epochs, batch 4, lr 2e-05) |
| Backend | mlx-lm 0.31.3 |
| Duration | 7.7 min |
| Train loss | 0.667 → 0.051 (-92.4%) |
| Validation loss | 2.387 → 0.009 (best 0.009) |

## Data

- Dataset: `demo/sql_generation/data.jsonl` (1,830 rows, sha256 `362905952fda…`)
- Dropped before training: 330 duplicate, 75 malformed, 125 low-quality
- Splits: 1,039 train · 131 valid · 130 test (grouped by normalized completion, seed 0)
- Database: `demo/sql_generation/shop.sql` (sha256 `a89b1db609b5…`)

## Reproduce

```bash
trainjudge train --dataset demo/sql_generation/data.jsonl --model Qwen/Qwen3-0.6B --iters 150 --batch-size 4 --learning-rate 2e-05 --rank 8 --num-layers 8 --seed 0
trainjudge eval <run-dir> --db demo/sql_generation/shop.sql
trainjudge verify <run-dir>
```

## Caveats

- Execution accuracy checks result rows on one database. A query can match by coincidence on this data and still be wrong in general.
- The regression suite is a small heuristic check, not a full benchmark.
- Both models use greedy decoding with the same prompt, so results are deterministic for this setup but may differ with sampling or other prompts.
