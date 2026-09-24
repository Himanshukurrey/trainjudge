# Experiment report: 2026-09-24-sql_generation

**Verdict: ⚠ REGRESSED.** Task accuracy improved +66.9 points; general instruction-following regressed -30.0 points; format compliance regressed -50.0 points.

**Goal:** improve SQL generation for our shop database

## Task metric: SQL execution accuracy (held-out test split)

| | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Execution accuracy | 30.8% | 97.7% | +66.9 pts |
| Lenient (extra columns allowed) | 32.3% | 97.7% | +65.4 pts |

130 held-out examples. 89 went from wrong to right and 2 from right to wrong (McNemar exact test, p < 0.001). The bar for improvement is 3 points and p < 0.05.

| Outcome | Baseline | Fine-tuned |
|---|---|---|
| correct | 40 | 127 |
| wrong result | 20 | 3 |
| execution error | 31 | 0 |
| no sql | 39 | 0 |

## Regression check (held-out general tasks)

| Category | Baseline | Fine-tuned | Change | |
|---|---|---|---|---|
| General instruction-following (30) | 77% | 47% | -30.0 pts | ⚠ regressed |
| Format compliance (18) | 100% | 50% | -50.0 pts | ⚠ regressed |
| Hallucination resistance (12) | 0% | 0% | +0.0 pts | ✓ |

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

### Broken by fine-tuning (2)

**Q:** How many orders were placed in December 2025?

- Gold: `SELECT COUNT(*) FROM orders WHERE order_date >= '2025-12-01' AND order_date < '2026-01-01';`
- Baseline (correct): `SELECT COUNT(*) AS order_count FROM orders WHERE order_date >= '2025-12-01' AND order_date <= '2025-12-31';`
- Fine-tuned (wrong result): `SELECT COUNT(*) FROM orders WHERE order_date >= '2025-12-01' AND order_date < '2031-12-01';`

**Q:** Count orders from December 2025.

- Gold: `SELECT COUNT(*) FROM orders WHERE order_date >= '2025-12-01' AND order_date < '2026-01-01';`
- Baseline (correct): `SELECT COUNT(*) AS order_count FROM orders WHERE order_date BETWEEN '2025-12-01' AND '2025-12-31';`
- Fine-tuned (wrong result): `SELECT COUNT(*) FROM orders WHERE order_date >= '2025-12-01' AND order_date < '2031-12-01';`

### General-capability items that broke (19)

- **Write exactly 3 bullet points about the ocean. Start each bullet with '- ' and write nothing else.**
  - Fine-tuned: `- 'The ocean is a vast, mysterious, and life-giving environment that includes the deep sea, coral reefs, and marine life.`
- **Write two sentences about the ocean. End your reply with the exact phrase 'That is all.'**
  - Fine-tuned: `The ocean is a vast and mysterious place, full of life and the sound of waves that echo through the water.`
- **Write exactly 3 bullet points about coffee. Start each bullet with '- ' and write nothing else.**
  - Fine-tuned: `- '咖啡是世界上最具代表性的饮品之一，它不仅提供美味，还具有多种健康属性。`
- **Write two sentences about coffee. End your reply with the exact phrase 'That is all.'**
  - Fine-tuned: `☕ Coffee is a drink that is enjoyed by many, and it is a source of energy and relaxation.`
- **Write exactly 3 bullet points about bicycles. Start each bullet with '- ' and write nothing else.**
  - Fine-tuned: `- 'Bicycles are a popular mode of transportation that is both fun and efficient. They are known for their simplicity and ease of use.`
- _…and 14 more in `eval/regression_finetuned.json`._

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Method | LoRA, rank 16, 16 layers, scale 20 |
| Steps | 520 (2 epochs, batch 4, lr 5e-05) |
| Backend | mlx-lm 0.31.3 |
| Duration | 20.1 min |
| Train loss | 0.653 → 0.001 (-99.8%) |
| Validation loss | 2.387 → 0.008 (best 0.001) |

## Data

- Dataset: `demo/sql_generation/data.jsonl` (1,830 rows, sha256 `362905952fda…`)
- Dropped before training: 330 duplicate, 75 malformed, 125 low-quality
- Splits: 1,039 train · 131 valid · 130 test (grouped by normalized completion, seed 0)
- Database: `demo/sql_generation/shop.sql` (sha256 `a89b1db609b5…`)

## Reproduce

```bash
trainjudge train --dataset demo/sql_generation/data.jsonl --model Qwen/Qwen3-0.6B --iters 520 --batch-size 4 --learning-rate 5e-05 --rank 16 --num-layers 16 --seed 0
trainjudge eval <run-dir> --db demo/sql_generation/shop.sql
trainjudge verify <run-dir>
```

## Caveats

- Execution accuracy checks result rows on one database. A query can match by coincidence on this data and still be wrong in general.
- The regression suite is a small heuristic check, not a full benchmark.
- Both models use greedy decoding with the same prompt, so results are deterministic for this setup but may differ with sampling or other prompts.
