# Experiment report: 2026-09-24-sql_generation-5

**Verdict: ✗ REJECTED.** Training loss dropped 3%, but task accuracy changed -6.2 points (below the 3-point bar).

**Goal:** improve SQL generation for our shop database

## Task metric: SQL execution accuracy (held-out test split)

| | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Execution accuracy | 30.8% | 24.6% | -6.2 pts |
| Lenient (extra columns allowed) | 32.3% | 26.2% | -6.2 pts |

130 held-out examples. 3 went from wrong to right and 11 from right to wrong (McNemar exact test, p = 0.057). The bar for improvement is 3 points and p < 0.05.

| Outcome | Baseline | Fine-tuned |
|---|---|---|
| correct | 40 | 32 |
| wrong result | 20 | 21 |
| execution error | 31 | 32 |
| no sql | 39 | 45 |

## Regression check (held-out general tasks)

| Category | Baseline | Fine-tuned | Change | |
|---|---|---|---|---|
| General instruction-following (30) | 77% | 77% | +0.0 pts | ✓ |
| Format compliance (18) | 100% | 100% | +0.0 pts | ✓ |
| Hallucination resistance (12) | 0% | 0% | +0.0 pts | ✓ |

A category counts as regressed when its pass rate drops by more than 5 points and it loses at least 2 items net. The suite is small (60 prompts), so treat changes of a few points as noise.

## Examples

### Fixed by fine-tuning (3)

**Q:** Which customers joined after 2024-01-22? Give their full names.

- Gold: `SELECT full_name FROM customers WHERE signup_date > '2024-01-22';`
- Baseline (no sql): _no SQL_
- Fine-tuned (correct): `SELECT full_name FROM customers WHERE signup_date > '2024-01-22';`

**Q:** What's the order count for Rosa Ito?

- Gold: `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Rosa Ito';`
- Baseline (no sql): _no SQL_
- Fine-tuned (correct): `SELECT COUNT(*) AS order_count FROM orders WHERE customer_id = (SELECT customer_id FROM customers WHERE full_name = 'Rosa Ito');`

**Q:** Which customers joined after 2025-04-06? Give their full names.

- Gold: `SELECT full_name FROM customers WHERE signup_date > '2025-04-06';`
- Baseline (wrong result): `SELECT customers.full_name FROM customers JOIN orders ON customers.customer_id = orders.customer_id JOIN order_items ON orders.order_id = order_items.order_id WHERE orders.order_date > '2025-04-06' AND orders.status = 'placed';`
- Fine-tuned (correct): `SELECT full_name FROM customers WHERE signup_date > '2025-04-06';`

### Broken by fine-tuning (11)

**Q:** Count orders from December 2025.

- Gold: `SELECT COUNT(*) FROM orders WHERE order_date >= '2025-12-01' AND order_date < '2026-01-01';`
- Baseline (correct): `SELECT COUNT(*) AS order_count FROM orders WHERE order_date BETWEEN '2025-12-01' AND '2025-12-31';`
- Fine-tuned (wrong result): `SELECT COUNT(*) AS order_count FROM orders JOIN order_items ON orders.order_id = order_items.order_id WHERE order_date >= '2025-12-01' AND order_date <= '2025-12-31';`

**Q:** Which customers joined after 2024-09-04? Give their full names

- Gold: `SELECT full_name FROM customers WHERE signup_date > '2024-09-04';`
- Baseline (correct): `SELECT full_name FROM customers WHERE signup_date > '2024-09-04';`
- Fine-tuned (no sql): _no SQL_

**Q:** What's the order count for Kofi Brooks?

- Gold: `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Kofi Brooks';`
- Baseline (correct): `SELECT COUNT(*) AS order_count FROM orders WHERE customer_id = (SELECT customer_id FROM customers WHERE full_name = 'Kofi Brooks');`
- Fine-tuned (no sql): _no SQL_

**Q:** How many orders has Kofi Brooks placed?

- Gold: `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Kofi Brooks';`
- Baseline (correct): `SELECT COUNT(*) AS order_count FROM orders WHERE customer_id = (SELECT customer_id FROM customers WHERE full_name = 'Kofi Brooks');`
- Fine-tuned (no sql): _no SQL_

**Q:** How many orders has Hana Fischer placed?

- Gold: `SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.full_name = 'Hana Fischer';`
- Baseline (correct): `SELECT COUNT(*) AS num_orders FROM orders WHERE customer_id = (SELECT customer_id FROM customers WHERE full_name = 'Hana Fischer');`
- Fine-tuned (no sql): _no SQL_

_…and 6 more in `eval/*.json`._

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Method | LoRA, rank 4, 4 layers, scale 20 |
| Steps | 20 (0.08 epochs, batch 4, lr 1e-06) |
| Backend | mlx-lm 0.31.3 |
| Duration | 1.0 min |
| Train loss | 2.365 → 2.303 (-2.6%) |
| Validation loss | 2.387 → 2.109 (best 2.109) |

## Data

- Dataset: `demo/sql_generation/data.jsonl` (1,830 rows, sha256 `362905952fda…`)
- Dropped before training: 330 duplicate, 75 malformed, 125 low-quality
- Splits: 1,039 train · 131 valid · 130 test (grouped by normalized completion, seed 0)
- Database: `demo/sql_generation/shop.sql` (sha256 `a89b1db609b5…`)

## Reproduce

```bash
trainjudge train --dataset demo/sql_generation/data.jsonl --model Qwen/Qwen3-0.6B --iters 20 --batch-size 4 --learning-rate 1e-06 --rank 4 --num-layers 4 --seed 0
trainjudge eval <run-dir> --db demo/sql_generation/shop.sql
trainjudge verify <run-dir>
```

## Caveats

- Execution accuracy checks result rows on one database. A query can match by coincidence on this data and still be wrong in general.
- The regression suite is a small heuristic check, not a full benchmark.
- Both models use greedy decoding with the same prompt, so results are deterministic for this setup but may differ with sampling or other prompts.
