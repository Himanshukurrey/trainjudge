# Demo: SQL generation (the "yes, fine-tune" case)

Text-to-SQL pairs over a small shop database ([schema.sql](schema.sql)). The schema has
conventions a base model can't guess: money in integer cents, an `is_active` flag on
products, and revenue that excludes cancelled orders. Learning those conventions is a
format/behavior gap, which fine-tuning handles well, and execution accuracy measures it
objectively.

The data is synthetic and generated from templates by [generate.py](generate.py) (the
output is identical on every run). Every clean gold query is executed against a seeded
SQLite database and must return rows.

| Rows | Count |
|---|---|
| Clean | 1,300 |
| Duplicates (exact, whitespace, case and trailing-punctuation variants) | 330 |
| Low-quality (refusals, placeholders, prompt echoes, degenerate repetition) | 125 |
| Malformed (truncated JSON, missing/empty/non-string completion, non-object rows) | 75 |
| **Total** | **1,830** |

```bash
trainjudge audit demo/sql_generation/data.jsonl
python demo/sql_generation/generate.py   # regenerate
```
