# Demo: loan & deposit FAQ (BFSI, the "no, don't fine-tune" case)

Q&A pairs about the products of Northstar Bank, a fictional bank: home and personal
loan rates, processing fees, FD rates, minimum balance charges, card fees and KYC rules.
The source documents are in [docs/](docs/).

Every answer is a fact that moves: rates reset when the repo rate changes, schedules of
charges get revised, and KYC rules follow regulator circulars. A model fine-tuned on
these answers would give confident, stale numbers after the next reset, with no way to
show which version of a document it used. TrainJudge classifies this as a knowledge gap
and recommends retrieval over versioned documents with effective dates.

| Rows | Count |
|---|---|
| Clean (24 facts × 6 question phrasings) | 144 |
| Duplicates | 8 |
| Low-quality | 3 |
| Malformed | 2 |
| **Total** | **157** |

```bash
trainjudge diagnose --dataset demo/bfsi_loan_faq/data.jsonl --model Qwen3-0.6B \
  --goal "answer customer questions about our loan and FD interest rates and charges"
python demo/bfsi_loan_faq/generate.py   # regenerate (output is identical on every run)
```
