# Demo: bank transaction categorization (BFSI, the "yes, fine-tune" case)

Bank statement narrations paired with a JSON label, as a bank's own categorization
engine would produce them:

```
Narration: NACH/DR/SUNRISE FINANCE/LN12345678901
→ {"category": "emi", "merchant": "Sunrise Finance", "channel": "NACH", "direction": "debit"}
```

Narrations cover UPI, NEFT, IMPS, NACH/ACH, card POS and e-commerce, ATM/AePS, BBPS,
bank charges and interest credits. The model has to learn the bank's taxonomy (NACH
debits to lenders are `emi`, `ACH D-` to AMCs is `mutual_fund_sip`, `INT.PD` is
`interest`) and how to normalize merchant names. That is a format/behavior gap, and
field-level exact match measures it objectively.

**Sensitive data, planted on purpose.** 14 clean rows leak unmasked identifiers: card
numbers in POS narrations, Aadhaar numbers in AePS withdrawals, a PAN in NEFT
narrations, and personal (phone-number) UPI IDs. They show the diagnosis's
sensitive-data warning. All values are checksum-valid test numbers (Luhn and Verhoeff),
generated at random. All banks, merchants and people are fictional.

| Rows | Count |
|---|---|
| Clean (14 of them contain PII) | 600 |
| Duplicates | 36 |
| Low-quality (placeholders, refusals) | 12 |
| Malformed | 8 |
| **Total** | **656** |

```bash
trainjudge diagnose --dataset demo/bfsi_transactions/data.jsonl --model Qwen3-0.6B \
  --goal "categorize bank transaction narrations into our category JSON"
python demo/bfsi_transactions/generate.py   # regenerate (output is identical on every run)
```
