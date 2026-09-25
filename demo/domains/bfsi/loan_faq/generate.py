"""Generate the BFSI loan & deposit FAQ demo dataset (deterministic).

Northstar Bank is a fictional bank. This writes its product and charges
documents to docs/ and a Q&A dataset to data.jsonl whose answers are rates,
fees and KYC rules recalled from those documents. These facts move with repo
rate resets, revised schedules of charges and regulator circulars, which is
the "knowledge gap" case: TrainJudge should recommend retrieval over
versioned documents instead of fine-tuning. A few duplicate, low-quality and
malformed rows are mixed in.

    python demo/domains/bfsi/loan_faq/generate.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260924

N_DUPLICATE = 8
N_LOW_QUALITY = 3
N_MALFORMED = 2

DOCS = {
    "home_loans.md": (
        "Home Loans",
        [
            (
                "the current home loan interest rate",
                (
                    "Northstar Bank home loans start at 8.35% p.a. for salaried borrowers. The rate is "
                    "linked to the RBI repo rate and resets every 3 months."
                ),
            ),
            (
                "the processing fee for home loans",
                "The home loan processing fee is 0.50% of the loan amount, capped at ₹10,000 plus GST.",
            ),
            (
                "the prepayment charge on floating-rate home loans",
                (
                    "There is no prepayment or foreclosure charge on floating-rate home loans taken by "
                    "individuals."
                ),
            ),
            (
                "the maximum home loan tenure",
                "Home loans are available for up to 30 years, ending by the borrower's 70th birthday.",
            ),
        ],
    ),
    "personal_loans.md": (
        "Personal Loans",
        [
            (
                "the personal loan interest rate range",
                (
                    "Personal loans are offered at 10.99% to 18.50% p.a., depending on credit score and "
                    "income."
                ),
            ),
            (
                "the foreclosure charge for personal loans",
                (
                    "Personal loans can be foreclosed after 12 EMIs with a 3% charge on the outstanding "
                    "principal."
                ),
            ),
            (
                "the maximum personal loan amount",
                ("Personal loans go up to ₹25 lakh for salaried customers earning at least ₹30,000 a month."),
            ),
            (
                "the penal charge for a late EMI",
                "A late EMI attracts a penal charge of 2% per month on the overdue amount.",
            ),
        ],
    ),
    "fixed_deposits.md": (
        "Fixed Deposits",
        [
            (
                "the one-year fixed deposit rate",
                "One-year fixed deposits earn 6.80% p.a. for general customers.",
            ),
            (
                "the extra FD rate for senior citizens",
                "Senior citizens earn an additional 0.50% p.a. on fixed deposits of all tenures.",
            ),
            (
                "the penalty for breaking a fixed deposit early",
                "Premature withdrawal of a fixed deposit carries a 1% penalty on the applicable rate.",
            ),
            (
                "the minimum fixed deposit amount",
                "The minimum fixed deposit is ₹5,000, for tenures from 7 days to 10 years.",
            ),
        ],
    ),
    "savings_accounts.md": (
        "Savings Accounts",
        [
            (
                "the minimum balance for a regular savings account",
                (
                    "Regular savings accounts need an average monthly balance of ₹10,000 in metro branches "
                    "and ₹5,000 elsewhere."
                ),
            ),
            (
                "the savings account interest rate",
                "Savings balances up to ₹5 lakh earn 3.00% p.a.; balances above ₹5 lakh earn 3.50% p.a.",
            ),
            (
                "the charge for not maintaining the minimum balance",
                "Falling short of the minimum balance costs 5% of the shortfall, up to ₹500 a month.",
            ),
            (
                "the free ATM withdrawal limit",
                (
                    "Customers get 5 free withdrawals a month at Northstar ATMs and 3 at other banks' ATMs "
                    "in metro cities."
                ),
            ),
        ],
    ),
    "cards_and_charges.md": (
        "Cards and Charges",
        [
            (
                "the annual fee for the Northstar Platinum credit card",
                (
                    "The Platinum credit card costs ₹999 a year, waived if you spent ₹1.5 lakh in the "
                    "previous year."
                ),
            ),
            (
                "the interest rate on unpaid credit card balances",
                "Unpaid credit card balances attract interest of 3.6% per month.",
            ),
            (
                "the forex markup on credit cards",
                "International transactions on Northstar credit cards carry a 3.5% forex markup.",
            ),
            (
                "the debit card annual maintenance charge",
                "Classic debit cards have an annual maintenance charge of ₹250 plus GST.",
            ),
        ],
    ),
    "kyc.md": (
        "KYC Requirements",
        [
            (
                "the documents accepted for KYC",
                (
                    "Accepted KYC documents are Aadhaar, passport, voter ID and driving licence, along "
                    "with PAN or Form 60."
                ),
            ),
            (
                "the re-KYC schedule",
                (
                    "Low-risk customers must complete re-KYC every 10 years; high-risk customers every "
                    "2 years."
                ),
            ),
            (
                "the Video KYC option",
                (
                    "New customers can complete Video KYC with a bank official in about 10 minutes using "
                    "Aadhaar and PAN."
                ),
            ),
            (
                "the balance limit before full KYC",
                (
                    "Accounts opened with OTP-based e-KYC are limited to a ₹1 lakh balance until full KYC "
                    "is completed, within 1 year."
                ),
            ),
        ],
    ),
}

QUESTION_TEMPLATES = [
    "What is {topic}?",
    "Can you tell me {topic}?",
    "Quick question: what's {topic}?",
    "As per Northstar Bank's current terms, what is {topic}?",
    "A customer is asking about {topic}. What should I tell them?",
    "Please confirm {topic}.",
]
# Only used for injected bad rows, so they never collide with a clean prompt.
RESERVED_TEMPLATE = "Hi, what's {topic}?"


def _row(question: str, answer, source: str) -> dict:
    return {"prompt": question, "completion": answer, "source": source}


def write_docs() -> None:
    docs_dir = HERE / "docs"
    docs_dir.mkdir(exist_ok=True)
    for filename, (title, facts) in DOCS.items():
        body = "\n".join(f"- {answer}" for _, answer in facts)
        (docs_dir / filename).write_text(
            f"# Northstar Bank: {title}\n\nEffective September 1, 2026. Rates and charges are "
            f"subject to change.\n\n{body}\n",
            encoding="utf-8",
        )


def main() -> None:
    rng = random.Random(SEED)
    write_docs()

    clean = [
        _row(template.format(topic=topic), answer, source)
        for source, (_, facts) in DOCS.items()
        for topic, answer in facts
        for template in QUESTION_TEMPLATES
    ]
    lines = [json.dumps(r) for r in clean]
    for i in range(N_DUPLICATE):
        row = dict(rng.choice(clean))
        row["prompt"] = [row["prompt"], row["prompt"].lower()][i % 2]
        lines.append(json.dumps(row))

    facts = [(s, t) for s, (_, fs) in DOCS.items() for t, _ in fs]
    bad = rng.sample(facts, N_LOW_QUALITY + N_MALFORMED)
    low_quality_answers = ["N/A", "TBD", "I'm sorry, I can't share rate information."]
    for (source, topic), answer in zip(bad, low_quality_answers):
        lines.append(json.dumps(_row(RESERVED_TEMPLATE.format(topic=topic), answer, source)))
    for i, (source, topic) in enumerate(bad[N_LOW_QUALITY:]):
        question = RESERVED_TEMPLATE.format(topic=topic)
        lines.append(
            [
                json.dumps({"prompt": question, "source": source}),
                json.dumps(_row(question, 6.8, source)),
            ][i]
        )
    rng.shuffle(lines)

    out = HERE / "data.jsonl"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {len(lines)} rows to {out} ({len(clean)} clean, {N_DUPLICATE} duplicate, "
        f"{N_LOW_QUALITY} low-quality, {N_MALFORMED} malformed) and {len(DOCS)} docs"
    )


if __name__ == "__main__":
    main()
