"""Generate the policy-docs demo dataset (deterministic).

Brightlane is a fictional home-goods retailer. This writes its policy
documents to docs/ and a Q&A dataset to data.jsonl whose answers are facts
recalled from those documents — the "knowledge gap" case where TrainJudge
should recommend retrieval instead of fine-tuning. A few duplicate,
low-quality and malformed rows are mixed in so the audit has something to find.

    python demo/policy_docs/generate.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260924

N_DUPLICATE = 12
N_LOW_QUALITY = 5
N_MALFORMED = 3

DOCS = {
    "refund_policy.md": (
        "Refund Policy",
        [
            (
                "the return window for unopened items",
                "Unopened items can be returned within 45 days of delivery for a full refund.",
            ),
            (
                "the return window for opened electronics",
                (
                    "Opened electronics can be returned within 15 days of delivery and are subject to a "
                    "10% restocking fee."
                ),
            ),
            (
                "the processing time for refunds",
                (
                    "Refunds are issued to the original payment method within 5 business days after the "
                    "returned item reaches our warehouse."
                ),
            ),
            (
                "the refund policy for final-sale items",
                (
                    "Final-sale items, marked with a red tag on the product page, cannot be returned or "
                    "refunded."
                ),
            ),
            (
                "the policy on who pays return shipping",
                (
                    "Brightlane pays return shipping for defective or incorrect items; for all other "
                    "returns a $7.95 label fee is deducted from the refund."
                ),
            ),
            (
                "the refund option when there's no receipt",
                (
                    "Without an order number or receipt, returns are accepted for store credit only, at "
                    "the item's lowest price in the last 90 days."
                ),
            ),
        ],
    ),
    "shipping_policy.md": (
        "Shipping Policy",
        [
            (
                "the free shipping threshold",
                ("Standard shipping is free on orders of $75 or more within the contiguous United States."),
            ),
            (
                "the standard shipping delivery time",
                "Standard shipping arrives in 4 to 7 business days.",
            ),
            (
                "the cost of express shipping",
                "Express shipping costs $19.95 and arrives in 2 business days.",
            ),
            (
                "the order cutoff time for same-day dispatch",
                "Orders placed before 1:00 p.m. Eastern Time on a business day ship the same day.",
            ),
            (
                "the shipping policy for Alaska and Hawaii",
                ("We ship to Alaska and Hawaii with a flat $24.95 surcharge; free shipping does not apply."),
            ),
            (
                "the policy on international shipping",
                (
                    "Brightlane ships internationally to Canada and Mexico only; duties and taxes are "
                    "collected at checkout."
                ),
            ),
        ],
    ),
    "warranty_policy.md": (
        "Warranty Policy",
        [
            (
                "the warranty length for furniture",
                "Furniture is covered by a 5-year limited warranty against structural defects.",
            ),
            (
                "the warranty length for small appliances",
                "Small appliances carry a 2-year limited warranty from the date of delivery.",
            ),
            (
                "the list of warranty exclusions",
                (
                    "The warranty does not cover normal wear, accidental damage, or damage from "
                    "commercial use."
                ),
            ),
            (
                "the process for filing a warranty claim",
                (
                    "Warranty claims are filed from the Order History page with a photo of the defect; "
                    "claims are reviewed within 3 business days."
                ),
            ),
            (
                "the rule on transferring a warranty to a new owner",
                "Warranties are non-transferable and apply only to the original purchaser.",
            ),
            (
                "the extended warranty option",
                (
                    "The Brightlane Care plan extends any warranty by 3 years and costs 12% of the item's "
                    "purchase price."
                ),
            ),
        ],
    ),
    "price_matching.md": (
        "Price Matching",
        [
            (
                "the price-match window",
                "We match a lower price on an identical item within 14 days of purchase.",
            ),
            (
                "the list of competitors eligible for price matching",
                (
                    "Price matching applies only to authorized U.S. retailers with the item in stock; "
                    "marketplace sellers are excluded."
                ),
            ),
            (
                "the policy on matching Brightlane's own sale prices",
                (
                    "If Brightlane lowers its own price within 14 days of your purchase, we refund the "
                    "difference."
                ),
            ),
            (
                "the limit on price-match requests",
                "Each customer may make up to 3 price-match requests per calendar month.",
            ),
            (
                "the price-match rule for clearance items",
                "Clearance and open-box items are not eligible for price matching.",
            ),
            (
                "the payout method for price-match refunds",
                (
                    "Price-match differences are refunded to the original payment method within 2 "
                    "business days."
                ),
            ),
        ],
    ),
    "rewards_program.md": (
        "Brightlane Rewards",
        [
            (
                "the points earning rate for Rewards members",
                (
                    "Brightlane Rewards members earn 2 points per dollar spent; Gold members earn 3 points "
                    "per dollar."
                ),
            ),
            ("the value of reward points", "Every 100 points is worth $1 off a future order."),
            (
                "the expiration rule for reward points",
                (
                    "Points expire 12 months after they are earned if the account has no purchases in "
                    "that period."
                ),
            ),
            (
                "the requirement for Gold status",
                "Members reach Gold status after spending $1,000 in a calendar year.",
            ),
            (
                "the Gold member birthday perk",
                "Gold members receive a $25 birthday reward, valid for 30 days.",
            ),
            (
                "the rule on earning points for gift cards",
                "Points are not earned on gift card purchases, shipping fees, or taxes.",
            ),
        ],
    ),
    "support_policy.md": (
        "Customer Support",
        [
            (
                "the phone support hours",
                (
                    "Phone support is available Monday through Friday, 8:00 a.m. to 8:00 p.m. Eastern "
                    "Time, and Saturday 9:00 a.m. to 5:00 p.m."
                ),
            ),
            (
                "the live chat response time target",
                ("Live chat aims to connect customers with an agent within 2 minutes during support hours."),
            ),
            (
                "the email support response time",
                "Email tickets receive a first response within 24 hours, including weekends.",
            ),
            (
                "the process for escalating a support ticket",
                (
                    "Customers can request escalation after 2 unresolved contacts; escalated tickets go "
                    "to a senior specialist within 1 business day."
                ),
            ),
            (
                "the order cancellation policy",
                (
                    "Orders can be cancelled free of charge within 1 hour of being placed; after that "
                    "they can only be returned once delivered."
                ),
            ),
            (
                "the identity check for account changes",
                (
                    "Account changes require a one-time code sent to the email on file; agents never ask "
                    "for a full password."
                ),
            ),
        ],
    ),
}

QUESTION_TEMPLATES = [
    "What is {topic}?",
    "Can you tell me {topic}?",
    "Quick question: what's {topic}?",
    "According to Brightlane policy, what is {topic}?",
    "A customer is asking about {topic}. What should I tell them?",
    "I need to confirm {topic}.",
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
            f"# Brightlane {title}\n\nEffective March 1, 2026.\n\n{body}\n", encoding="utf-8"
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
        row["prompt"] = [row["prompt"], row["prompt"].lower(), " " + row["prompt"] + "  "][i % 3]
        lines.append(json.dumps(row))

    facts = [(s, t, a) for s, (_, fs) in DOCS.items() for t, a in fs]
    bad = rng.sample(facts, N_LOW_QUALITY + N_MALFORMED)
    low_quality_answers = [
        "I'm sorry, I don't have that information.",
        "TBD",
        "N/A",
        "As an AI, I can't access company policies.",
        "TODO",
    ]
    for (source, topic, _), answer in zip(bad, low_quality_answers):
        lines.append(json.dumps(_row(RESERVED_TEMPLATE.format(topic=topic), answer, source)))
    for i, (source, topic, _) in enumerate(bad[N_LOW_QUALITY:]):
        question = RESERVED_TEMPLATE.format(topic=topic)
        lines.append(
            [
                json.dumps({"prompt": question, "source": source}),
                json.dumps(_row(question, "", source)),
                json.dumps(_row(question, "See policy.", source))[:40],
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
