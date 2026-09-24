"""The committed demo datasets contain a known number of injected issues."""

from pathlib import Path

import pytest

from trainjudge.dataset_audit import audit_dataset

DEMO = Path(__file__).parent.parent / "demo"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("sql_generation", {"clean": 1300, "duplicate": 330, "low_quality": 125, "malformed": 75}),
        ("policy_docs", {"clean": 216, "duplicate": 12, "low_quality": 5, "malformed": 3}),
        ("bfsi_transactions", {"clean": 600, "duplicate": 36, "low_quality": 12, "malformed": 8}),
        ("bfsi_loan_faq", {"clean": 144, "duplicate": 8, "low_quality": 3, "malformed": 2}),
    ],
)
def test_demo_audit_counts(name, expected):
    report = audit_dataset(DEMO / name / "data.jsonl")
    assert report.format == "completions"
    assert report.counts() == expected
    assert report.conflicting_prompts == 0


def test_sql_demo_matches_plan_mix():
    report = audit_dataset(DEMO / "sql_generation" / "data.jsonl")
    assert report.percentages() == {"clean": 71, "duplicate": 18, "low_quality": 7, "malformed": 4}


@pytest.mark.parametrize("name", ["sql_generation", "policy_docs", "bfsi_loan_faq"])
def test_demos_without_planted_pii_are_clean(name):
    assert audit_dataset(DEMO / name / "data.jsonl").sensitive == {}


def test_bfsi_transactions_planted_pii():
    sensitive = audit_dataset(DEMO / "bfsi_transactions" / "data.jsonl").sensitive
    assert {k: len(v) for k, v in sensitive.items()} == {
        "card number": 4,
        "Aadhaar number": 4,
        "PAN": 3,
        "UPI ID": 3,
        "phone number": 3,
    }
