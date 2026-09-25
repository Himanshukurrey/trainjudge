"""Every committed demo dataset: known injected issues, and the diagnosis it should get."""

from pathlib import Path

import pytest

from trainjudge.dataset_audit import audit_dataset
from trainjudge.diagnosis import FORMAT, KNOWLEDGE, diagnose

DEMO = Path(__file__).parent.parent / "demo"

FINETUNE = {"clean": 600, "duplicate": 36, "low_quality": 18, "malformed": 12}
RETRIEVAL = {"clean": 72, "duplicate": 4, "low_quality": 2, "malformed": 2}

# (path, audit counts, planted PII, goal, expected classification, expected domain)
DEMOS = [
    ("sql_generation", {"clean": 1300, "duplicate": 330, "low_quality": 125, "malformed": 75}, {},
     "improve SQL generation for our shop database", FORMAT, "ecommerce"),
    ("policy_docs", {"clean": 216, "duplicate": 12, "low_quality": 5, "malformed": 3}, {},
     "make it answer from our internal support policy documents", KNOWLEDGE, "customer_support"),
    ("domains/healthcare/clinical_coding", FINETUNE, {"medical record number": 3, "date of birth": 3},
     "extract diagnoses and ICD-10 codes from clinical notes into our JSON format", FORMAT, "healthcare"),
    ("domains/healthcare/formulary_faq", RETRIEVAL, {},
     "answer staff questions about medication dosing from our hospital formulary", KNOWLEDGE, "healthcare"),
    ("domains/legal/clause_extraction", FINETUNE, {"email address": 6},
     "classify contract clauses and extract key terms into our JSON schema", FORMAT, "legal"),
    ("domains/legal/statutes_faq", RETRIEVAL, {},
     "answer questions about the statutes and filing rules in our jurisdiction", KNOWLEDGE, "legal"),
    ("domains/ecommerce/product_attributes", FINETUNE, {},
     "extract product attributes from listing titles into our catalog JSON", FORMAT, "ecommerce"),
    ("domains/ecommerce/catalog_faq", RETRIEVAL, {},
     "answer shopper questions about our product prices, stock and current promotions",
     KNOWLEDGE, "ecommerce"),
    ("domains/customer_support/ticket_triage", FINETUNE, {"phone number": 4, "email address": 2},
     "triage support tickets into category, priority and team as JSON", FORMAT, "customer_support"),
    ("domains/customer_support/help_center_faq", RETRIEVAL, {},
     "answer customer questions from our help center articles", KNOWLEDGE, "customer_support"),
    ("domains/hr/resume_parsing", FINETUNE, {"date of birth": 3, "email address": 3},
     "parse resumes into our candidate JSON fields", FORMAT, "hr"),
    ("domains/hr/benefits_faq", RETRIEVAL, {},
     "answer employee questions about benefits and the leave policy", KNOWLEDGE, "hr"),
    ("domains/bfsi/transactions", {"clean": 600, "duplicate": 36, "low_quality": 12, "malformed": 8},
     {"card number": 4, "Aadhaar number": 4, "PAN": 3, "UPI ID": 3, "phone number": 3},
     "categorize bank transaction narrations into our category JSON", FORMAT, "bfsi"),
    ("domains/bfsi/loan_faq", {"clean": 144, "duplicate": 8, "low_quality": 3, "malformed": 2}, {},
     "answer customer questions about our loan and FD interest rates and charges", KNOWLEDGE, "bfsi"),
    ("domains/education/question_tagging", FINETUNE, {},
     "tag exam questions with subject, topic and difficulty in our JSON format", FORMAT, "education"),
    ("domains/education/course_faq", RETRIEVAL, {},
     "answer student questions about exam dates, deadlines and course policies", KNOWLEDGE, "education"),
]  # fmt: skip
IDS = [d[0] for d in DEMOS]


@pytest.mark.parametrize("path, counts, pii, goal, classification, domain", DEMOS, ids=IDS)
def test_demo_audit(path, counts, pii, goal, classification, domain):
    report = audit_dataset(DEMO / path / "data.jsonl")
    assert report.format == "completions"
    assert report.counts() == counts
    assert report.conflicting_prompts == 0
    assert {k: len(v) for k, v in report.sensitive.items()} == pii


@pytest.mark.parametrize("path, counts, pii, goal, classification, domain", DEMOS, ids=IDS)
def test_demo_diagnosis(path, counts, pii, goal, classification, domain):
    d = diagnose(goal, audit_dataset(DEMO / path / "data.jsonl"), model="Qwen3-0.6B")
    assert d.classification == classification
    assert d.domain is not None and d.domain.name == domain


def test_every_domain_pack_has_both_demos():
    from trainjudge import domains

    for name in domains.PACKS:
        cases = {c for p, _, _, _, c, dom in DEMOS if dom == name and p.startswith("domains/")}
        assert cases == {FORMAT, KNOWLEDGE}, name


def test_readme_goals_match_the_table():
    for path, _, _, goal, _, _ in DEMOS:
        readme = (DEMO / path / "README.md").read_text(encoding="utf-8")
        assert goal in " ".join(readme.split()), path


def test_sql_demo_matches_plan_mix():
    report = audit_dataset(DEMO / "sql_generation" / "data.jsonl")
    assert report.percentages() == {"clean": 71, "duplicate": 18, "low_quality": 7, "malformed": 4}
