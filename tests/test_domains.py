import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainjudge import domains
from trainjudge.cli import main
from trainjudge.dataset_audit import audit_dataset
from trainjudge.diagnosis import KNOWLEDGE, diagnose, format_diagnosis
from trainjudge.domains import DomainPack

DEMO = Path(__file__).parent.parent / "demo"

HEALTHCARE = DomainPack(
    name="healthcare",
    label="Healthcare",
    description="test pack",
    terms=("patient", "diagnosis", "clinical", "dosage", "hospital"),
    changing_fact_terms=("dosage", "guidelines"),
    changing_facts="updated clinical guidelines and drug labels",
    high_stakes_terms=("triage",),
)


def test_all_packs_are_registered():
    assert set(domains.PACKS) == {
        "bfsi", "healthcare", "legal", "ecommerce", "customer_support", "hr", "education",
    }  # fmt: skip
    for pack in domains.PACKS.values():
        assert pack.terms and pack.changing_fact_terms and pack.high_stakes_terms
        assert pack.label and pack.description and pack.changing_facts


def _qa_dataset(tmp_path, rows):
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return audit_dataset(path)


# One realistic goal per pack: (pack, goal, expected changing-fact and high-stakes terms).
DOMAIN_GOALS = [
    ("bfsi", "answer customer questions about our home loan interest rates", ["interest rates"], []),
    ("healthcare", "answer patient questions about medication dosing from the clinical guidelines",
     ["dosing", "guidelines", "clinical guidelines"], []),
    ("healthcare", "triage patients in the emergency department", [], ["triage"]),
    ("legal", "answer questions about case law and statutes in our jurisdiction",
     ["case law", "statutes", "jurisdiction"], []),
    ("legal", "extract indemnity clauses from contracts", [], []),
    ("ecommerce", "answer shopping questions about prices and stock for our products",
     ["prices", "stock"], []),
    ("customer_support", "answer support tickets using our help center articles", ["help center"], []),
    ("hr", "screen candidates and rank resumes for hiring", [],
     ["screen candidates"]),
    ("hr", "answer employee questions about benefits and the leave policy", ["benefits", "leave policy"], []),
    ("education", "answer student questions about exam dates and the syllabus",
     ["syllabus", "exam dates"], []),
    ("education", "grade essays for our university course", [], ["grade essays"]),
]  # fmt: skip


@pytest.mark.parametrize("name, goal, changing, high_stakes", DOMAIN_GOALS)
def test_each_pack_is_detected_from_a_realistic_goal(name, goal, changing, high_stakes):
    m = domains.detect(goal, "")
    assert m is not None and m.pack.name == name
    assert set(changing) <= set(m.changing_fact_terms)
    assert set(high_stakes) <= set(m.high_stakes_terms)


def test_knowledge_goal_in_a_new_domain_gets_domain_advice(tmp_path):
    rows = [
        {
            "prompt": f"What is the adult dose of drug {i}?",
            "completion": f"The adult dose is {i * 5} mg twice daily.",
        }
        for i in range(1, 60)
    ]
    d = diagnose(
        "answer patient questions about medication dosing from the clinical guidelines",
        _qa_dataset(tmp_path, rows),
    )
    assert d.domain.name == "healthcare" and d.classification == KNOWLEDGE
    text = " ".join(format_diagnosis(d).split())
    assert "Healthcare checks:" in text
    assert "updated clinical guidelines, drug labels" in text
    assert "protected health information" in text


def test_high_stakes_hr_goal_warns_about_bias(tmp_path):
    rows = [
        {"prompt": f"Resume {i}: 5 years Python", "completion": "shortlist" if i % 2 else "reject"}
        for i in range(80)
    ]
    d = diagnose("screen candidates and shortlist resumes for hiring", _qa_dataset(tmp_path, rows))
    assert d.domain.name == "hr"
    text = " ".join(format_diagnosis(d).split())
    assert "EU AI Act" in text and "bias audits" in text


def test_generic_goal_has_no_domain(tmp_path):
    rows = [{"prompt": f"Convert {i} to JSON", "completion": json.dumps({"n": i})} for i in range(80)]
    assert diagnose("return valid JSON for each input", _qa_dataset(tmp_path, rows)).domain is None


def test_detect_from_goal_or_enough_data_terms():
    assert domains.detect("improve our loan FAQ bot", "").pack.name == "bfsi"
    # One data term isn't enough on its own; three distinct ones are.
    assert domains.detect("improve answers", "the claim was filed") is None
    assert domains.detect("improve answers", "loan emi kyc").pack.name == "bfsi"


def test_detect_none_and_forced():
    assert domains.detect("loan decisions", "", "none") is None
    assert domains.detect("sql", "", "bfsi").pack.name == "bfsi"
    with pytest.raises(ValueError, match="unknown domain"):
        domains.detect("x", "", "astrology")


def test_a_new_pack_plugs_in_without_code_changes(monkeypatch, tmp_path):
    monkeypatch.setitem(domains.PACKS, "healthcare", HEALTHCARE)
    rows = [
        {"prompt": f"What is the adult dosage of drug {i}?", "completion": f"The adult dosage is {i * 5} mg."}
        for i in range(1, 60)
    ]
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    d = diagnose("answer patient dosage questions from the clinical guidelines", audit_dataset(path))
    assert d.domain.name == "healthcare"
    assert d.classification == KNOWLEDGE
    assert d.changing_fact_terms == ["dosage", "guidelines"]
    text = " ".join(format_diagnosis(d).split())
    assert "Healthcare checks:" in text
    assert "updated clinical guidelines and drug labels" in text


def test_goal_match_beats_data_match(monkeypatch):
    monkeypatch.setitem(domains.PACKS, "healthcare", HEALTHCARE)
    m = domains.detect("triage hospital patients", "loan emi kyc upi neft")
    assert m.pack.name == "healthcare"
    assert m.high_stakes_terms == ["triage"]


def test_json_output_has_domain():
    report = audit_dataset(DEMO / "bfsi_loan_faq" / "data.jsonl")
    data = diagnose("answer questions about our loan interest rates", report).to_dict()
    assert data["domain"]["name"] == "bfsi"
    assert "interest rates" in data["domain"]["changing_fact_terms"]
    assert (
        diagnose(
            "improve SQL", audit_dataset(DEMO / "sql_generation" / "data.jsonl"), domain="none"
        ).to_dict()["domain"]
        is None
    )


def test_cli_domain_flag():
    args = ["diagnose", "--dataset", str(DEMO / "bfsi_loan_faq" / "data.jsonl"), "--model", "m",
            "--goal", "answer questions about our loan interest rates", "--json"]  # fmt: skip
    runner = CliRunner()
    assert json.loads(runner.invoke(main, args).output)["domain"]["name"] == "bfsi"
    assert json.loads(runner.invoke(main, [*args, "--domain", "none"]).output)["domain"] is None
    assert runner.invoke(main, [*args, "--domain", "astrology"]).exit_code != 0
