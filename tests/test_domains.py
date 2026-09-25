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


def test_bfsi_is_registered():
    assert "bfsi" in domains.PACKS
    assert domains.PACKS["bfsi"].label == "BFSI"


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
        diagnose("improve SQL", audit_dataset(DEMO / "sql_generation" / "data.jsonl")).to_dict()["domain"]
        is None
    )


def test_cli_domain_flag():
    args = ["diagnose", "--dataset", str(DEMO / "bfsi_loan_faq" / "data.jsonl"), "--model", "m",
            "--goal", "answer questions about our loan interest rates", "--json"]  # fmt: skip
    runner = CliRunner()
    assert json.loads(runner.invoke(main, args).output)["domain"]["name"] == "bfsi"
    assert json.loads(runner.invoke(main, [*args, "--domain", "none"]).output)["domain"] is None
    assert runner.invoke(main, [*args, "--domain", "astrology"]).exit_code != 0
