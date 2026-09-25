"""Legal services, contracts and compliance."""

from trainjudge.domains import DomainPack

PACK = DomainPack(
    name="legal",
    label="Legal",
    description="Legal services, contracts, litigation and compliance",
    terms=(
        "contract",
        "contracts",
        "clause",
        "clauses",
        "nda",
        "ndas",
        "statute",
        "statutes",
        "case law",
        "precedent",
        "litigation",
        "plaintiff",
        "defendant",
        "attorney",
        "attorneys",
        "lawyer",
        "lawyers",
        "legal",
        "jurisdiction",
        "indemnity",
        "indemnification",
        "court",
        "judgment",
        "counsel",
        "privilege",
        "e-discovery",
        "redline",
        "redlines",
        "legislation",
    ),
    changing_fact_terms=(
        "case law",
        "statutes",
        "statute",
        "regulations",
        "precedent",
        "precedents",
        "legislation",
        "amendments",
        "rulings",
        "court rulings",
        "compliance requirements",
        "jurisdiction",
    ),
    changing_facts="new rulings, amended statutes and changing regulations",
    changing_fact_note=(
        "change with new rulings and amendments, and differ by jurisdiction. Retrieve them from "
        "dated, citable sources so every answer can be checked, rather than trusting recall."
    ),
    retrieval_hint=" (with the jurisdiction and date on each source)",
    high_stakes_terms=(
        "legal advice",
        "sentencing",
        "bail",
        "parole",
        "immigration decision",
        "immigration decisions",
        "custody",
        "liability determination",
        "eligibility decision",
    ),
    high_stakes_note=(
        "Keep a qualified lawyer in the loop. Automated legal determinations can seriously harm "
        "people, so outputs should be reviewed before anyone relies on them."
    ),
    closing_notes=(
        "Contracts and case files are often confidential or privileged. Check you're permitted to "
        "use them for training, and keep run folders access-controlled.",
    ),
)
