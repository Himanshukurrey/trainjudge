"""Healthcare and life sciences."""

from trainjudge.domains import DomainPack

PACK = DomainPack(
    name="healthcare",
    label="Healthcare",
    description="Healthcare, clinical care and life sciences",
    terms=(
        "patient",
        "patients",
        "clinical",
        "clinician",
        "clinicians",
        "diagnosis",
        "diagnoses",
        "symptom",
        "symptoms",
        "medication",
        "medications",
        "drug",
        "drugs",
        "dosage",
        "dosing",
        "prescription",
        "prescriptions",
        "hospital",
        "ehr",
        "emr",
        "icd-10",
        "cpt",
        "snomed",
        "radiology",
        "discharge summary",
        "lab results",
        "nurse",
        "nurses",
        "physician",
        "pharmacy",
        "medical",
        "healthcare",
        "health record",
        "health records",
        "hipaa",
    ),
    changing_fact_terms=(
        "dosage",
        "dosing",
        "drug interactions",
        "formulary",
        "guidelines",
        "clinical guidelines",
        "treatment guidelines",
        "protocols",
        "drug label",
        "drug labels",
        "coverage",
        "reimbursement",
        "vaccine schedule",
    ),
    changing_facts="updated clinical guidelines, drug labels, formularies and coverage rules",
    changing_fact_note=(
        "change as guidelines and drug labels are revised. Serve them from versioned, citable "
        "sources, and keep a clinician in the loop for anything patient-facing."
    ),
    retrieval_hint=" (cite the guideline or label version for every answer)",
    high_stakes_terms=(
        "diagnose",
        "triage",
        "prescribe",
        "prescribing",
        "treatment decision",
        "treatment decisions",
        "clinical decision",
        "clinical decisions",
        "discharge decision",
        "prior authorization",
        "deny coverage",
    ),
    high_stakes_note=(
        "Keep a qualified clinician in the loop: the model should support clinical decisions, "
        "not make them. Check performance across patient groups, and note that software like "
        "this can fall under medical-device rules in some jurisdictions."
    ),
    closing_notes=(
        "If the data contains patient records, treat the dataset and run folder as protected "
        "health information: names, dates and record numbers identify people even without an SSN.",
    ),
)
