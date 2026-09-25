"""HR, recruiting and people operations."""

from trainjudge.domains import DomainPack

PACK = DomainPack(
    name="hr",
    label="HR/recruiting",
    description="Recruiting, hiring and people operations",
    terms=(
        "resume",
        "resumes",
        "cv",
        "cvs",
        "candidate",
        "candidates",
        "applicant",
        "applicants",
        "job description",
        "job descriptions",
        "recruiting",
        "recruitment",
        "recruiter",
        "hiring",
        "interview",
        "interviews",
        "employee",
        "employees",
        "payroll",
        "onboarding",
        "performance review",
        "performance reviews",
        "human resources",
        "talent acquisition",
        "job posting",
        "job postings",
        "offer letter",
    ),
    changing_fact_terms=(
        "benefits",
        "leave policy",
        "pay scales",
        "pay bands",
        "salary bands",
        "labor law",
        "labour law",
        "employment law",
        "holidays",
        "handbook",
        "employee handbook",
        "policies",
    ),
    changing_facts="benefit changes, updated pay bands, handbook revisions and employment law",
    changing_fact_note=(
        "change with each policy revision and differ by country. Serve them from the current "
        "handbook and policy documents rather than baking them into weights."
    ),
    high_stakes_terms=(
        "screen candidates",
        "screening",
        "shortlist",
        "shortlisting",
        "reject candidates",
        "rank candidates",
        "ranking candidates",
        "hiring decision",
        "hiring decisions",
        "promotion decision",
        "termination",
        "terminations",
        "performance rating",
        "salary decision",
    ),
    high_stakes_note=(
        "Automated screening and ranking of people is a known bias risk and is regulated in some "
        "places (for example, the EU AI Act treats hiring as high-risk, and New York City "
        "requires bias audits of automated hiring tools). Keep a human decision-maker and audit "
        "outcomes across demographic groups."
    ),
    closing_notes=(
        "Resumes and HR records are personal data. Remove names, contact details and protected "
        "attributes before training where you can.",
    ),
)
