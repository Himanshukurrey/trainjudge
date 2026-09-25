"""Generate the per-domain demo datasets (deterministic).

Every domain gets two demos:

- a "fine-tune" case: a structured-output task (text -> JSON) that fits
  fine-tuning, with some domain-typical sensitive identifiers planted in it
- a "retrieval" case: Q&A that recalls facts from the domain's documents,
  facts that change, so diagnosis should recommend retrieval instead

All organizations, people, drugs, statutes and products are fictional. Each
dataset has a known number of duplicate, low-quality and malformed rows so
`trainjudge audit` has something to find. The BFSI demos live in bfsi/ with
their own generators.

    python demo/domains/generate.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260925

N_DUPLICATE = {"finetune": 36, "retrieval": 4}
N_LOW_QUALITY = {"finetune": 18, "retrieval": 2}
N_MALFORMED = {"finetune": 12, "retrieval": 2}

QUESTION_TEMPLATES = [
    "What is {topic}?",
    "Can you tell me {topic}?",
    "Quick question: what's {topic}?",
    "According to current policy, what is {topic}?",
    "Someone is asking about {topic}. What should I tell them?",
    "Please confirm {topic}.",
]
RESERVED_TEMPLATE = "Hi, what's {topic}?"  # only used for injected bad rows


@dataclass
class FinetuneCase:
    domain: str
    name: str
    title: str
    goal: str
    instruction: str
    make_example: object  # (rng) -> (input_text, output_dict)
    make_pii_example: object  # (rng, i) -> (input_text, output_dict)
    # 600 clean rows -> a ~60-example held-out test split, enough for a real gain to be significant.
    n_clean: int = 600
    n_pii: int = 6
    pii_kinds: dict[str, int] = field(default_factory=dict)


@dataclass
class RetrievalCase:
    domain: str
    name: str
    title: str
    goal: str
    org: str
    docs: dict[str, tuple[str, list[tuple[str, str]]]]


# --- healthcare ----------------------------------------------------------------

CONDITIONS = [
    ("community-acquired pneumonia", "J18.9", "cough, fever and crackles at the right lung base"),
    ("type 2 diabetes mellitus without complications", "E11.9", "raised HbA1c on routine labs"),
    ("essential hypertension", "I10", "repeated blood pressure readings above 150/95"),
    ("acute bronchitis", "J20.9", "a productive cough for ten days without fever"),
    ("urinary tract infection", "N39.0", "dysuria and urinary frequency"),
    ("migraine", "G43.909", "recurrent unilateral headaches with photophobia"),
    ("asthma", "J45.909", "wheeze and shortness of breath on exertion"),
    ("hyperlipidemia", "E78.5", "elevated LDL cholesterol"),
    ("gastroesophageal reflux disease", "K21.9", "heartburn after meals"),
    ("low back pain", "M54.50", "lower back pain after lifting"),
    ("iron deficiency anemia", "D50.9", "fatigue and low ferritin"),
    ("hypothyroidism", "E03.9", "fatigue, weight gain and raised TSH"),
]
SEVERITIES = ["mild", "moderate", "severe"]
PLANS = [
    "follow up in two weeks",
    "start treatment and recheck",
    "refer to specialist",
    "lifestyle advice and review",
    "repeat labs in six weeks",
]


def clinical_note(rng: random.Random) -> tuple[str, dict]:
    diagnosis, code, finding = rng.choice(CONDITIONS)
    severity = rng.choice(SEVERITIES)
    age, sex = rng.randint(19, 88), rng.choice(["male", "female"])
    note = (
        f"{age}-year-old {sex} presents with {finding}. Assessment: {severity} {diagnosis}. "
        f"Plan: {rng.choice(PLANS)}."
    )
    return note, {"diagnosis": diagnosis, "icd10": code, "severity": severity}


def clinical_note_with_pii(rng: random.Random, i: int) -> tuple[str, dict]:
    note, out = clinical_note(rng)
    if i % 2:
        return f"MRN: {rng.randint(10_000_000, 99_999_999)}. {note}", out
    return f"DOB: {rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1940, 2004)}. {note}", out


# --- legal ---------------------------------------------------------------------

PARTIES = [
    "Harbor Logistics LLC",
    "Pinecrest Software Inc.",
    "Meridian Foods Ltd.",
    "Bluewater Analytics LLC",
    "Oakridge Builders Inc.",
    "Silverline Media LLC",
    "Crestview Health Partners",
    "Ironbridge Manufacturing Co.",
    "Lumen Retail Group",
    "Redwood Energy Ltd.",
]
CLAUSES = {
    "termination": [
        "Either party may terminate this Agreement on {months} months' written notice to {party}.",
        "{party} may end this Agreement for convenience by giving {months} months' notice.",
    ],
    "confidentiality": [
        "{party} shall keep all Confidential Information secret for {months} months after disclosure.",
        "For {months} months after termination, {party} must not disclose Confidential Information.",
    ],
    "non_compete": [
        "{party} shall not engage in a competing business for {months} months after the Effective Date.",
        "For a period of {months} months, {party} will not solicit the other party's customers.",
    ],
    "indemnification": [
        "{party} shall indemnify the other party against all third-party claims arising from its breach.",
        "{party} will defend and hold harmless the other party from losses caused by its negligence.",
    ],
    "payment_terms": [
        "{party} shall pay each invoice within {days} days of receipt.",
        "Invoices are payable by {party} net {days} days, with interest on late amounts.",
    ],
    "governing_law": [
        "This Agreement is governed by the laws of the State of Westmark, and {party} submits to its courts.",
        "{party} agrees that disputes will be resolved exclusively in the courts of Westmark.",
    ],
}  # fmt: skip


def contract_clause(rng: random.Random) -> tuple[str, dict]:
    kind = rng.choice(list(CLAUSES))
    party = rng.choice(PARTIES)
    months = rng.choice([1, 2, 3, 6, 9, 12, 18, 24, 30, 36, 48, 60])
    text = rng.choice(CLAUSES[kind]).format(party=party, months=months, days=rng.choice([15, 30, 45, 60, 90]))
    duration = months if kind in ("termination", "confidentiality", "non_compete") else None
    return text, {"clause_type": kind, "party": party, "duration_months": duration}


def contract_clause_with_pii(rng: random.Random, i: int) -> tuple[str, dict]:
    party = rng.choice(PARTIES)
    user = rng.choice(["legal", "notices", "counsel", "contracts"])
    domain = party.split()[0].lower()
    text = (
        f"All notices under this Agreement shall be sent to {party} at {user}@{domain}-corp.fake "
        f"and are effective on receipt."
    )
    return text, {"clause_type": "notices", "party": party, "duration_months": None}


# --- e-commerce ----------------------------------------------------------------

BRANDS = ["Trailnest", "Kindlework", "Northpeak", "Softloom", "Brightfield", "Urbanweave"]
CATEGORIES = {
    "jacket": ["S", "M", "L", "XL"], "sneakers": ["7", "8", "9", "10", "11"], "backpack": ["20L", "30L"],
    "t-shirt": ["S", "M", "L"], "hoodie": ["M", "L", "XL"], "water bottle": ["500ml", "750ml", "1L"],
}  # fmt: skip
COLORS = ["black", "navy", "olive", "red", "grey", "white", "sand"]
TITLE_FORMS = [
    "{brand} {color} {category} - size {size}",
    "{brand} {category}, {color}, {size}",
    "New! {brand} {category} ({color}) size {size}",
    "{color} {category} by {brand} | {size}",
]


def listing_title(rng: random.Random) -> tuple[str, dict]:
    category = rng.choice(list(CATEGORIES))
    brand, color, size = rng.choice(BRANDS), rng.choice(COLORS), rng.choice(CATEGORIES[category])
    title = rng.choice(TITLE_FORMS).format(brand=brand, color=color, category=category, size=size)
    return title, {"brand": brand, "category": category, "color": color, "size": size}


# --- customer support ----------------------------------------------------------

TICKETS = [
    ("billing", "billing", "I was charged twice for my {plan} plan this month."),
    ("billing", "billing", "Can I get an invoice with our VAT number for the {plan} plan?"),
    ("login", "accounts", "I can't log in after resetting my password on the {plan} plan."),
    ("login", "accounts", "Two-factor codes aren't arriving for my account."),
    ("bug", "engineering", "Exports to CSV fail with an error since the last update."),
    ("bug", "engineering", "The dashboard shows a blank page in Firefox."),
    ("feature_request", "product", "Please add dark mode to the mobile app."),
    ("feature_request", "product", "Could you support SSO for the {plan} plan?"),
    ("cancellation", "retention", "How do I cancel my {plan} subscription?"),
    ("cancellation", "retention", "We want to downgrade from {plan} at renewal."),
]
URGENT = ["URGENT: ", "Our whole team is blocked. ", "Production is down! "]


def support_ticket(rng: random.Random) -> tuple[str, dict]:
    category, team, text = rng.choice(TICKETS)
    text = text.format(plan=rng.choice(["Starter", "Team", "Business", "Enterprise"]))
    urgent = rng.random() < 0.3
    if urgent:
        text = rng.choice(URGENT) + text
    suffix = rng.choice(["", " Thanks!", " Any help appreciated.", f" (ticket #{rng.randint(1000, 9999)})"])
    priority = (
        "high" if urgent else rng.choice(["normal", "low"]) if category == "feature_request" else "normal"
    )
    return text + suffix, {"category": category, "priority": priority, "team": team}


def support_ticket_with_pii(rng: random.Random, i: int) -> tuple[str, dict]:
    text, out = support_ticket(rng)
    contact = [
        f" You can reach me at {rng.choice(['sam', 'priya', 'lee', 'ana'])}.{rng.randint(10, 99)}@mailhub.fake.",
        " Call me on +44 20 7946 0958.",
        f" My number is (415) 555-01{rng.randint(10, 99)}.",
    ][i % 3]
    return text + contact, out


# --- HR ------------------------------------------------------------------------

TITLES = [
    "data analyst",
    "backend engineer",
    "product designer",
    "account executive",
    "HR generalist",
    "QA engineer",
    "marketing manager",
]
SKILLS = ["SQL", "Python", "Figma", "Salesforce", "Kubernetes", "Excel", "SEO", "Selenium"]
DEGREES = ["BSc Computer Science", "BA Economics", "MBA", "BDes Interaction Design", "MSc Statistics", "none"]


def resume_summary(rng: random.Random) -> tuple[str, dict]:
    title, skill, degree = rng.choice(TITLES), rng.choice(SKILLS), rng.choice(DEGREES)
    years = rng.randint(1, 15)
    edu = f"Degree: {degree}." if degree != "none" else "No formal degree; self-taught."
    text = rng.choice([
        f"{title.title()} with {years} years of experience, strongest in {skill}. {edu}",
        f"Currently working as a {title}. {years}+ years in industry. Key skill: {skill}. {edu}",
        f"{years} yrs experience | {title} | {skill} | {edu}",
    ])  # fmt: skip
    return text, {"current_title": title, "years_experience": years, "top_skill": skill, "degree": degree}


def resume_summary_with_pii(rng: random.Random, i: int) -> tuple[str, dict]:
    text, out = resume_summary(rng)
    if i % 2:
        return f"{text} Email: {rng.choice(['jordan', 'mei', 'arjun', 'zoe'])}.cv@mailhub.fake", out
    return f"{text} DOB: {rng.randint(1970, 2002)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", out


# --- education -----------------------------------------------------------------

QUESTIONS = [
    ("mathematics", "algebra", "Solve for x: {a}x + {b} = {c}."),
    ("mathematics", "geometry", "Find the area of a triangle with base {a} cm and height {b} cm."),
    ("mathematics", "calculus", "Differentiate f(x) = {a}x^3 + {b}x with respect to x."),
    ("physics", "mechanics", "A car accelerates from rest at {a} m/s^2 for {b} s. What is its final speed?"),
    ("physics", "electricity", "What current flows through a {a} ohm resistor at {b} V?"),
    ("biology", "cell biology", "Name the organelle responsible for producing ATP and describe its role."),
    ("biology", "genetics", "Explain how a recessive trait can skip a generation, using a Punnett square."),
    ("history", "world war II", "Explain two causes of the Second World War."),
    ("history", "industrial revolution", "Describe one social effect of the Industrial Revolution."),
]
DIFFICULTY = {"algebra": "easy", "geometry": "easy", "calculus": "hard", "mechanics": "medium",
              "electricity": "easy", "cell biology": "easy", "genetics": "medium", "world war II": "medium",
              "industrial revolution": "medium"}  # fmt: skip


def exam_question(rng: random.Random) -> tuple[str, dict]:
    subject, topic, text = rng.choice(QUESTIONS)
    marks = rng.choice([2, 3, 4, 5, 6])
    text = (
        text.format(a=rng.randint(2, 12), b=rng.randint(2, 20), c=rng.randint(20, 90)) + f" [{marks} marks]"
    )
    return text, {"subject": subject, "topic": topic, "difficulty": DIFFICULTY[topic]}


def no_pii(rng: random.Random, i: int) -> tuple[str, dict]:
    raise AssertionError("this case has no planted PII")


FINETUNE_CASES = [
    FinetuneCase("healthcare", "clinical_coding", "Clinical note coding",
                 "extract diagnoses and ICD-10 codes from clinical notes into our JSON format",
                 "Extract the primary diagnosis and ICD-10 code from this clinical note. "
                 "Reply with JSON: diagnosis, icd10, severity.",
                 clinical_note, clinical_note_with_pii, pii_kinds={"medical record number": 3, "date of birth": 3}),
    FinetuneCase("legal", "clause_extraction", "Contract clause extraction",
                 "classify contract clauses and extract key terms into our JSON schema",
                 "Classify this contract clause and extract its key terms. "
                 "Reply with JSON: clause_type, party, duration_months.",
                 contract_clause, contract_clause_with_pii, pii_kinds={"email address": 6}),
    FinetuneCase("ecommerce", "product_attributes", "Product attribute extraction",
                 "extract product attributes from listing titles into our catalog JSON",
                 "Extract product attributes from this listing title. Reply with JSON: brand, category, color, size.",
                 listing_title, no_pii, n_pii=0),
    FinetuneCase("customer_support", "ticket_triage", "Support ticket triage",
                 "triage support tickets into category, priority and team as JSON",
                 "Triage this support ticket. Reply with JSON: category, priority, team.",
                 support_ticket, support_ticket_with_pii, pii_kinds={"email address": 2, "phone number": 4}),
    FinetuneCase("hr", "resume_parsing", "Resume parsing",
                 "parse resumes into our candidate JSON fields",
                 "Extract structured fields from this resume summary. "
                 "Reply with JSON: current_title, years_experience, top_skill, degree.",
                 resume_summary, resume_summary_with_pii, pii_kinds={"email address": 3, "date of birth": 3}),
    FinetuneCase("education", "question_tagging", "Exam question tagging",
                 "tag exam questions with subject, topic and difficulty in our JSON format",
                 "Tag this exam question. Reply with JSON: subject, topic, difficulty.",
                 exam_question, no_pii, n_pii=0),
]  # fmt: skip

RETRIEVAL_CASES = [
    RetrievalCase("healthcare", "formulary_faq", "Hospital formulary Q&A",
                  "answer staff questions about medication dosing from our hospital formulary",
                  "Riverside General Hospital", {
        "formulary_analgesics.md": ("Formulary: Analgesics", [
            ("the adult dose of Zolvarin", "Adults take 20 mg of Zolvarin once daily; the formulary limit is 40 mg a day."),
            ("the maximum daily dose of Pentacor", "Pentacor is capped at 3 g per day across all doses, 2 g for patients over 75."),
            ("the dosing interval for Relvix", "Relvix is given every 8 hours, with at least 6 hours between doses."),
            ("the pediatric Zolvarin rule", "Children aged 6 to 12 take 0.3 mg/kg of Zolvarin, up to 10 mg a day."),
        ]),
        "formulary_antibiotics.md": ("Formulary: Antibiotics", [
            ("the first-line antibiotic for community pneumonia", "First-line is Amoxatrin 1 g three times daily for 5 days."),
            ("the Cefalyn renal adjustment", "Halve the Cefalyn dose when eGFR falls below 30 mL/min."),
            ("the restricted antibiotics list", "Meropax and Linezorin need infectious-disease approval within 24 hours."),
            ("the IV-to-oral switch rule", "Switch to oral after 48 hours if the patient is afebrile and eating."),
        ]),
        "formulary_policy.md": ("Formulary Policy", [
            ("the formulary review cycle", "The formulary committee reviews every drug class every 12 months."),
            ("the non-formulary request process", "Non-formulary requests need a pharmacist review within 2 working days."),
            ("the high-alert medication rule", "High-alert medications need an independent double check by 2 nurses."),
            ("the generic substitution policy", "Pharmacy substitutes generics automatically unless the order says 'no substitution'."),
        ]),
    }),
    RetrievalCase("legal", "statutes_faq", "Statutes and filing rules Q&A",
                  "answer questions about the statutes and filing rules in our jurisdiction",
                  "State of Westmark (fictional)", {
        "limitation_periods.md": ("Limitation Periods", [
            ("the limitation period for contract claims", "Contract claims must be filed within 6 years of the breach under Westmark Civil Code §4-210."),
            ("the limitation period for personal injury", "Personal injury claims must be filed within 3 years of the injury."),
            ("the limitation period for defamation", "Defamation claims must be filed within 1 year of publication."),
            ("the tolling rule for minors", "The limitation period is paused until the claimant turns 18."),
        ]),
        "court_fees.md": ("Court Fees", [
            ("the filing fee for a civil complaint", "Filing a civil complaint in Westmark District Court costs $435."),
            ("the small claims limit", "Small claims court handles disputes up to $12,500."),
            ("the appeal filing fee", "Filing an appeal costs $505, due within 30 days of judgment."),
            ("the fee waiver threshold", "Fees are waived for filers earning under 150% of the poverty line."),
        ]),
        "procedure.md": ("Civil Procedure", [
            ("the deadline to answer a complaint", "Defendants must answer within 21 days of being served."),
            ("the page limit for motions", "Motions are limited to 25 pages unless the court allows more."),
            ("the notice period for depositions", "Depositions need at least 14 days' written notice."),
            ("the electronic filing rule", "Represented parties must file electronically; paper filings are refused."),
        ]),
    }),
    RetrievalCase("ecommerce", "catalog_faq", "Prices, stock and promotions Q&A",
                  "answer shopper questions about our product prices, stock and current promotions",
                  "Trailnest Outdoor Store", {
        "prices.md": ("Current Prices", [
            ("the price of the Trailnest Ridge jacket", "The Ridge jacket is $189, or $159 for members."),
            ("the price of the Summit 30L backpack", "The Summit 30L backpack costs $129."),
            ("the price of the Trek sneakers", "Trek sneakers are $110; wide sizes cost $10 more."),
            ("the price of the insulated bottle", "The 750ml insulated bottle is $34."),
        ]),
        "stock.md": ("Stock and Restocks", [
            ("the stock status of the Ridge jacket", "The Ridge jacket is in stock in S to L; XL restocks on October 14."),
            ("the restock date for Trek sneakers in size 11", "Size 11 Trek sneakers restock on October 3."),
            ("the store pickup availability", "Store pickup is available at 12 locations within 2 hours of ordering."),
            ("the backorder policy", "Backorders ship within 21 days or are refunded automatically."),
        ]),
        "promotions.md": ("Promotions", [
            ("the current autumn sale", "The autumn sale takes 20% off outerwear until October 31."),
            ("the free shipping promotion", "Shipping is free on orders over $60 until the end of the month."),
            ("the bundle discount", "Buy any backpack and bottle together and save $15."),
            ("the member points promotion", "Members earn triple points on footwear through October 20."),
        ]),
    }),
    RetrievalCase("customer_support", "help_center_faq", "Help center Q&A",
                  "answer customer questions from our help center articles",
                  "Quillbox (fictional SaaS)", {
        "plans.md": ("Plans and Pricing", [
            ("the price of the Team plan", "The Team plan costs $12 per user per month, billed annually."),
            ("the seat limit on the Starter plan", "The Starter plan includes up to 5 seats."),
            ("the storage included in the Business plan", "The Business plan includes 1 TB of storage per workspace."),
            ("the free trial length", "Every paid plan starts with a 14-day free trial."),
        ]),
        "support_policy.md": ("Support Policy", [
            ("the support response time for Business customers", "Business customers get a first response within 4 business hours."),
            ("the support hours", "Chat support is open 7:00 to 19:00 UTC, Monday to Friday."),
            ("the refund window", "Annual plans can be refunded in full within 30 days of purchase."),
            ("the uptime commitment", "Quillbox commits to 99.9% monthly uptime on the Business plan."),
        ]),
        "data_policy.md": ("Data and Security", [
            ("the data retention period after cancellation", "Workspace data is kept for 60 days after cancellation, then deleted."),
            ("the export formats", "Workspaces can be exported as JSON or CSV at any time."),
            ("the data residency options", "Business and Enterprise plans can store data in the EU or the US."),
            ("the SSO availability", "SAML SSO is available on the Business plan and above."),
        ]),
    }),
    RetrievalCase("hr", "benefits_faq", "Benefits and leave Q&A",
                  "answer employee questions about benefits and the leave policy",
                  "Cobaltworks (fictional)", {
        "leave.md": ("Leave Policy", [
            ("the annual leave allowance", "Full-time employees get 25 days of annual leave plus public holidays."),
            ("the parental leave policy", "Parents get 20 weeks of fully paid parental leave in the first year."),
            ("the sick leave allowance", "Employees get 10 paid sick days a year; a note is needed after 3 days in a row."),
            ("the carry-over rule for annual leave", "Up to 5 unused leave days carry over and expire on March 31."),
        ]),
        "benefits.md": ("Benefits", [
            ("the health plan enrollment window", "Health plan enrollment is open November 1 to 15 each year."),
            ("the retirement contribution match", "The company matches retirement contributions up to 5% of salary."),
            ("the remote work stipend", "Remote employees get a $600 home-office stipend once a year."),
            ("the learning budget", "Each employee has a $1,500 annual learning budget."),
        ]),
        "pay.md": ("Pay and Reviews", [
            ("the pay review cycle", "Pay is reviewed every April, effective from May 1."),
            ("the payday", "Salaries are paid on the 25th of each month."),
            ("the overtime rule", "Overtime above 40 hours a week is paid at 1.5 times the hourly rate."),
            ("the referral bonus", "A successful employee referral earns a $2,000 bonus after 90 days."),
        ]),
    }),
    RetrievalCase("education", "course_faq", "Exam dates and course policy Q&A",
                  "answer student questions about exam dates, deadlines and course policies",
                  "Westbrook University (fictional)", {
        "exams.md": ("Exam Timetable", [
            ("the date of the Calculus I final exam", "The Calculus I final is on December 12 at 09:00 in Hall B."),
            ("the date of the Physics 101 midterm", "The Physics 101 midterm is on October 22 at 14:00."),
            ("the resit exam period", "Resit exams run from January 8 to 19."),
            ("the exam results release date", "Autumn exam results are released on January 5."),
        ]),
        "deadlines.md": ("Academic Deadlines", [
            ("the add/drop deadline", "Courses can be added or dropped until September 26."),
            ("the tuition payment deadline", "Autumn tuition is due by October 1; late payment adds a $75 fee."),
            ("the thesis submission deadline", "Final-year theses are due on April 30 at 17:00."),
            ("the withdrawal deadline", "Students can withdraw with a W grade until November 14."),
        ]),
        "policies.md": ("Course Policies", [
            ("the late submission penalty", "Late coursework loses 5% per day, up to 5 days, then scores zero."),
            ("the attendance requirement", "Students must attend at least 80% of seminars to sit the exam."),
            ("the extension policy", "Extensions of up to 7 days need a request before the deadline."),
            ("the academic integrity process", "Suspected misconduct is reviewed by a panel within 15 working days."),
        ]),
    }),
]  # fmt: skip


def _inject_noise(rng, clean_rows, make_bad_prompt, kind):
    """Append duplicate, low-quality and malformed rows; return JSONL lines."""
    lines = [json.dumps(r) for r in clean_rows]
    for i in range(N_DUPLICATE[kind]):
        row = dict(rng.choice(clean_rows))
        row["prompt"] = [row["prompt"], row["prompt"].lower(), " " + row["prompt"] + "  "][i % 3]
        lines.append(json.dumps(row))
    for i in range(N_LOW_QUALITY[kind]):
        answer = ["N/A", "TODO", "I'm sorry, I can't help with that."][i % 3]
        lines.append(json.dumps({"prompt": make_bad_prompt(i), "completion": answer}))
    for i in range(N_MALFORMED[kind]):
        prompt = make_bad_prompt(100 + i)
        lines.append(
            [
                json.dumps({"prompt": prompt}),
                json.dumps({"prompt": prompt, "completion": ""}),
                json.dumps({"prompt": prompt, "completion": None}),
                json.dumps({"prompt": prompt, "completion": "ok"})[:25],
            ][i % 4]
        )
    rng.shuffle(lines)
    return lines


def _write_case_readme(folder: Path, title: str, goal: str, kind: str, counts: dict, pii: dict) -> None:
    verdict = (
        "Format/behavior gap: fine-tuning fits"
        if kind == "finetune"
        else "Knowledge gap: use retrieval, not fine-tuning"
    )
    lines = [f"# {title}", "", f"Expected diagnosis: **{verdict}**.", "",
             "```bash", f"trainjudge diagnose --dataset {folder.relative_to(HERE.parent.parent)}/data.jsonl \\",
             f'  --model Qwen3-0.6B --goal "{goal}"', "```", "",
             "| Rows | Count |", "|---|---|"]  # fmt: skip
    lines += [f"| {k} | {v} |" for k, v in counts.items()]
    if pii:
        lines += [
            "",
            "Planted sensitive identifiers (fictional values): "
            + ", ".join(f"{k} ×{v}" for k, v in pii.items())
            + ".",
        ]
    if kind == "retrieval":
        lines += ["", "The answers come from the documents in [docs/](docs/)."]
    lines += ["", "Generated by [../../generate.py](../../generate.py). All names are fictional.", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_finetune(case: FinetuneCase, rng: random.Random) -> None:
    folder = HERE / case.domain / case.name
    folder.mkdir(parents=True, exist_ok=True)
    seen, clean = set(), []

    def add(text, out):
        if text not in seen:
            seen.add(text)
            clean.append({"prompt": f"{case.instruction}\nInput: {text}", "completion": json.dumps(out)})
            return True
        return False

    for i in range(case.n_pii):
        while not add(*case.make_pii_example(rng, i)):
            pass
    while len(clean) < case.n_clean:
        add(*case.make_example(rng))

    def bad_prompt(i):
        while True:
            text, _ = case.make_example(rng)
            if text not in seen:
                seen.add(text)
                return f"{case.instruction}\nInput: {text}"

    pii_rows = clean[: case.n_pii]
    rest = clean[case.n_pii :]
    lines = [json.dumps(r) for r in pii_rows] + _inject_noise(rng, rest, bad_prompt, "finetune")
    rng.shuffle(lines)
    (folder / "data.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = {"Clean": case.n_clean, "Duplicates": N_DUPLICATE["finetune"],
              "Low-quality": N_LOW_QUALITY["finetune"], "Malformed": N_MALFORMED["finetune"]}  # fmt: skip
    _write_case_readme(folder, case.title, case.goal, "finetune", counts, case.pii_kinds)


def build_retrieval(case: RetrievalCase, rng: random.Random) -> None:
    folder = HERE / case.domain / case.name
    docs_dir = folder / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for filename, (title, facts) in case.docs.items():
        body = "\n".join(f"- {answer}" for _, answer in facts)
        (docs_dir / filename).write_text(
            f"# {case.org}: {title}\n\nEffective September 1, 2026. Subject to change.\n\n{body}\n",
            encoding="utf-8",
        )
    clean = [
        {"prompt": t.format(topic=topic), "completion": answer, "source": source}
        for source, (_, facts) in case.docs.items()
        for topic, answer in facts
        for t in QUESTION_TEMPLATES
    ]
    topics = [(source, topic) for source, (_, facts) in case.docs.items() for topic, _ in facts]

    def bad_prompt(i):
        return RESERVED_TEMPLATE.format(topic=topics[i % len(topics)][1]) + ("" if i < 100 else " (again)")

    lines = _inject_noise(rng, clean, bad_prompt, "retrieval")
    (folder / "data.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = {"Clean": len(clean), "Duplicates": N_DUPLICATE["retrieval"],
              "Low-quality": N_LOW_QUALITY["retrieval"], "Malformed": N_MALFORMED["retrieval"]}  # fmt: skip
    _write_case_readme(folder, case.title, case.goal, "retrieval", counts, {})


def main() -> None:
    rng = random.Random(SEED)
    for case in FINETUNE_CASES:
        build_finetune(case, rng)
        print(f"wrote {case.domain}/{case.name}")
    for case in RETRIEVAL_CASES:
        build_retrieval(case, rng)
        print(f"wrote {case.domain}/{case.name}")


if __name__ == "__main__":
    main()
