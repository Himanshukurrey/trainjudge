# TrainJudge demos

## Core demo

| Demo | Diagnosis | What it shows |
|---|---|---|
| [sql_generation](sql_generation/) | Format/behavior gap: fine-tune | The full loop: train, eval, verify. [example-runs/](sql_generation/example-runs/) has real IMPROVED, REGRESSED and REJECTED reports. |
| [policy_docs](policy_docs/) | Knowledge gap: don't fine-tune | Q&A that recalls facts from changing policy documents belongs in retrieval |

## Domain demos

Each domain pack has a "fine-tune" demo (a structured-output task, with domain-typical sensitive
identifiers planted in it) and a "retrieval" demo (Q&A over that domain's documents).

| Domain | Fine-tune: format/behavior gap | Don't fine-tune: knowledge gap |
|---|---|---|
| Healthcare | [clinical_coding](domains/healthcare/clinical_coding/): note → diagnosis + ICD-10 (planted MRNs, dates of birth) | [formulary_faq](domains/healthcare/formulary_faq/): hospital formulary dosing |
| Legal | [clause_extraction](domains/legal/clause_extraction/): clause → type, party, duration (planted emails) | [statutes_faq](domains/legal/statutes_faq/): limitation periods, fees, procedure |
| E-commerce/retail | [product_attributes](domains/ecommerce/product_attributes/): listing title → brand, category, color, size | [catalog_faq](domains/ecommerce/catalog_faq/): prices, stock, promotions |
| Customer support | [ticket_triage](domains/customer_support/ticket_triage/): ticket → category, priority, team (planted emails, phones) | [help_center_faq](domains/customer_support/help_center_faq/): plans, SLAs, data policy |
| HR/recruiting | [resume_parsing](domains/hr/resume_parsing/): resume → title, experience, skill, degree (planted emails, dates of birth) | [benefits_faq](domains/hr/benefits_faq/): leave, benefits, pay |
| BFSI | [transactions](domains/bfsi/transactions/): narration → category JSON (planted cards, Aadhaar, PAN, UPI) | [loan_faq](domains/bfsi/loan_faq/): rates and charges |
| Education | [question_tagging](domains/education/question_tagging/): question → subject, topic, difficulty | [course_faq](domains/education/course_faq/): exam dates, deadlines, policies |

All organizations, people, drugs, statutes and products are fictional, and the planted
identifiers are made-up values. Each dataset is generated deterministically: the domain demos by
[domains/generate.py](domains/generate.py), BFSI and the core demos by their own `generate.py`.
CI checks that the committed files match, and the tests pin every demo's audit counts, planted
identifiers and expected diagnosis.

## Demo GIFs

`trainjudge-diagnose-demo.gif` and `trainjudge-verify-demo.gif` are real recordings, not mockups,
captured with [asciinema](https://asciinema.org/) and rendered with [agg](https://github.com/asciinema/agg)
(`brew install asciinema agg`). `record-diagnose-demo.sh` and `record-verify-demo.sh` reproduce them;
the comment header in each script has the exact `asciinema rec` / `agg` commands. The verify demo
replays two trained runs from their saved evals, so it takes seconds rather than re-training.
