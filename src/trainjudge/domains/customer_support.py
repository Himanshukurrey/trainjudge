"""Customer support and service."""

from trainjudge.domains import DomainPack

PACK = DomainPack(
    name="customer_support",
    label="Customer support",
    description="Customer support, help desks and service chatbots",
    terms=(
        "customer support",
        "customer service",
        "support ticket",
        "support tickets",
        "ticket",
        "tickets",
        "helpdesk",
        "help desk",
        "help center",
        "help centre",
        "faq",
        "faqs",
        "chatbot",
        "support bot",
        "escalation",
        "escalations",
        "csat",
        "live chat",
        "zendesk",
        "intercom",
        "freshdesk",
        "canned responses",
        "macros",
        "support agents",
        "support",
    ),
    changing_fact_terms=(
        "policy",
        "policies",
        "help center",
        "help centre",
        "knowledge base",
        "faq",
        "faqs",
        "product features",
        "plans",
        "pricing",
        "sla",
        "support hours",
        "troubleshooting steps",
    ),
    changing_facts="policy changes, product updates and new help-center articles",
    changing_fact_note=(
        "change whenever the product or policy does. Keep them in the help-center or "
        "knowledge-base index the bot retrieves from, and fine-tune only for tone, format and "
        "routing."
    ),
    retrieval_hint=" (your help-center articles are the natural index)",
    high_stakes_terms=(
        "refund decision",
        "refund decisions",
        "deny refunds",
        "account closure",
        "close accounts",
        "ban users",
        "compensation decision",
    ),
    high_stakes_note=(
        "Let the model draft or route, but have a person approve actions that cost customers "
        "money or access, and give customers a way to reach a human."
    ),
    closing_notes=(
        "Support transcripts usually contain customer names, emails, phone numbers and order "
        "details. The scan catches common identifiers but not names or addresses, so review a "
        "sample before training.",
    ),
)
