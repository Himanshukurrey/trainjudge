"""Domain packs: industry-specific knowledge layered on the domain-agnostic diagnosis.

The diagnose -> train -> verify core works for any task. A domain pack adds
what's specific to one industry:

- terms that identify the domain in a goal or dataset
- "facts that change" terms: when a goal involves them, retrieval over
  versioned documents fits better than fine-tuning
- high-stakes decision terms that warrant a human-in-the-loop warning
- domain-specific notes for the diagnosis output

Sensitive-data detection lives in `trainjudge.pii` and runs on every dataset,
whatever the domain.

To add a pack, create a module in this package that defines a `PACK`, add it
to `PACKS` below, and add tests plus a "fine-tune" and a "don't fine-tune"
demo dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainPack:
    name: str
    label: str
    description: str
    terms: tuple[str, ...]
    # Facts in this domain that change and belong in retrieval.
    changing_fact_terms: tuple[str, ...] = ()
    changing_facts: str = "policy updates, new products"
    changing_fact_note: str = (
        "change over time. Serve them from versioned documents with effective dates and "
        "citations, so every answer can be traced."
    )
    retrieval_hint: str = ""
    # Decisions about people that shouldn't be fully automated.
    high_stakes_terms: tuple[str, ...] = ()
    high_stakes_note: str = (
        "Keep a human in the loop and prefer interpretable models for the decision itself. "
        "People may be owed reasons for adverse outcomes, and outcomes should be checked for "
        "bias across groups."
    )
    closing_notes: tuple[str, ...] = field(default_factory=tuple)
    # Data-only detection needs this many distinct domain terms; a goal match is enough.
    min_data_hits: int = 3


@dataclass
class DomainMatch:
    pack: DomainPack
    terms: list[str]
    changing_fact_terms: list[str]
    high_stakes_terms: list[str]


def find_terms(text: str, terms) -> list[str]:
    """Terms that occur in text as whole words (case-insensitive)."""
    lowered = text.lower()
    return [t for t in terms if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", lowered)]


def _load_packs() -> dict[str, DomainPack]:
    from trainjudge.domains import bfsi, customer_support, ecommerce, education, healthcare, hr, legal

    packs = (bfsi, healthcare, legal, ecommerce, customer_support, hr, education)
    return {m.PACK.name: m.PACK for m in packs}


PACKS: dict[str, DomainPack] = _load_packs()


def match(pack: DomainPack, goal: str, sample_text: str) -> DomainMatch:
    return DomainMatch(
        pack=pack,
        terms=find_terms(f"{goal} {sample_text}", pack.terms),
        changing_fact_terms=find_terms(goal, pack.changing_fact_terms),
        high_stakes_terms=find_terms(goal, pack.high_stakes_terms),
    )


def detect(goal: str, sample_text: str, domain: str = "auto") -> DomainMatch | None:
    """The best-matching domain pack, or None.

    `domain` is "auto" (detect from the goal and data), "none" (no domain
    checks), or a pack name to force it.
    """
    if domain == "none":
        return None
    if domain != "auto":
        if domain not in PACKS:
            raise ValueError(f"unknown domain {domain!r}; choose from {', '.join(sorted(PACKS))}")
        return match(PACKS[domain], goal, sample_text)

    # The goal says what the user is doing, so goal matches outweigh data matches;
    # data matches break ties. Ties keep registration order.
    best: tuple[tuple[int, int], DomainMatch] | None = None
    for pack in PACKS.values():
        m = match(pack, goal, sample_text)
        goal_hits = len(find_terms(goal, pack.terms))
        if not goal_hits and len(m.terms) < pack.min_data_hits:
            continue
        score = (goal_hits, len(m.terms))
        if best is None or score > best[0]:
            best = (score, m)
    return best[1] if best else None
