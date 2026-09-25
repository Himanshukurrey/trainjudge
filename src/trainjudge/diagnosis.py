"""Diagnosis: decide whether fine-tuning is the right fix before any training.

Sorts a goal + dataset into one of four gaps:

- knowledge: the model lacks facts -> use retrieval, not fine-tuning
- format:    the model needs to learn an output format/behavior -> fine-tune
- cost:      a big model already does the task; make it cheaper/faster -> distill
- prompt:    instructions or few-shot examples haven't been tried -> fix the prompt

This is a transparent, rule-based scorer: every point a bucket earns comes
with a line of evidence. It is a heuristic, not a guarantee, and mixed goals
(part knowledge, part format) are reported as such. The JSON output carries
all signals so a coding agent can review and override the call.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass

from trainjudge import domains, pii
from trainjudge.dataset_audit import CLEAN, AuditReport, normalize, summary_line

KNOWLEDGE = "knowledge"
FORMAT = "format"
COST = "cost"
PROMPT = "prompt"
UNCLEAR = "unclear"
BUCKETS = (KNOWLEDGE, PROMPT, COST, FORMAT)  # tie-break order: most cautious first

TITLES = {
    KNOWLEDGE: "KNOWLEDGE GAP",
    FORMAT: "FORMAT/BEHAVIOR GAP",
    COST: "COST/LATENCY GAP",
    PROMPT: "PROMPT-ENGINEERING GAP",
    UNCLEAR: "UNCLEAR",
}

KNOWLEDGE_TERMS = [
    "document",
    "documents",
    "docs",
    "policy",
    "policies",
    "knowledge base",
    "faq",
    "faqs",
    "manual",
    "handbook",
    "wiki",
    "internal",
    "facts",
    "latest",
    "up to date",
    "up-to-date",
    "catalog",
    "catalogue",
    "product information",
    "product details",
    "answer questions about",
    "answer from",
    "remember",
    "know about",
    "regulation",
    "regulations",
    "guidelines",
    "terms and conditions",
]
FORMAT_TERMS = [
    "sql",
    "json",
    "schema",
    "format",
    "formatted",
    "structured",
    "extract",
    "extraction",
    "parse",
    "classify",
    "classification",
    "categorize",
    "categorise",
    "categorization",
    "tag",
    "label",
    "labels",
    "style",
    "tone",
    "convention",
    "conventions",
    "template",
    "code",
    "normalize",
    "iso 20022",
    "swift",
    "mt103",
    "narration",
    "narrations",
    "triage",
]
COST_TERMS = [
    "cheaper",
    "cost",
    "costs",
    "latency",
    "faster",
    "speed up",
    "distill",
    "distillation",
    "expensive",
    "api bill",
    "api costs",
    "smaller model",
    "reduce spend",
    "throughput",
    "on-prem",
    "on-premise",
    "on premises",
    "self-host",
    "self-hosted",
    "data residency",
    "replace gpt",
    "replace claude",
    "replace the api",
]
PROMPT_TERMS = [
    "prompt",
    "system prompt",
    "few-shot",
    "instructions",
    "ignores",
    "sometimes",
    "occasionally",
    "inconsistent",
]

SOURCE_FIELDS = {
    "source",
    "sources",
    "doc",
    "document",
    "doc_id",
    "reference",
    "references",
    "citation",
    "url",
}
STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
    "with",
    "that",
    "this",
    "from",
    "your",
    "you",
    "our",
    "has",
    "have",
    "can",
    "not",
    "all",
    "any",
    "per",
    "its",
    "into",
    "within",
    "after",
    "before",
    "what",
    "which",
    "who",
    "how",
    "when",
    "does",
    "their",
    "them",
    "they",
    "will",
}

_SQL_RE = re.compile(r"^\s*(select|with|insert|update|delete|create)\b", re.IGNORECASE)
_CODE_RE = re.compile(r"^\s*(```|def |class |import |from \S+ import|function |const |let |#include|public )")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[a-z0-9]+")

SHAPE_DESCRIPTIONS = {
    "sql": "text → SQL pairs",
    "json": "text → JSON pairs",
    "code": "text → code pairs",
    "label": "text → label pairs",
    "prose": "question → prose answer pairs",
}
METRIC_HINTS = {
    "sql": "does the query execute and return the right rows",
    "json": "does the output parse and match the expected fields",
    "code": "do the tests pass",
    "label": "accuracy against held-out labels",
    "prose": "a rubric or held-out judge; harder to measure objectively",
}


@dataclass
class DatasetProfile:
    examples: int
    output_shape: str
    shape_share: float
    distinct_completions: int
    novel_number_rate: float
    copy_ratio: float
    source_field_rate: float
    prompt_words_median: float

    @property
    def grounded(self) -> bool:
        """Prompts carry the context and answers mostly copy from it."""
        return self.prompt_words_median >= 120 and self.copy_ratio >= 0.6


@dataclass
class Diagnosis:
    goal: str
    model: str
    classification: str
    confidence: str
    scores: dict[str, int]
    evidence: dict[str, list[str]]
    secondary: str | None
    profile: DatasetProfile
    audit: AuditReport
    tried_prompting: bool | None
    domain_match: domains.DomainMatch | None = None

    @property
    def fine_tune_recommended(self) -> bool:
        return self.classification in (FORMAT, COST)

    @property
    def domain(self) -> domains.DomainPack | None:
        return self.domain_match.pack if self.domain_match else None

    @property
    def changing_fact_terms(self) -> list[str]:
        return self.domain_match.changing_fact_terms if self.domain_match else []

    @property
    def high_stakes_terms(self) -> list[str]:
        return self.domain_match.high_stakes_terms if self.domain_match else []

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "model": self.model,
            "classification": self.classification,
            "confidence": self.confidence,
            "fine_tune_recommended": self.fine_tune_recommended,
            "secondary": self.secondary,
            "scores": self.scores,
            "evidence": self.evidence,
            "tried_prompting": self.tried_prompting,
            "profile": {**self.profile.__dict__, "grounded": self.profile.grounded},
            "domain": (
                {
                    "name": self.domain.name,
                    "label": self.domain.label,
                    "terms": self.domain_match.terms,
                    "changing_fact_terms": self.changing_fact_terms,
                    "high_stakes_terms": self.high_stakes_terms,
                }
                if self.domain
                else None
            ),
            "audit": {
                "total": self.audit.total,
                "counts": self.audit.counts(),
                "percentages": self.audit.percentages(),
            },
            "sensitive_data": self.audit.sensitive,
        }


def diagnose(
    goal: str,
    audit: AuditReport,
    model: str = "",
    tried_prompting: bool | None = None,
    domain: str = "auto",
) -> Diagnosis:
    profile = profile_dataset(audit)
    scores = {b: 0 for b in BUCKETS}
    evidence: dict[str, list[str]] = {b: [] for b in BUCKETS}

    def add(bucket: str, points: int, reason: str) -> None:
        scores[bucket] += points
        evidence[bucket].append(reason)

    knowledge_hits = _find_terms(goal, KNOWLEDGE_TERMS)
    format_hits = _find_terms(goal, FORMAT_TERMS)
    cost_hits = _find_terms(goal, COST_TERMS)
    prompt_hits = _find_terms(goal, PROMPT_TERMS)
    sample_text = " ".join(
        f"{r.example.prompt} {r.example.completion}" for r in audit.rows[:300] if r.example is not None
    )
    domain_match = domains.detect(goal, sample_text, domain)
    changing_hits = domain_match.changing_fact_terms if domain_match else []

    # Goal text.
    if knowledge_hits:
        add(KNOWLEDGE, min(3, 1 + len(knowledge_hits)), f"goal mentions {_quote(knowledge_hits)}")
    if changing_hits:
        add(
            KNOWLEDGE,
            1,
            f"goal involves {domain_match.pack.label} facts that change ({_quote(changing_hits)})",
        )
    if format_hits:
        add(FORMAT, min(3, 1 + len(format_hits)), f"goal mentions {_quote(format_hits)}")
    if cost_hits:
        add(COST, 5, f"goal mentions {_quote(cost_hits)}")
    if prompt_hits:
        add(PROMPT, 1, f"goal mentions {_quote(prompt_hits)}")

    # Dataset shape.
    n = profile.examples
    shape_pct = f"{profile.shape_share:.0%}"
    if profile.output_shape in ("sql", "json", "code", "label") and n:
        add(
            FORMAT,
            3,
            f"{shape_pct} of completions are {profile.output_shape.upper()} "
            "(structured output that can be checked automatically)",
        )
        if cost_hits:
            add(COST, 1, "the task is narrow and structured, which suits a small model")
    elif profile.output_shape == "prose" and n:
        if profile.grounded:
            add(
                FORMAT,
                2,
                "prompts include the source text and answers draw on it "
                "(the model learns to answer from given context)",
            )
            scores[KNOWLEDGE] -= 3
            evidence[KNOWLEDGE].append("answers are grounded in the prompt, not recalled (-3)")
        else:
            add(KNOWLEDGE, 1, "completions are free-form prose answers")
            if profile.novel_number_rate >= 0.3:
                add(
                    KNOWLEDGE,
                    2,
                    f"{profile.novel_number_rate:.0%} of answers state numbers, "
                    "dates or amounts that aren't in the question",
                )
            if n >= 20 and profile.distinct_completions <= 0.5 * n:
                add(
                    KNOWLEDGE,
                    2,
                    f"{n:,} prompts map to only {profile.distinct_completions:,} "
                    "distinct answers, so the dataset teaches recall of fixed facts",
                )
            if profile.copy_ratio < 0.35:
                add(
                    KNOWLEDGE,
                    1,
                    f"only {profile.copy_ratio:.0%} of answer words appear in the "
                    "prompt; the facts have to come from the model's weights",
                )
    if profile.source_field_rate >= 0.5 and not profile.grounded:
        add(KNOWLEDGE, 2, f"{profile.source_field_rate:.0%} of rows cite a source document")

    # Prompting and dataset size.
    if n < 50:
        add(PROMPT, 3, f"only {n:,} clean examples; few-shot prompting can use them directly")
    elif n < 200:
        add(PROMPT, 1, f"{n:,} clean examples is small for fine-tuning")
    if tried_prompting is False:
        add(PROMPT, 3, "few-shot prompting hasn't been tried yet")
    elif tried_prompting is True:
        scores[PROMPT] -= 2
        evidence[PROMPT].append("few-shot prompting was already tried (-2)")

    ranked = sorted(BUCKETS, key=lambda b: (-scores[b], BUCKETS.index(b)))
    top, second = ranked[0], ranked[1]
    margin = scores[top] - scores[second]
    if scores[top] < 3:
        classification, confidence, secondary = UNCLEAR, "low", None
    else:
        classification = top
        confidence = "high" if margin >= 4 else "medium" if margin >= 2 else "low"
        secondary = second if scores[second] >= 3 and margin < 4 else None

    return Diagnosis(
        goal=goal,
        model=model,
        classification=classification,
        confidence=confidence,
        scores=scores,
        evidence={b: e for b, e in evidence.items() if e},
        secondary=secondary,
        profile=profile,
        audit=audit,
        tried_prompting=tried_prompting,
        domain_match=domain_match,
    )


def profile_dataset(audit: AuditReport) -> DatasetProfile:
    rows = [r for r in audit.rows if r.status == CLEAN and r.example is not None]
    examples = [r.example for r in rows]
    n = len(examples)
    if not n:
        return DatasetProfile(0, "prose", 0.0, 0, 0.0, 0.0, 0.0, 0.0)

    shapes = Counter(_shape(e.completion) for e in examples)
    shape, count = shapes.most_common(1)[0]
    distinct = len({normalize(e.completion) for e in examples})
    if shape == "prose":
        words = statistics.median(len(e.completion.split()) for e in examples)
        if words <= 3 and distinct <= max(50, n // 10):
            shape = "label"

    novel = sum(1 for e in examples if _novel_numbers(e))
    copy_ratios = [r for r in map(_copy_ratio, examples) if r is not None]
    return DatasetProfile(
        examples=n,
        output_shape=shape,
        shape_share=count / n,
        distinct_completions=distinct,
        novel_number_rate=novel / n,
        copy_ratio=statistics.mean(copy_ratios) if copy_ratios else 0.0,
        source_field_rate=sum(1 for r in rows if SOURCE_FIELDS & set(r.fields)) / n,
        prompt_words_median=statistics.median(len(e.prompt.split()) for e in examples),
    )


def format_diagnosis(d: Diagnosis) -> str:
    p = d.profile
    audit = d.audit
    lines = ["TRAINJUDGE DIAGNOSIS", "", f"Goal: {d.goal}"]
    if d.model:
        lines.append(f"Model: {d.model}")
    described = SHAPE_DESCRIPTIONS[p.output_shape]
    if p.source_field_rate >= 0.5:
        described += " citing source documents"
    lines.append(f"Dataset: {audit.total:,} examples ({described})")
    lines += ["", f"Classification: {TITLES[d.classification]}   (confidence: {d.confidence})"]

    why, recommendation, steps = _advice(d)
    lines += ["", *_wrap(f"Why: {why}")]
    evidence = d.evidence.get(d.classification, [])
    if evidence:
        lines += ["", "Evidence:"]
        for e in evidence:
            lines += _wrap(e, first="  • ", rest="    ")
    if d.secondary:
        lines += [
            "",
            *_wrap(
                f"Mixed goal: this also looks like a {TITLES[d.secondary]}. "
                f"{_mixed_hint(d.classification, d.secondary)}"
            ),
        ]
    lines += ["", f"Recommendation: {recommendation}"]
    if steps:
        lines += ["", "Try instead:" if not d.fine_tune_recommended else "Next steps:"]
        for i, step in enumerate(steps, start=1):
            lines += _wrap(step, first=f"  {i}. ", rest="     ")

    if d.domain or audit.sensitive:
        lines += ["", f"{d.domain.label} checks:" if d.domain else "Data checks:"]
        for marker, text in _domain_notes(d):
            lines += _wrap(text, first=f"  {marker} ", rest="    ")

    counts = audit.counts()
    lines += [
        "",
        "Dataset audit:",
        f"  {audit.total:,} examples",
        f"  {summary_line(audit)}",
    ]
    removable = counts["duplicate"] + counts["malformed"]
    if removable:
        lines.append(f"  {removable:,} duplicate/malformed rows would be removed before training")

    if d.fine_tune_recommended:
        lines += [
            "",
            *_wrap(
                "Next: clean the dataset (trainjudge audit --write-clean) and "
                "run a baseline eval before training."
            ),
        ]
    elif d.classification != UNCLEAR:
        lines += [
            "",
            *_wrap(
                "Want to fine-tune anyway? TrainJudge will still measure the "
                "result against a baseline, so the verdict will show whether "
                "this diagnosis held."
            ),
        ]
    return "\n".join(lines)


def _advice(d: Diagnosis) -> tuple[str, str, list[str]]:
    p = d.profile
    if d.classification == KNOWLEDGE:
        changing = d.domain.changing_facts if d.domain else "policy updates, new products"
        why = (
            "your dataset teaches the model to recall specific facts that will change over "
            f"time ({changing}). Fine-tuning bakes these facts into weights, so updating them "
            "later means re-training, and the model can't show where an answer came from. "
            "Retrieval (RAG) keeps facts in a swappable index instead."
        )
        steps = [
            "Index the source documents with a retrieval pipeline"
            + (d.domain.retrieval_hint if d.domain else ""),
            "Use the base model with retrieved context in the prompt",
            (
                "If answers are still wrong, THEN consider fine-tuning the answer *format* "
                "(citations, tone, structure) on context + question → answer pairs, not the facts"
            ),
        ]
        return why, "❌ Do not fine-tune for this goal.", steps
    if d.classification == FORMAT:
        task = {
            "sql": "SQL generation",
            "json": "Producing JSON",
            "code": "Code generation",
            "label": "Labeling/classification",
        }.get(p.output_shape)
        if task:
            why = (
                f"{task} is a structured-output task. The model already knows the general "
                "format; it needs to learn your conventions and patterns. This is directly "
                f"measurable ({METRIC_HINTS[p.output_shape]}) and doesn't go stale the way "
                "facts do."
            )
        else:
            why = (
                "matching a consistent output style or behavior is something fine-tuning does "
                "well, and it doesn't go stale the way facts do. Measuring it needs "
                f"{METRIC_HINTS['prose']}."
            )
        steps = [
            "Clean the dataset: drop duplicate and malformed rows",
            "Measure the base model on a held-out split (the baseline)",
            f"Fine-tune {d.model or 'the model'} with LoRA and compare on the same split",
        ]
        return why, "✓ Fine-tuning is a reasonable fit for this goal.", steps
    if d.classification == COST:
        why = (
            "you want the same narrow task done cheaper or faster. That's a distillation fit: "
            "train a small model on the expensive model's outputs. Judge it on accuracy per unit "
            "of cost and latency, not accuracy alone. A small model that's 3% worse but 20× "
            "cheaper may be the right trade."
        )
        steps = [
            (
                "Record the expensive model's accuracy, cost and latency on a held-out set "
                "(that's the baseline)"
            ),
            f"Fine-tune {d.model or 'a small model'} on the expensive model's outputs",
            "Accept it only if accuracy stays within your tolerance at the lower cost",
        ]
        return why, "✓ Distillation candidate: fine-tune a small model.", steps
    if d.classification == PROMPT:
        tried = "" if d.tried_prompting else ", and few-shot prompting hasn't been tried"
        why = (
            f"the dataset has {p.examples:,} clean examples{tried}. Behavior like this can "
            "often be fixed with clear instructions and a handful of examples in the prompt: "
            "no training, and instant iteration."
        )
        steps = [
            "Put 3–8 of your best examples in the prompt as few-shot demonstrations",
            "Write the rules explicitly in a system prompt",
            "If it still fails on a held-out set, re-run diagnose with --tried-prompting",
        ]
        return why, "❌ Don't fine-tune yet. Fix the prompt first.", steps
    why = (
        "there isn't enough signal to classify this goal. Say what should change in the "
        "model's output, and check that the dataset is in prompt/completion or chat format."
    )
    return why, "? No recommendation.", []


def _mixed_hint(primary: str, secondary: str) -> str:
    pair = {primary, secondary}
    if pair == {KNOWLEDGE, FORMAT}:
        return "Retrieve the facts, and fine-tune only for the output format."
    if pair == {FORMAT, COST}:
        return "Measure cost and latency alongside accuracy in the verdict."
    if PROMPT in pair:
        return "Try few-shot prompting first; it's the cheapest experiment."
    return "Review the evidence for both before training."


def _domain_notes(d: Diagnosis) -> list[tuple[str, str]]:
    """(marker, paragraph) pairs for the domain / data checks section."""
    notes = []
    sensitive = d.audit.sensitive
    if sensitive:
        rows = {line for lines in sensitive.values() for line in lines}
        kinds = ", ".join(f"{k} ({len(v):,})" for k, v in sensitive.items())
        refs = pii.references(sensitive)
        see = f" (see {' and '.join(refs)})" if refs else ""
        notes.append(
            (
                "⚠",
                f"Sensitive data in {len(rows):,} rows: {kinds}. Mask or tokenize before training. "
                f"Fine-tuned models can memorize and repeat personal data{see}. "
                "Run `trainjudge audit` for line numbers.",
            )
        )
    elif d.domain:
        notes.append(
            ("✓", "No sensitive identifiers found (cards, national IDs, IBANs, accounts, phones, emails).")
        )
    if d.changing_fact_terms:
        notes.append(
            ("⚠", f"{d.domain.label} facts ({_quote(d.changing_fact_terms)}) {d.domain.changing_fact_note}")
        )
    if d.high_stakes_terms:
        notes.append(
            (
                "⚠",
                f"The goal involves automated decisions ({_quote(d.high_stakes_terms)}). "
                + d.domain.high_stakes_note,
            )
        )
    if d.domain:
        notes += [("•", note) for note in d.domain.closing_notes]
    return notes


def _shape(completion: str) -> str:
    s = completion.strip()
    if s[:1] in "{[":
        try:
            json.loads(s)
            return "json"
        except ValueError:
            pass
    if _SQL_RE.match(s):
        return "sql"
    if _CODE_RE.match(s):
        return "code"
    return "prose"


def _numbers(text: str) -> set[str]:
    return {m.replace(",", "") for m in _NUMBER_RE.findall(text)}


def _novel_numbers(example) -> bool:
    return bool(_numbers(example.completion) - _numbers(example.prompt))


def _copy_ratio(example) -> float | None:
    words = [w for w in _WORD_RE.findall(example.completion.lower()) if len(w) >= 3 and w not in STOPWORDS]
    if not words:
        return None
    prompt_words = set(_WORD_RE.findall(example.prompt.lower()))
    return sum(1 for w in words if w in prompt_words) / len(words)


_find_terms = domains.find_terms


def _quote(terms: list[str], limit: int = 3) -> str:
    shown = ", ".join(f'"{t}"' for t in terms[:limit])
    return shown + (f" +{len(terms) - limit} more" if len(terms) > limit else "")


def _wrap(text: str, width: int = 76, first: str = "", rest: str = "") -> list[str]:
    lines, current, empty = [], first, True
    for w in text.split():
        if not empty and len(current) + 1 + len(w) > width:
            lines.append(current)
            current, empty = rest + w, False
        else:
            current, empty = (current + w if empty else f"{current} {w}"), False
    if not empty:
        lines.append(current)
    return lines
