"""Example TrainJudge scorer: does a prose answer keep the gold answer's key facts?

Use it for free-form answers, where exact match is too strict:

    trainjudge verify <run-dir> --scorer trainjudge.scorers.key_facts

An answer is correct when it contains every number in the gold answer (amounts,
days, percentages) and most of its other content words; `score` is the share of
key facts found, for partial credit. It's a starting point: copy it and adapt
the rules to what "right" means for your task.
"""

import re

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
WORD_RE = re.compile(r"[a-z][a-z'-]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "only", "or", "that", "the", "their", "this", "to", "with", "within",
}  # fmt: skip
MIN_WORD_SHARE = 0.6


def _words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS}


def score(prompt: str, output: str, gold: str) -> dict:
    numbers = set(NUMBER_RE.findall(gold))
    words = _words(gold)
    found_numbers = numbers & set(NUMBER_RE.findall(output))
    found_words = words & _words(output)

    word_share = len(found_words) / len(words) if words else 1.0
    facts = len(numbers) + len(words)
    share = (len(found_numbers) + len(found_words)) / facts if facts else 0.0
    correct = found_numbers == numbers and word_share >= MIN_WORD_SHARE

    missing = sorted(numbers - found_numbers) + sorted(words - found_words)
    reason = "all key facts present" if not missing else "missing: " + ", ".join(missing[:6])
    return {"correct": correct, "score": share, "reason": reason}
