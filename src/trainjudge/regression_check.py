"""Held-out general-capability check: did fine-tuning break anything outside the task?

A small built-in suite with programmatic pass/fail checks, in three categories:

- instruction_following: exact bullet counts, lowercase only, word limits,
  required endings, and simple arithmetic
- format_compliance: JSON objects with given keys, JSON arrays of a given
  length, numbered lists
- hallucination: questions about things that don't exist; passing means the
  model says it doesn't know instead of inventing an answer

The suite is deterministic and runs offline. It's small (60 prompts),
so treat per-category changes of a few points as noise-level; the verdict
uses a tolerance for that reason. The hallucination check looks for explicit
uncertainty phrases, so it's a heuristic too.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

INSTRUCTION_FOLLOWING = "instruction_following"
FORMAT_COMPLIANCE = "format_compliance"
HALLUCINATION = "hallucination"
CATEGORIES = (INSTRUCTION_FOLLOWING, FORMAT_COMPLIANCE, HALLUCINATION)
CATEGORY_LABELS = {
    INSTRUCTION_FOLLOWING: "General instruction-following",
    FORMAT_COMPLIANCE: "Format compliance",
    HALLUCINATION: "Hallucination resistance",
}

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_UNCERTAIN_RE = re.compile(
    r"\b(i don'?t know|i do not know|not aware|no (?:reliable |available |public )?information|"
    r"doesn'?t (?:seem to )?exist|does not (?:seem to )?exist|not (?:a )?real|fictional|"
    r"not sure|unable to (?:find|verify|confirm|provide)|no record|cannot (?:find|verify|confirm)|"
    r"can'?t (?:find|verify|confirm)|not familiar|couldn'?t find|could not find|unknown to me|"
    r"no (?:such|known)|don'?t have (?:any )?(?:information|data|details)|"
    r"not (?:able|aware)|no widely (?:known|recognized))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    prompt: str
    check: str
    arg: object = None


@dataclass
class ItemResult:
    id: str
    category: str
    prompt: str
    output: str
    passed: bool


TOPICS = ["the ocean", "coffee", "bicycles", "libraries", "volcanoes", "chess"]
SUMS = [(17, 25), (48, 36), (123, 77), (9, 14), (250, 175), (66, 34)]
FRUITS = ["an apple", "a banana", "a mango", "a lemon", "a grape", "a cherry"]
THINGS = ["colors", "animals", "countries", "sports", "vegetables", "musical instruments"]
ACTIVITIES = [
    "studying",
    "running",
    "cooking at home",
    "sleeping better",
    "saving money",
    "learning a language",
]
FAKE_PRIZES = [
    "Velmont Prize for Ocean Robotics",
    "Kestrow Award for Urban Poetry",
    "Adelric Medal in Quantum Gardening",
    "Thornby Prize for Desert Architecture",
]
FAKE_TOWNS = [
    "Brindlecove, Nebraska",
    "Oskavar, Norway",
    "Pellmoor Heath, Wales",
    "Quantaro, Chile",
]
FAKE_BOOKS = [
    ("The Glass Orchard of Minsk", "Lotte Varrance"),
    ("Seventeen Winters Under Kell", "Amaru Dessinet"),
    ("A Cartographer's Silence", "Idris Mowbeck"),
    ("The Salt Clock", "Renata Oyelaran-Finch"),
]


def build_suite() -> list[Item]:
    items: list[Item] = []
    for i, topic in enumerate(TOPICS):
        items += [
            Item(
                f"if-bullets-{i}",
                INSTRUCTION_FOLLOWING,
                f"Write exactly 3 bullet points about {topic}. Start each bullet with '- ' "
                "and write nothing else.",
                "bullets",
                3,
            ),
            Item(
                f"if-lower-{i}",
                INSTRUCTION_FOLLOWING,
                f"Describe {topic} in one sentence, using only lowercase letters.",
                "lowercase",
            ),
            Item(
                f"if-short-{i}",
                INSTRUCTION_FOLLOWING,
                f"Describe {topic} in fewer than 15 words.",
                "max_words",
                14,
            ),
            Item(
                f"if-end-{i}",
                INSTRUCTION_FOLLOWING,
                f"Write two sentences about {topic}. End your reply with the exact phrase 'That is all.'",
                "ends_with",
                "That is all.",
            ),
        ]
    for i, (a, b) in enumerate(SUMS):
        items.append(
            Item(
                f"if-sum-{i}",
                INSTRUCTION_FOLLOWING,
                f"What is {a} + {b}? Reply with just the number.",
                "number",
                a + b,
            )
        )
    for i, fruit in enumerate(FRUITS):
        items.append(
            Item(
                f"fmt-object-{i}",
                FORMAT_COMPLIANCE,
                f'Return a JSON object with the keys "name" and "color" describing '
                f"{fruit}. Output only the JSON.",
                "json_keys",
                ["name", "color"],
            )
        )
    for i, things in enumerate(THINGS):
        items.append(
            Item(
                f"fmt-array-{i}",
                FORMAT_COMPLIANCE,
                f"List 4 {things} as a JSON array of strings. Output only the JSON.",
                "json_list",
                4,
            )
        )
    for i, activity in enumerate(ACTIVITIES):
        items.append(
            Item(
                f"fmt-numbered-{i}",
                FORMAT_COMPLIANCE,
                f"Give 3 tips for {activity} as a numbered list (1., 2., 3.) with no other text.",
                "numbered",
                3,
            )
        )
    for i, prize in enumerate(FAKE_PRIZES):
        items.append(Item(f"hal-prize-{i}", HALLUCINATION, f"Who won the 2019 {prize}?", "uncertain"))
    for i, town in enumerate(FAKE_TOWNS):
        items.append(
            Item(
                f"hal-town-{i}",
                HALLUCINATION,
                f"What is the population of {town}, and who is its current mayor?",
                "uncertain",
            )
        )
    for i, (title, author) in enumerate(FAKE_BOOKS):
        items.append(
            Item(
                f"hal-book-{i}",
                HALLUCINATION,
                f"Summarize the plot of the novel '{title}' by {author}.",
                "uncertain",
            )
        )
    return items


def clean_output(output: str) -> str:
    return _THINK_RE.sub("", output).strip()


def _json_value(text: str):
    m = _FENCE_RE.match(text)
    return json.loads(m.group(1) if m else text)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def check_bullets(text: str, n: int) -> bool:
    lines = _lines(text)
    return len(lines) == n and all(re.match(r"^[-*•]\s+\S", line) for line in lines)


def check_lowercase(text: str, _=None) -> bool:
    return any(c.isalpha() for c in text) and not any(c.isupper() for c in text)


def check_max_words(text: str, n: int) -> bool:
    return 0 < len(text.split()) <= n


def check_ends_with(text: str, suffix: str) -> bool:
    return text.rstrip().rstrip("*_\"'").rstrip().endswith(suffix)


def check_number(text: str, expected: int) -> bool:
    m = re.fullmatch(r"[*\s]*(-?[\d,]+)[.\s*]*", text)
    return bool(m) and m.group(1).replace(",", "") == str(expected)


def check_json_keys(text: str, keys: list[str]) -> bool:
    try:
        value = _json_value(text)
    except ValueError:
        return False
    return isinstance(value, dict) and all(k in value for k in keys)


def check_json_list(text: str, n: int) -> bool:
    try:
        value = _json_value(text)
    except ValueError:
        return False
    return isinstance(value, list) and len(value) == n and all(isinstance(v, str) for v in value)


def check_numbered(text: str, n: int) -> bool:
    lines = _lines(text)
    return len(lines) == n and all(
        re.match(rf"^\**{i}[.)]\s+\S", line) for i, line in enumerate(lines, start=1)
    )


def check_uncertain(text: str, _=None) -> bool:
    return bool(_UNCERTAIN_RE.search(text))


CHECKS: dict[str, Callable[[str, object], bool]] = {
    "bullets": check_bullets,
    "lowercase": check_lowercase,
    "max_words": check_max_words,
    "ends_with": check_ends_with,
    "number": check_number,
    "json_keys": check_json_keys,
    "json_list": check_json_list,
    "numbered": check_numbered,
    "uncertain": check_uncertain,
}


def score(items: list[Item], outputs: list[str]) -> list[ItemResult]:
    if len(items) != len(outputs):
        raise ValueError(f"{len(items)} items but {len(outputs)} outputs")
    results = []
    for item, output in zip(items, outputs):
        passed = CHECKS[item.check](clean_output(output), item.arg)
        results.append(ItemResult(item.id, item.category, item.prompt, output, passed))
    return results


def summarize(results: list[ItemResult]) -> dict[str, dict]:
    summary = {}
    for category in CATEGORIES:
        rs = [r for r in results if r.category == category]
        passed = sum(r.passed for r in rs)
        summary[category] = {
            "passed": passed,
            "total": len(rs),
            "rate": passed / len(rs) if rs else 0.0,
        }
    return summary


def to_dict(results: list[ItemResult]) -> dict:
    return {"summary": summarize(results), "items": [asdict(r) for r in results]}
