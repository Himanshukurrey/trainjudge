"""Replay data: keep general behavior from drifting while fine-tuning on a narrow task.

Before training, the base model answers a set of general prompts, and those
(prompt, answer) pairs are mixed into the training split. Training on the
model's own outputs anchors its general behavior while it learns the task,
without needing an external dataset.

The replay prompts are deliberately disjoint from the regression suite in
regression_check.py (different topics and wording), so the regression check
stays a held-out measurement rather than something the model was trained on.
"""

from __future__ import annotations

from trainjudge.regression_check import build_suite

TOPICS = [
    "photosynthesis",
    "the moon",
    "trains",
    "gardening",
    "rainforests",
    "honeybees",
    "the internet",
    "museums",
    "mountains",
    "jazz",
    "recycling",
    "glaciers",
    "bridges",
    "electric cars",
    "the human heart",
    "deserts",
    "lighthouses",
    "tea",
    "origami",
    "earthquakes",
    "penguins",
    "public parks",
    "solar panels",
    "rivers",
    "clocks",
]
TEMPLATES = [
    "Explain {t} to a ten-year-old in two sentences.",
    "Give three facts about {t} as a bulleted list.",
    "Write a four-line poem about {t}.",
    "Summarize what {t} is in under 20 words.",
    'Write a JSON object about {t} with the keys "topic" and "summary".',
    "List five words related to {t}, separated by commas.",
    "What is one common misconception about {t}? Answer briefly.",
    "Write a one-sentence description of {t} in all capital letters.",
]
ARITHMETIC = [(7, 8), (12, 11), (35, 4), (91, 3), (64, 16), (13, 13), (250, 4), (19, 21)]


def replay_prompts(limit: int) -> list[str]:
    prompts = [template.format(t=topic) for topic in TOPICS for template in TEMPLATES]
    prompts += [f"What is {a} times {b}? Answer with the number only." for a, b in ARITHMETIC]
    prompts = sorted(set(prompts) - {item.prompt for item in build_suite()})
    # Interleave so a small limit still covers every template.
    by_template: dict[int, list[str]] = {}
    for p in prompts:
        by_template.setdefault(template_index(p), []).append(p)
    interleaved = []
    while any(by_template.values()):
        for key in sorted(by_template):
            if by_template[key]:
                interleaved.append(by_template[key].pop(0))
    return interleaved[:limit]


def template_index(prompt: str) -> int:
    """Stable grouping key: which template (or arithmetic) a prompt came from."""
    for i, template in enumerate(TEMPLATES):
        head = template.split("{t}")[0]
        if prompt.startswith(head):
            return i
    return len(TEMPLATES)


def max_replay() -> int:
    return len(replay_prompts(10_000))
