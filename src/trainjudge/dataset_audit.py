"""Dataset audit: flag malformed rows, near-duplicates and low-quality examples.

Each row gets exactly one status, checked in priority order:
malformed > duplicate > low_quality > clean. The audit only flags rows;
nothing is removed unless the caller writes a cleaned copy.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from trainjudge import pii

CLEAN = "clean"
DUPLICATE = "duplicate"
LOW_QUALITY = "low_quality"
MALFORMED = "malformed"
STATUSES = (CLEAN, DUPLICATE, LOW_QUALITY, MALFORMED)

LABELS = {
    CLEAN: "clean",
    DUPLICATE: "duplicates",
    LOW_QUALITY: "low-quality",
    MALFORMED: "malformed",
}

# mlx-lm's supported JSONL layouts.
FORMAT_NAMES = {"completions": "prompt/completion", "chat": "chat messages", "text": "text"}
CHAT_ROLES = {"system", "user", "assistant", "tool"}

PLACEHOLDERS = {"todo", "tbd", "n/a", "na", "none", "null", "...", "?", "-", "xxx", "placeholder"}
REFUSAL = re.compile(
    r"^(i'?m sorry|i am sorry|sorry,|as an ai\b|i cannot\b|i can'?t\b|i'?m unable|i am unable"
    r"|i'?m not able)",
    re.IGNORECASE,
)
MIN_ECHO_CHARS = 8
MAX_TOKEN_RUN = 6


class MalformedRow(ValueError):
    """A row that can't be read as a training example."""


@dataclass(frozen=True)
class Example:
    prompt: str
    completion: str


@dataclass
class RowResult:
    line: int
    status: str
    reason: str = ""
    raw: str = ""
    example: Example | None = None
    fields: tuple[str, ...] = ()


@dataclass
class AuditReport:
    path: str
    format: str | None
    rows: list[RowResult]
    conflicting_prompts: int
    prompt_chars_median: float
    completion_chars_median: float
    sensitive: dict[str, list[int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.rows)

    def counts(self) -> dict[str, int]:
        counter = Counter(r.status for r in self.rows)
        return {s: counter.get(s, 0) for s in STATUSES}

    def percentages(self) -> dict[str, int]:
        return _whole_percentages(self.counts(), self.total)

    def issues(self, status: str) -> list[RowResult]:
        return [r for r in self.rows if r.status == status]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "total": self.total,
            "counts": self.counts(),
            "percentages": self.percentages(),
            "conflicting_prompts": self.conflicting_prompts,
            "median_chars": {
                "prompt": self.prompt_chars_median,
                "completion": self.completion_chars_median,
            },
            "sensitive_data": self.sensitive,
            "issues": [
                {"line": r.line, "status": r.status, "reason": r.reason}
                for r in self.rows
                if r.status != CLEAN
            ],
        }


def audit_dataset(path: str | Path) -> AuditReport:
    """Audit a JSONL dataset in any of mlx-lm's formats."""
    path = Path(path)
    parsed: list[tuple[int, str, object]] = []
    with path.open("rb") as f:
        for lineno, raw_bytes in enumerate(f, start=1):
            try:
                raw = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                parsed.append((lineno, "", MalformedRow("not valid UTF-8")))
                continue
            if not raw.strip():
                continue
            try:
                parsed.append((lineno, raw, json.loads(raw.rstrip("\r\n"))))
            except json.JSONDecodeError as e:
                parsed.append((lineno, raw, MalformedRow(f"invalid JSON ({e.msg})")))

    fmt = _dominant_format(obj for _, _, obj in parsed if not isinstance(obj, MalformedRow))

    rows: list[RowResult] = []
    first_seen: dict[tuple[str, str], int] = {}
    completions_by_prompt: dict[str, set[str]] = defaultdict(set)
    prompt_lens: list[int] = []
    completion_lens: list[int] = []
    sensitive: dict[str, list[int]] = defaultdict(list)

    for lineno, raw, obj in parsed:
        kinds = pii.scan_text(raw) if isinstance(obj, MalformedRow) else pii.scan_value(obj)
        for kind in kinds:
            sensitive[kind].append(lineno)
        try:
            if isinstance(obj, MalformedRow):
                raise obj
            example = _extract(obj, fmt)
        except MalformedRow as e:
            rows.append(RowResult(lineno, MALFORMED, str(e), raw))
            continue

        fields = tuple(obj)
        key = (normalize(example.prompt), normalize(example.completion))
        if key in first_seen:
            reason = f"duplicate of line {first_seen[key]}"
            rows.append(RowResult(lineno, DUPLICATE, reason, raw, example, fields))
            continue
        first_seen[key] = lineno

        issue = _quality_issue(example)
        if issue:
            rows.append(RowResult(lineno, LOW_QUALITY, issue, raw, example, fields))
            continue

        rows.append(RowResult(lineno, CLEAN, "", raw, example, fields))
        prompt_lens.append(len(example.prompt))
        completion_lens.append(len(example.completion))
        if example.prompt:
            completions_by_prompt[key[0]].add(key[1])

    return AuditReport(
        path=str(path),
        format=fmt,
        rows=rows,
        conflicting_prompts=sum(1 for c in completions_by_prompt.values() if len(c) > 1),
        prompt_chars_median=statistics.median(prompt_lens) if prompt_lens else 0,
        completion_chars_median=statistics.median(completion_lens) if completion_lens else 0,
        sensitive={k: sensitive[k] for k in pii.KINDS if k in sensitive},
    )


def write_clean_dataset(
    report: AuditReport, out_path: str | Path, drop_low_quality: bool = False
) -> int:
    """Write rows that aren't duplicates or malformed; return how many were kept."""
    drop = {DUPLICATE, MALFORMED} | ({LOW_QUALITY} if drop_low_quality else set())
    kept = [r for r in report.rows if r.status not in drop]
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.writelines(r.raw if r.raw.endswith("\n") else r.raw + "\n" for r in kept)
    return len(kept)


def summary_line(report: AuditReport) -> str:
    pct = report.percentages()
    return " · ".join(f"{pct[s]}% {LABELS[s]}" for s in STATUSES)


def format_report(report: AuditReport, max_examples: int = 5) -> str:
    lines = [
        "TRAINJUDGE DATASET AUDIT",
        "",
        f"Dataset: {report.path}",
        f"Format:  {FORMAT_NAMES.get(report.format, 'unrecognized')}",
        f"  {report.total:,} examples",
        f"  {summary_line(report)}",
    ]

    duplicates = report.issues(DUPLICATE)
    if duplicates:
        lines += ["", f"Duplicates ({len(duplicates):,})"]
        lines += [f"  line {r.line:<6} {r.reason}" for r in duplicates[:max_examples]]
        if len(duplicates) > max_examples:
            lines.append(f"  … and {len(duplicates) - max_examples:,} more")

    for status in (LOW_QUALITY, MALFORMED):
        issues = report.issues(status)
        if issues:
            lines += ["", f"{LABELS[status].capitalize()} ({len(issues):,})"]
            lines += _grouped_by_reason(issues)

    if report.sensitive:
        lines += ["", *format_sensitive(report)]

    if report.conflicting_prompts:
        lines += [
            "",
            "Warnings",
            (f"  {report.conflicting_prompts:,} prompts have conflicting completions "
            "(same prompt, different answer)"),
        ]
    return "\n".join(lines)


def format_sensitive(report: AuditReport) -> list[str]:
    rows = {line for lines in report.sensitive.values() for line in lines}
    out = [f"Sensitive data ({len(rows):,} rows)"]
    for kind, line_nos in report.sensitive.items():
        sample = ", ".join(map(str, line_nos[:3])) + (", …" if len(line_nos) > 3 else "")
        out.append(f"  ⚠ {kind}: {len(line_nos):,} rows  (line {sample})")
    out.append(
        "  Mask or remove these before training. Fine-tuned models can memorize and repeat "
        "training data."
    )
    return out


def _grouped_by_reason(issues: list[RowResult]) -> list[str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for r in issues:
        groups[r.reason].append(r.line)
    out = []
    for reason, line_nos in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        sample = ", ".join(map(str, line_nos[:3])) + (", …" if len(line_nos) > 3 else "")
        out.append(f"  {len(line_nos):>5,}  {reason}  (line {sample})")
    return out


def _row_format(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return None
    if "messages" in obj:
        return "chat"
    if "prompt" in obj or "completion" in obj:
        return "completions"
    if "text" in obj:
        return "text"
    return None


def _dominant_format(objs) -> str | None:
    counts = Counter(f for f in map(_row_format, objs) if f)
    return counts.most_common(1)[0][0] if counts else None


def _extract(obj: object, fmt: str | None) -> Example:
    if not isinstance(obj, dict):
        raise MalformedRow("row is not a JSON object")
    row_fmt = _row_format(obj)
    if row_fmt is None:
        raise MalformedRow("no prompt/completion, messages or text field")
    if row_fmt != fmt:
        raise MalformedRow(f"{FORMAT_NAMES[row_fmt]} row in a {FORMAT_NAMES[fmt]} dataset")
    if fmt == "completions":
        return Example(_text_field(obj, "prompt"), _text_field(obj, "completion"))
    if fmt == "text":
        return Example("", _text_field(obj, "text"))
    return _chat_example(obj["messages"])


def _text_field(obj: dict, key: str) -> str:
    if key not in obj:
        raise MalformedRow(f"missing '{key}'")
    value = obj[key]
    if not isinstance(value, str):
        raise MalformedRow(f"'{key}' is not a string")
    if not value.strip():
        raise MalformedRow(f"'{key}' is empty")
    return value


def _chat_example(messages: object) -> Example:
    if not isinstance(messages, list) or not messages:
        raise MalformedRow("'messages' is not a non-empty list")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in CHAT_ROLES:
            raise MalformedRow(f"message {i} has no valid role")
        if not isinstance(m.get("content"), str):
            raise MalformedRow(f"message {i} content is not a string")
    last = messages[-1]
    if last["role"] != "assistant":
        raise MalformedRow("last message is not from the assistant")
    if not last["content"].strip():
        raise MalformedRow("assistant reply is empty")
    return Example("\n".join(m["content"] for m in messages[:-1]), last["content"])


def normalize(text: str) -> str:
    """Case/whitespace/trailing-punctuation-insensitive form used for duplicate checks."""
    return re.sub(r"\s+", " ", text).strip().casefold().rstrip(".?!;").strip()


def _quality_issue(example: Example) -> str | None:
    completion = example.completion.strip()
    folded = completion.casefold()
    if folded in PLACEHOLDERS or folded.startswith("lorem ipsum"):
        return "placeholder completion"
    if REFUSAL.match(completion):
        return "completion is a refusal"
    norm = normalize(completion)
    if len(norm) >= MIN_ECHO_CHARS and normalize(example.prompt).endswith(norm):
        return "completion repeats the prompt"
    if _longest_token_run(completion) >= MAX_TOKEN_RUN:
        return "degenerate repetition (same token repeated)"
    return None


def _longest_token_run(text: str) -> int:
    best = run = 0
    prev = None
    for token in text.casefold().split():
        run = run + 1 if token == prev else 1
        best = max(best, run)
        prev = token
    return best


def _whole_percentages(counts: dict[str, int], total: int) -> dict[str, int]:
    """Round to whole percentages that still sum to 100 (largest-remainder method)."""
    if total == 0:
        return {k: 0 for k in counts}
    exact = {k: v * 100 / total for k, v in counts.items()}
    result = {k: int(v) for k, v in exact.items()}
    shortfall = 100 - sum(result.values())
    for k in sorted(exact, key=lambda k: exact[k] - result[k], reverse=True)[:shortfall]:
        result[k] += 1
    return result
