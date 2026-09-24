import json

import pytest
from click.testing import CliRunner

from trainjudge.cli import main
from trainjudge.dataset_audit import (
    CLEAN,
    DUPLICATE,
    LOW_QUALITY,
    MALFORMED,
    audit_dataset,
    format_report,
    write_clean_dataset,
)


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def pc(prompt, completion):
    return {"prompt": prompt, "completion": completion}


def statuses(report):
    return [(r.line, r.status) for r in report.rows]


def test_clean_rows(tmp_path):
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [
            pc("List users", "SELECT * FROM users;"),
            pc("Count users", "SELECT COUNT(*) FROM users;"),
        ],
    )
    report = audit_dataset(path)
    assert report.format == "completions"
    assert report.counts() == {CLEAN: 2, DUPLICATE: 0, LOW_QUALITY: 0, MALFORMED: 0}
    assert report.percentages()[CLEAN] == 100


@pytest.mark.parametrize(
    "line, reason",
    [
        ('{"prompt": "x", "completion": ', "invalid JSON"),
        ('["x", "y"]', "row is not a JSON object"),
        ('{"prompt": "List users"}', "missing 'completion'"),
        ('{"prompt": "List users", "completion": null}', "'completion' is not a string"),
        ('{"prompt": "List users", "completion": "   "}', "'completion' is empty"),
        ('{"question": "a", "answer": "b"}', "no prompt/completion, messages or text field"),
        ('{"text": "hello"}', "text row in a prompt/completion dataset"),
    ],
)
def test_malformed_rows(tmp_path, line, reason):
    good = [json.dumps(pc(f"q{i}", f"SELECT {i};")) for i in range(3)]
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", [*good, line]))
    row = report.rows[-1]
    assert row.status == MALFORMED
    assert row.reason.startswith(reason)


def test_invalid_utf8_is_malformed(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_bytes(json.dumps(pc("a", "SELECT 1;")).encode() + b"\n\xff\xfe\n")
    report = audit_dataset(path)
    assert statuses(report) == [(1, CLEAN), (2, MALFORMED)]


def test_blank_lines_are_skipped(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [pc("a", "SELECT 1;"), "", "   ", pc("b", "SELECT 2;")])
    assert statuses(audit_dataset(path)) == [(1, CLEAN), (4, CLEAN)]


def test_near_duplicates_ignore_case_whitespace_and_trailing_punctuation(tmp_path):
    rows = [
        pc("How many users?", "SELECT COUNT(*) FROM users;"),
        pc("how  many users", "SELECT COUNT(*) FROM users"),
        pc("How many users?", "SELECT COUNT(*) FROM orders;"),
    ]
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows))
    assert statuses(report) == [(1, CLEAN), (2, DUPLICATE), (3, CLEAN)]
    assert report.rows[1].reason == "duplicate of line 1"
    assert report.conflicting_prompts == 1


@pytest.mark.parametrize(
    "prompt, completion, reason",
    [
        ("List users", "TODO", "placeholder completion"),
        ("List users", "...", "placeholder completion"),
        ("List users", "I'm sorry, I can't help with that.", "completion is a refusal"),
        ("List users", "As an AI language model, I cannot.", "completion is a refusal"),
        ("Question: How many active users?", "How many active users?", "completion repeats"),
        ("List users", "SELECT * FROM FROM FROM FROM FROM FROM users", "degenerate repetition"),
    ],
)
def test_low_quality_rows(tmp_path, prompt, completion, reason):
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", [pc(prompt, completion)]))
    assert report.rows[0].status == LOW_QUALITY
    assert report.rows[0].reason.startswith(reason)


def test_short_answer_is_not_an_echo(tmp_path):
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", [pc("What is 2 + 3? 5", "5")]))
    assert report.rows[0].status == CLEAN


def test_chat_format(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "List users"},
                {"role": "assistant", "content": "SELECT * FROM users;"},
            ]
        },
        {"messages": [{"role": "user", "content": "List users"}]},
        {
            "messages": [
                {"role": "robot", "content": "hi"},
                {"role": "assistant", "content": "SELECT 1;"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "List users"},
                {"role": "assistant", "content": "SELECT * FROM users;"},
            ]
        },
    ]
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows))
    assert report.format == "chat"
    assert statuses(report) == [(1, CLEAN), (2, MALFORMED), (3, MALFORMED), (4, DUPLICATE)]
    assert report.rows[1].reason == "last message is not from the assistant"


def test_text_format(tmp_path):
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", [{"text": "a b c"}, {"text": "A B C"}]))
    assert report.format == "text"
    assert statuses(report) == [(1, CLEAN), (2, DUPLICATE)]


def test_percentages_sum_to_100(tmp_path):
    rows = [pc(f"q{i}", f"SELECT {i};") for i in range(5)] + [pc("q0", "SELECT 0;")] * 1 + ["{"]
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows))
    assert sum(report.percentages().values()) == 100


def test_empty_file(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("", encoding="utf-8")
    report = audit_dataset(path)
    assert report.total == 0
    assert report.format is None
    assert "0 examples" in format_report(report)


def test_write_clean_keeps_low_quality_by_default(tmp_path):
    rows = [pc("a", "SELECT 1;"), pc("a", "SELECT 1;"), pc("b", "TODO"), "{bad"]
    report = audit_dataset(write_jsonl(tmp_path / "d.jsonl", rows))
    out = tmp_path / "clean.jsonl"
    assert write_clean_dataset(report, out) == 2
    assert [json.loads(line)["prompt"] for line in out.read_text(encoding="utf-8").splitlines()] == ["a", "b"]
    assert write_clean_dataset(report, out, drop_low_quality=True) == 1


def test_cli_audit_text_and_json(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [pc("a", "SELECT 1;"), pc("a", "SELECT 1;")])
    runner = CliRunner()

    result = runner.invoke(main, ["audit", str(path)])
    assert result.exit_code == 0
    assert "50% clean · 50% duplicates · 0% low-quality · 0% malformed" in result.output
    assert "Nothing was removed" in result.output

    result = runner.invoke(main, ["audit", str(path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["counts"]["duplicate"] == 1
    assert data["issues"] == [{"line": 2, "status": "duplicate", "reason": "duplicate of line 1"}]


def test_cli_write_clean(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [pc("a", "SELECT 1;"), pc("a", "SELECT 1;")])
    out = tmp_path / "clean.jsonl"
    result = CliRunner().invoke(main, ["audit", str(path), "--write-clean", str(out)])
    assert result.exit_code == 0
    assert "Wrote 1 rows" in result.output
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_cli_refuses_to_overwrite_input(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [pc("a", "SELECT 1;")])
    result = CliRunner().invoke(main, ["audit", str(path), "--write-clean", str(path)])
    assert result.exit_code != 0
    assert "must not overwrite" in result.output


def test_cli_drop_low_quality_needs_write_clean(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [pc("a", "SELECT 1;")])
    result = CliRunner().invoke(main, ["audit", str(path), "--drop-low-quality"])
    assert result.exit_code != 0


def test_write_clean_normalizes_crlf(tmp_path):
    # Files written on Windows use CRLF; the cleaned copy must not gain blank lines.
    path = tmp_path / "d.jsonl"
    lines = [
        json.dumps(pc("a", "SELECT 1;")),
        json.dumps(pc("a", "SELECT 1;")),
        json.dumps(pc("b", "SELECT 2;")),
    ]
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    out = tmp_path / "clean.jsonl"
    assert write_clean_dataset(audit_dataset(path), out) == 2
    assert out.read_bytes() == (lines[0] + "\n" + lines[2] + "\n").encode()
