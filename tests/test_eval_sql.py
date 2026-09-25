import json
import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainjudge import evaluation, training
from trainjudge.cli import main
from trainjudge.dataset_audit import audit_dataset
from trainjudge.eval_sql import (
    CORRECT,
    EXECUTION_ERROR,
    GOLD_ERROR,
    NO_SQL,
    TIMEOUT,
    WRONG_RESULT,
    Database,
    QueryTimeout,
    evaluate,
    extract_sql,
    results_match,
    results_match_lenient,
    score_example,
)

DEMO = Path(__file__).parent.parent / "demo" / "sql_generation"
SHOP = DEMO / "shop.sql"


@pytest.fixture(scope="module")
def db():
    return Database(SHOP)


@pytest.mark.parametrize(
    "output, expected",
    [
        ("SELECT 1;", "SELECT 1;"),
        (
            "<think>\n\n</think>\n\nSELECT full_name FROM customers;",
            "SELECT full_name FROM customers;",
        ),
        (
            "Here is the query:\n\n```sql\nSELECT a\nFROM t\nLIMIT 3;\n```\nThis returns...",
            "SELECT a\nFROM t\nLIMIT 3;",
        ),
        ("```\nSELECT a FROM t\n```", "SELECT a FROM t"),
        ("We join with that customer.\n\n```sql\nSELECT a FROM t;\n```", "SELECT a FROM t;"),
        (
            "WITH s AS (SELECT 1) SELECT * FROM s; and more text",
            "WITH s AS (SELECT 1) SELECT * FROM s;",
        ),
        ("To answer: SELECT a FROM t\n\nThis query selects a.", "SELECT a FROM t"),
        ("select a from t;", "select a from t;"),
        ("<think>should I SELECT x?</think>The answer is SELECT y FROM t;", "SELECT y FROM t;"),
    ],
)
def test_extract_sql(output, expected):
    assert extract_sql(output) == expected


@pytest.mark.parametrize(
    "output",
    ["The email is not provided in the schema.", "Ivan's email is **[insert email]**.", ""],
)
def test_extract_sql_none(output):
    assert extract_sql(output) is None


def test_results_match_order_and_floats():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert results_match([(0.1 + 0.2,)], [(0.3,)], ordered=True)
    assert results_match([(1,)], [(1.0,)], ordered=True)
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)


def test_results_match_lenient():
    gold = [("a",), ("b",)]
    assert results_match_lenient(gold, [(1, "a", 9), (2, "b", 8)])
    assert not results_match_lenient(gold, [(1, "a"), (2, "c")])
    assert not results_match_lenient([("a", 1)], [("a",)])
    assert results_match_lenient([], [])


def test_score_outcomes(db):
    gold = "SELECT COUNT(*) FROM customers WHERE city = 'Austin';"
    assert (
        score_example(db, "q", gold, "SELECT COUNT(*) FROM customers WHERE city='Austin'").outcome == CORRECT
    )
    wrong = score_example(db, "q", gold, "SELECT COUNT(*) FROM customers;")
    assert wrong.outcome == WRONG_RESULT and not wrong.lenient_correct
    error = score_example(db, "q", gold, "SELECT nope FROM customers;")
    assert error.outcome == EXECUTION_ERROR and "no such column" in error.error
    assert score_example(db, "q", gold, "I don't know.").outcome == NO_SQL
    assert score_example(db, "q", "SELECT broken FROM x;", "SELECT 1;").outcome == GOLD_ERROR


def test_extra_columns_are_wrong_but_lenient(db):
    gold = "SELECT product_name FROM products ORDER BY unit_price_cents DESC LIMIT 5;"
    pred = "SELECT product_id, product_name FROM products ORDER BY unit_price_cents DESC LIMIT 5;"
    result = score_example(db, "q", gold, pred)
    assert result.outcome == WRONG_RESULT
    assert result.lenient_correct


def test_ordered_comparison_only_for_top_level_order_by(db):
    gold = "SELECT full_name FROM customers WHERE city = 'Austin' ORDER BY full_name;"
    reversed_order = "SELECT full_name FROM customers WHERE city = 'Austin' ORDER BY full_name DESC;"
    assert score_example(db, "q", gold, reversed_order).outcome == WRONG_RESULT
    unordered_gold = (
        "SELECT full_name FROM customers WHERE customer_id IN "
        "(SELECT customer_id FROM orders ORDER BY order_date)"
    )
    assert (
        score_example(db, "q", unordered_gold, unordered_gold + " ORDER BY full_name DESC").outcome == CORRECT
    )


def test_database_is_read_only(db):
    # Only SELECT/WITH statements are extracted, and the connection rejects writes anyway.
    result = score_example(db, "q", "SELECT COUNT(*) FROM customers;", "DELETE FROM customers;")
    assert result.outcome == NO_SQL
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        db.execute("DELETE FROM customers")
    assert db.execute("SELECT COUNT(*) FROM customers")[0][0] == 100


def test_slow_query_times_out(db):
    slow = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT MAX(x) FROM n;"
    result = score_example(db, "q", "SELECT 1;", slow)
    assert result.outcome in (TIMEOUT, EXECUTION_ERROR)
    start = time.monotonic()
    with pytest.raises(QueryTimeout):
        db.execute(slow, timeout_s=0.2)
    assert time.monotonic() - start < 2


def test_database_from_sqlite_file(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.execute("INSERT INTO t VALUES (7)")
    conn.commit()
    conn.close()
    assert Database(path).execute("SELECT a FROM t") == [(7,)]


def test_gold_queries_in_demo_all_run(db):
    report = audit_dataset(DEMO / "data.jsonl")
    golds = {r.example.completion for r in report.rows if r.status == "clean" and r.example is not None}
    assert len(golds) > 400
    for sql in golds:
        assert db.execute(sql)


def test_evaluate_accuracy(db):
    rows = [
        {"prompt": "a", "completion": "SELECT 1;"},
        {"prompt": "b", "completion": "SELECT 2;"},
        {"prompt": "c", "completion": "SELECT bad FROM nowhere;"},
    ]
    report = evaluate(db, rows, ["SELECT 1;", "no idea", "SELECT 3;"])
    assert report.accuracy == 0.5  # gold errors are excluded
    assert report.outcomes()[GOLD_ERROR] == 1
    with pytest.raises(ValueError):
        evaluate(db, rows, ["x"])


def fake_generate(answers):
    def generate(model, prompts, adapter_path=None, decoding=None, on_progress=None):
        return [answers(p, adapter_path) for p in prompts]

    return generate


@pytest.fixture
def trained_run(tmp_path):
    prepared = training.prepare_run(DEMO / "data.jsonl", "Qwen3-0.6B", tmp_path)
    adapters = prepared.run_dir / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"")
    return prepared.run_dir


def test_evaluate_target_writes_results(trained_run):
    gold = {r["prompt"]: r["completion"] for r in evaluation.test_rows(trained_run)}
    answers = fake_generate(lambda p, adapter: gold[p] if adapter else "I don't know.")

    base = evaluation.evaluate_target(trained_run, "baseline", SHOP, generate=answers)
    tuned = evaluation.evaluate_target(trained_run, "finetuned", SHOP, generate=answers)
    assert base["accuracy"] == 0.0 and base["outcomes"]["no_sql"] == base["scored"]
    assert tuned["accuracy"] == 1.0
    assert tuned["adapter_path"].endswith("adapters")
    assert len(tuned["database"]["sha256"]) == 64

    saved = json.loads((trained_run / "eval" / "finetuned.json").read_text(encoding="utf-8"))
    assert len(saved["examples"]) == len(gold)
    record = json.loads((trained_run / "run.json").read_text(encoding="utf-8"))
    assert record["task"] == {"type": "sql", "database": str(SHOP)}
    assert set(record["evals"]) == {"baseline", "finetuned"}


def test_evaluate_target_limit(trained_run):
    answers = fake_generate(lambda p, adapter: "SELECT 1;")
    result = evaluation.evaluate_target(trained_run, "baseline", SHOP, limit=5, generate=answers)
    assert result["test_split"] == {
        "path": str(trained_run / "data" / "test.jsonl"),
        "rows": 5,
        "limited": True,
    }


def test_finetuned_eval_needs_adapter(tmp_path):
    prepared = training.prepare_run(DEMO / "data.jsonl", "m", tmp_path)
    with pytest.raises(evaluation.runs.RunError, match="no trained adapter"):
        evaluation.evaluate_target(
            prepared.run_dir, "finetuned", SHOP, generate=fake_generate(lambda p, a: "")
        )


def test_cli_eval_requires_db_first_time(trained_run):
    result = CliRunner().invoke(main, ["eval", str(trained_run), "--json"])
    assert result.exit_code != 0
    assert "needs --db" in result.output
