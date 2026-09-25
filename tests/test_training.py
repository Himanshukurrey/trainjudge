import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from trainjudge import mlx_backend, runs, training
from trainjudge.cli import main
from trainjudge.dataset_audit import audit_dataset, normalize

DEMO = Path(__file__).parent.parent / "demo"
SQL = DEMO / "sql_generation" / "data.jsonl"

TRAIN_LINE = (
    "Iter {i}: Train loss {loss}, Learning Rate 5.000e-05, It/sec 2.500, Tokens/sec 300.0, "
    "Trained Tokens {tokens}, Peak mem 1.250 GB"
)
VAL_LINE = "Iter {i}: Val loss {loss}, Val took 1.5s"


def completions(path):
    return {normalize(json.loads(line)["completion"]) for line in path.open(encoding="utf-8")}


def test_prepare_run_splits_sql_demo_without_leakage(tmp_path):
    prepared = training.prepare_run(SQL, "Qwen3-0.6B", tmp_path, backend="mlx")
    run_dir = prepared.run_dir
    assert run_dir.name == f"{datetime.now().astimezone().date().isoformat()}-sql_generation"
    assert prepared.config.model == "Qwen/Qwen3-0.6B"
    assert prepared.dropped == {"duplicate": 330, "malformed": 75, "low_quality": 125}

    sizes = prepared.splits.sizes()
    assert sum(sizes.values()) == 1300
    assert sizes["test"] >= 130 and sizes["valid"] >= 130

    data = run_dir / "data"
    train, valid, test = (completions(data / f"{s}.jsonl") for s in ("train", "valid", "test"))
    assert not train & test and not train & valid and not valid & test

    first = json.loads((data / "train.jsonl").open(encoding="utf-8").readline())
    assert set(first) == {"prompt", "completion"}

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "prepared"
    assert record["prep"]["splits"] == sizes
    assert len(record["dataset"]["sha256"]) == 64

    config = json.loads((run_dir / "mlx_config.yaml").read_text(encoding="utf-8"))
    assert config["lora_parameters"]["rank"] == 16
    assert config["data"] == str(data)
    assert "--mask-prompt" in prepared.command and "--train" in prepared.command


def test_split_is_deterministic(tmp_path):
    a = training.prepare_run(SQL, "m", tmp_path / "a", seed=3).splits
    b = training.prepare_run(SQL, "m", tmp_path / "b", seed=3).splits
    c = training.prepare_run(SQL, "m", tmp_path / "c", seed=4).splits
    assert a.test == b.test
    assert a.test != c.test


def test_epochs_and_iters(tmp_path):
    p = training.prepare_run(SQL, "m", tmp_path, epochs=1, batch_size=8)
    assert p.config.iters == -(-len(p.splits.train) // 8)
    p = training.prepare_run(SQL, "m", tmp_path, iters=100, batch_size=4)
    assert p.config.iters == 100
    assert p.epochs == round(400 / len(p.splits.train), 2)


def test_keep_low_quality(tmp_path):
    p = training.prepare_run(SQL, "m", tmp_path, keep_low_quality=True)
    assert "low_quality" not in p.dropped
    assert sum(p.splits.sizes().values()) == 1425


def test_sensitive_data_is_refused_unless_allowed(tmp_path):
    dataset = DEMO / "domains" / "bfsi" / "transactions" / "data.jsonl"
    with pytest.raises(training.SensitiveDataError, match="card number"):
        training.prepare_run(dataset, "m", tmp_path)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())
    training.prepare_run(dataset, "m", tmp_path, allow_sensitive_data=True)


def test_new_run_dir_names(tmp_path):
    day = date(2026, 9, 24)
    assert runs.new_run_dir(tmp_path, Path("x/sql_pairs.jsonl"), day).name == "2026-09-24-sql_pairs"
    assert runs.new_run_dir(tmp_path, Path("demo/policy/data.jsonl"), day).name == "2026-09-24-policy"
    (tmp_path / "2026-09-24-policy").mkdir()
    assert runs.new_run_dir(tmp_path, Path("demo/policy/data.jsonl"), day).name == "2026-09-24-policy-2"


def test_split_rejects_tiny_datasets():
    rows = [("a", {"prompt": "p", "completion": "a"}), ("b", {"prompt": "p", "completion": "b"})]
    with pytest.raises(runs.RunError, match="at least 3"):
        runs.split_rows(rows, 0.1, 0.1, 0)


def test_chat_rows_keep_messages_only(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": i,
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ],
                }
            )
            + "\n"
            for i in range(10)
        ),
        encoding="utf-8",
    )
    rows = runs.training_rows(audit_dataset(path))
    assert all(set(row) == {"messages"} for _, row in rows)


def test_parse_line():
    event = mlx_backend.parse_line(TRAIN_LINE.format(i=10, loss=2.5, tokens=1200))
    assert event == {
        "type": "train",
        "iter": 10,
        "loss": 2.5,
        "learning_rate": 5e-05,
        "it_per_sec": 2.5,
        "tokens_per_sec": 300.0,
        "trained_tokens": 1200,
        "peak_mem_gb": 1.25,
    }
    assert mlx_backend.parse_line(VAL_LINE.format(i=1, loss=3.1))["loss"] == 3.1
    assert mlx_backend.parse_line("Loading pretrained model") is None


def fake_backend(tmp_path, lines, exit_code=0):
    script = tmp_path / "fake_mlx.py"
    script.write_text(
        "import sys\n"
        + "".join(f"print({line!r}, flush=True)\n" for line in lines)
        + f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_run_training_records_summary(tmp_path):
    prepared = training.prepare_run(SQL, "m", tmp_path / "runs", iters=20)
    prepared.command = fake_backend(
        tmp_path,
        [
            "Loading pretrained model",
            VAL_LINE.format(i=1, loss=3.0),
            TRAIN_LINE.format(i=10, loss=2.0, tokens=500),
            TRAIN_LINE.format(i=20, loss=0.5, tokens=1000),
            VAL_LINE.format(i=20, loss=0.8),
        ],
    )
    events = []
    summary = training.run_training(prepared, events.append)

    assert [e["type"] for e in events] == ["val", "train", "train", "val"]
    assert summary.iters_completed == 20
    assert summary.train_loss_drop_pct == 75.0
    assert (summary.first_val_loss, summary.final_val_loss) == (3.0, 0.8)

    record = runs.read_run_json(prepared.run_dir)
    assert record["status"] == "trained"
    assert record["training"]["train_loss_drop_pct"] == 75.0
    log = (prepared.run_dir / "logs" / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 4


def test_run_training_failure_is_recorded(tmp_path):
    prepared = training.prepare_run(SQL, "m", tmp_path / "runs", iters=20)
    prepared.command = fake_backend(tmp_path, ["Traceback: out of memory"], exit_code=1)
    with pytest.raises(mlx_backend.TrainingFailed) as excinfo:
        training.run_training(prepared)
    assert "out of memory" in excinfo.value.log_tail
    assert runs.read_run_json(prepared.run_dir)["status"] == "failed"


def test_cli_train_dry_run(tmp_path):
    result = CliRunner().invoke(
        main,
        [
            "train",
            "--dataset",
            str(SQL),
            "--model",
            "Qwen3-0.6B",
            "--runs-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1,830 rows → 1,300 usable" in result.output
    assert "Dry run: not training" in result.output
    assert len(list(tmp_path.iterdir())) == 1


def test_cli_train_refuses_sensitive_data(tmp_path):
    result = CliRunner().invoke(
        main,
        [
            "train",
            "--dataset",
            str(DEMO / "domains" / "bfsi" / "transactions" / "data.jsonl"),
            "--model",
            "m",
            "--runs-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "--allow-sensitive-data" in result.output


def test_replay_prompts_are_disjoint_from_regression_suite():
    from trainjudge import regression_check, replay

    prompts = replay.replay_prompts(replay.max_replay())
    suite = {i.prompt for i in regression_check.build_suite()}
    assert len(prompts) == len(set(prompts)) == replay.max_replay() == 208
    assert not set(prompts) & suite
    assert not {t for t in replay.TOPICS} & set(regression_check.TOPICS)
    # A small limit still covers every template.
    first = replay.replay_prompts(len(replay.TEMPLATES) + 1)
    assert len({replay.template_index(p) for p in first}) == len(replay.TEMPLATES) + 1


def test_add_replay_appends_to_train_only(tmp_path):
    prepared = training.prepare_run(SQL, "m", tmp_path, epochs=1, batch_size=4, replay_count=10)
    assert prepared.config.iters == -(-(len(prepared.splits.train) + 10) // 4)
    data = prepared.run_dir / "data"
    before = {s: (data / f"{s}.jsonl").read_text(encoding="utf-8") for s in ("train", "valid", "test")}

    def generate(model, prompts, adapter_path=None, decoding=None, on_progress=None):
        assert adapter_path is None  # replay answers come from the base model
        return [f"answer {i}" if i else "   " for i in range(len(prompts))]

    added = training.add_replay(prepared, generate=generate)
    assert added == 9  # the blank answer is skipped
    train = (data / "train.jsonl").read_text(encoding="utf-8")
    assert train.startswith(before["train"])
    assert len(train.splitlines()) == len(before["train"].splitlines()) + 9
    assert (data / "test.jsonl").read_text(encoding="utf-8") == before["test"]
    assert runs.read_run_json(prepared.run_dir)["prep"]["replay"]["added"] == 9


def test_replay_count_is_bounded(tmp_path):
    with pytest.raises(runs.RunError, match="--replay"):
        training.prepare_run(SQL, "m", tmp_path, replay_count=10_000)


def test_mask_sensitive_trains_on_masked_rows(tmp_path):
    dataset = DEMO / "domains" / "healthcare" / "clinical_coding" / "data.jsonl"
    prepared = training.prepare_run(dataset, "m", tmp_path, mask_sensitive=True)
    assert prepared.record["prep"]["masked"] == {"date of birth": 3, "medical record number": 3}
    text = "".join((prepared.run_dir / "data" / f"{s}.jsonl").read_text(encoding="utf-8")
                   for s in ("train", "valid", "test"))  # fmt: skip
    assert text.count("[MRN]") == 3 and text.count("[DOB]") == 3
    assert audit_dataset(prepared.run_dir / "data" / "train.jsonl").sensitive == {}


def test_label_like_answers_are_split_per_row(tmp_path):
    dataset = DEMO / "domains" / "education" / "question_tagging" / "data.jsonl"
    prepared = training.prepare_run(dataset, "m", tmp_path)
    assert prepared.splits.method == runs.PER_ROW
    train_labels = {row["completion"] for row in prepared.splits.train}
    assert {row["completion"] for row in prepared.splits.test} <= train_labels  # no unseen labels


def test_paraphrase_style_answers_stay_grouped(tmp_path):
    assert training.prepare_run(SQL, "m", tmp_path).splits.method == runs.GROUPED
