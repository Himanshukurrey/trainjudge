import json

import pytest
from click.testing import CliRunner

from trainjudge import status as st
from trainjudge.cli import main


def test_tracker_records_stages_and_finish(tmp_path):
    with st.StatusTracker(tmp_path, "verify") as tracker:
        tracker.stage("eval:baseline", total=130)
        tracker.progress(64, message="halfway")
        data = st.read_status(tmp_path)
        assert data["state"] == st.RUNNING
        assert (data["stage"], data["stage_label"]) == ("eval:baseline", "Baseline task eval")
        assert (data["step"], data["total"], data["message"]) == (64, 130, "halfway")
        assert data["eta_s"] is not None
        tracker.stage("regression:baseline", total=60)
        tracker.finish("IMPROVED: task accuracy improved", result="IMPROVED")

    data = st.read_status(tmp_path)
    assert data["state"] == st.DONE and data["result"] == "IMPROVED"
    assert [s["name"] for s in data["stages"]] == ["eval:baseline", "regression:baseline"]
    assert all(s["state"] == st.DONE for s in data["stages"])


def test_tracker_marks_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with st.StatusTracker(tmp_path, "train") as tracker:
            tracker.stage("train", total=100)
            raise RuntimeError("out of memory")
    data = st.read_status(tmp_path)
    assert data["state"] == st.FAILED
    assert data["message"] == "out of memory"
    assert data["stages"][-1] == {
        "name": "train",
        "label": "Training",
        "state": st.FAILED,
        "duration_s": data["stages"][-1]["duration_s"],
    }


def test_tracker_marks_interrupt(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with st.StatusTracker(tmp_path, "train"):
            raise KeyboardInterrupt
    assert st.read_status(tmp_path)["state"] == st.INTERRUPTED


def test_dead_process_reads_as_stopped(tmp_path):
    st.StatusTracker(tmp_path, "train").stage("train", total=10)
    data = st.read_status(tmp_path)
    assert st.effective_state(data) == st.RUNNING  # this process is alive
    data["pid"] = 999_999_999
    assert st.effective_state(data) == "stopped"
    assert "never reported finishing" in st.format_status(tmp_path, data)


def test_progress_text():
    status = {"stage_label": "Training", "step": 30, "total": 150, "eta_s": 95}
    assert st.progress_text(status) == "Training · 30/150 (20%) · ETA 1m 35s"
    assert st.progress_text({"stage_label": None}) == ""


def test_find_runs_orders_by_update(tmp_path):
    for name, updated in (("old", "2026-09-24T10:00:00+00:00"), ("new", "2026-09-24T12:00:00+00:00")):
        run = tmp_path / name
        run.mkdir()
        (run / st.STATUS_FILE).write_text(json.dumps({"updated_at": updated}), encoding="utf-8")
    (tmp_path / "no-status").mkdir()
    assert [p.name for p, _ in st.find_runs(tmp_path)] == ["new", "old"]


def test_cli_status_latest_and_all(tmp_path):
    runs_dir = tmp_path / "runs"
    for name in ("a", "b"):
        (runs_dir / name).mkdir(parents=True)
        tracker = st.StatusTracker(runs_dir / name, "verify")
        tracker.stage("eval:baseline", total=10)
        if name == "b":
            tracker.finish("REJECTED: no gain", result="REJECTED")

    runner = CliRunner()
    result = runner.invoke(main, ["status", "--runs-dir", str(runs_dir)])
    assert result.exit_code == 0, result.output
    assert "b" in result.output.splitlines()[0] and "✓ done" in result.output

    result = runner.invoke(main, ["status", str(runs_dir / "a"), "--json"])
    data = json.loads(result.output)
    assert data["state"] == st.RUNNING and data["stage"] == "eval:baseline"

    result = runner.invoke(main, ["status", "--all", "--runs-dir", str(runs_dir)])
    assert len(result.output.strip().splitlines()) == 2


def test_cli_status_watch_exits_when_done(tmp_path):
    tracker = st.StatusTracker(tmp_path, "train")
    tracker.finish("trained in 5m")
    result = CliRunner().invoke(main, ["status", str(tmp_path), "--watch", "--interval", "0.01"])
    assert result.exit_code == 0
    assert "trained in 5m" in result.output


def test_cli_status_without_runs(tmp_path):
    result = CliRunner().invoke(main, ["status", "--runs-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "No runs with status" in result.output


def _run_job(tmp_path, fail=False):
    """Simulate a job in a thread: two stages with progress, then finish or fail."""
    import threading
    import time

    started = threading.Event()

    def job():
        tracker = st.StatusTracker(tmp_path, "verify")
        started.set()
        for stage, total in (("eval:baseline", 8), ("regression:baseline", 4)):
            tracker.stage(stage, total=total)
            for step in range(1, total + 1):
                time.sleep(0.02)
                tracker.progress(step, total)
        if fail:
            tracker.fail("out of memory")
        else:
            tracker.finish("IMPROVED: task accuracy improved", result="IMPROVED")

    thread = threading.Thread(target=job)
    thread.start()
    started.wait()
    return thread


def test_milestones_follow_a_job_to_the_end(tmp_path):
    thread = _run_job(tmp_path)
    result = CliRunner().invoke(
        main, ["status", str(tmp_path), "--watch", "--milestones", "--interval", "0.005"]
    )
    thread.join()
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert all(line.startswith("[trainjudge verify] ") for line in lines)
    text = "\n".join(lines)
    assert "▶ Baseline task eval started (8 items)" in text
    assert "✓ Baseline task eval done" in text
    assert "✓ Baseline regression check done" in text
    assert lines[-1] == "[trainjudge verify] ✓ finished: IMPROVED: task accuracy improved"
    # Milestones only: no line per step.
    assert len(lines) <= 12


def test_milestones_report_failure(tmp_path):
    thread = _run_job(tmp_path, fail=True)
    result = CliRunner().invoke(
        main, ["status", str(tmp_path), "--watch", "--milestones", "--interval", "0.005"]
    )
    thread.join()
    assert result.exit_code == 1
    assert result.output.strip().splitlines()[-1] == "[trainjudge verify] ✗ failed: out of memory"


def test_milestones_report_a_dead_process(tmp_path):
    tracker = st.StatusTracker(tmp_path, "train")
    tracker.stage("train", total=10)
    data = st.read_status(tmp_path)
    data["pid"] = 999_999_999
    (tmp_path / st.STATUS_FILE).write_text(json.dumps(data), encoding="utf-8")
    result = CliRunner().invoke(
        main, ["status", str(tmp_path), "--watch", "--milestones", "--interval", "0.01"]
    )
    assert result.exit_code == 1
    assert "✗ stopped (process gone)" in result.output


def test_status_file_survives_windows_style_locking(tmp_path, monkeypatch):
    """Windows refuses to replace or read a file that's mid-update; both sides retry."""
    import os as os_module
    from pathlib import Path as PathClass

    tracker = st.StatusTracker(tmp_path, "train")
    real_replace, real_read = os_module.replace, PathClass.read_text
    busy = {"replace": 2, "read": 2}

    def flaky_replace(src, dst):
        if busy["replace"]:
            busy["replace"] -= 1
            raise PermissionError(13, "Permission denied")
        return real_replace(src, dst)

    def flaky_read(self, *args, **kwargs):
        if self.name == st.STATUS_FILE and busy["read"]:
            busy["read"] -= 1
            raise PermissionError(13, "Permission denied")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(st.os, "replace", flaky_replace)
    monkeypatch.setattr(PathClass, "read_text", flaky_read)
    tracker.stage("train", total=10)  # would raise without the writer's retry
    assert st.read_status(tmp_path)["stage"] == "train"  # would raise without the reader's retry
    assert busy == {"replace": 0, "read": 0}
