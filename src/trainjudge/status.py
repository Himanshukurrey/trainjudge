"""Live status for long-running commands, for people and coding agents alike.

`train`, `eval` and `verify` keep <run>/status.json up to date: which command
is running, which stage it's in, progress and ETA, the process ID, and every
finished stage with its duration. `trainjudge status` reads it, so anyone
(a person in another terminal, or Claude/Codex polling in the background) can
tell what's running, what finished, and whether a job died without finishing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATUS_FILE = "status.json"
RUNNING, DONE, FAILED, INTERRUPTED = "running", "done", "failed", "interrupted"

STAGE_LABELS = {
    "prepare": "Preparing run",
    "replay": "Generating replay examples",
    "train": "Training",
    "eval:baseline": "Baseline task eval",
    "eval:finetuned": "Fine-tuned task eval",
    "regression:baseline": "Baseline regression check",
    "regression:finetuned": "Fine-tuned regression check",
    "verdict": "Writing verdict and reports",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _age_s(iso: str | None) -> float | None:
    if not iso:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()


# On Windows a file can't be replaced while another process has it open, and can't be
# opened while it's being replaced. Readers and the writer both retry briefly.
_RETRIES = 20
_RETRY_DELAY_S = 0.02


def read_status(run_dir: Path) -> dict | None:
    """The job's status, or None if there's none yet (or it's mid-update on Windows)."""
    path = run_dir / STATUS_FILE
    for attempt in range(_RETRIES):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        except PermissionError:
            if attempt == _RETRIES - 1:
                return None
            time.sleep(_RETRY_DELAY_S)
    return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def effective_state(status: dict) -> str:
    """The recorded state, or "stopped" if a running job's process is gone."""
    if status["state"] == RUNNING and not pid_alive(status.get("pid")):
        return "stopped"
    return status["state"]


def notify(title: str, message: str) -> None:
    """Best-effort desktop notification (macOS only); never raises."""
    if sys.platform != "darwin" or not shutil.which("osascript"):
        return
    script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


class StatusTracker:
    """Writes status.json as a command moves through its stages.

    Use as a context manager: an exception marks the job failed, Ctrl+C marks
    it interrupted, and either way the status file says so.
    """

    def __init__(self, run_dir: Path, command: str, notify_on_finish: bool = False):
        self.run_dir = run_dir
        self.command = command
        self.notify_on_finish = notify_on_finish
        self.data = {
            "run": str(run_dir),
            "command": command,
            "state": RUNNING,
            "stage": None,
            "stage_label": None,
            "step": None,
            "total": None,
            "eta_s": None,
            "message": None,
            "pid": os.getpid(),
            "started_at": _now(),
            "updated_at": _now(),
            "stage_started_at": None,
            "stages": [],
            "result": None,
        }
        self._stage_start = time.monotonic()
        self._write()

    def _write(self) -> None:
        self.data["updated_at"] = _now()
        path = self.run_dir / STATUS_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        for attempt in range(_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _RETRIES - 1:
                    raise
                time.sleep(_RETRY_DELAY_S)

    def _close_stage(self, state: str) -> None:
        if self.data["stage"]:
            self.data["stages"].append(
                {
                    "name": self.data["stage"],
                    "label": self.data["stage_label"],
                    "state": state,
                    "duration_s": round(time.monotonic() - self._stage_start, 1),
                }
            )

    def stage(self, name: str, total: int | None = None, message: str | None = None) -> None:
        self._close_stage(DONE)
        self._stage_start = time.monotonic()
        self.data.update(
            stage=name,
            stage_label=STAGE_LABELS.get(name, name),
            step=0 if total else None,
            total=total,
            eta_s=None,
            message=message,
            stage_started_at=_now(),
        )
        self._write()

    def progress(
        self, step: int, total: int | None = None, eta_s: float | None = None, message: str | None = None
    ) -> None:
        total = total or self.data["total"]
        if eta_s is None and total and step:
            elapsed = time.monotonic() - self._stage_start
            eta_s = elapsed / step * (total - step)
        self.data.update(step=step, total=total, eta_s=round(eta_s, 1) if eta_s is not None else None)
        if message is not None:
            self.data["message"] = message
        self._write()

    def finish(self, message: str, result: str | None = None) -> None:
        self._close_stage(DONE)
        self.data.update(
            state=DONE,
            stage=None,
            stage_label=None,
            step=None,
            total=None,
            eta_s=None,
            message=message,
            result=result,
        )
        self._write()
        if self.notify_on_finish:
            notify(f"TrainJudge {self.command} finished", message)

    def fail(self, message: str, state: str = FAILED) -> None:
        self._close_stage(state)
        self.data.update(state=state, eta_s=None, message=message)
        self._write()
        if self.notify_on_finish:
            notify(f"TrainJudge {self.command} {state}", message)

    def __enter__(self) -> StatusTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            if self.data["state"] == RUNNING:
                self.finish(f"{self.command} finished")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.fail("interrupted by the user", INTERRUPTED)
        elif self.data["state"] == RUNNING:
            self.fail(str(exc) or exc_type.__name__)
        return False


def find_runs(runs_dir: Path) -> list[tuple[Path, dict]]:
    """Runs with a status file, most recently updated first."""
    found = []
    if runs_dir.is_dir():
        for child in runs_dir.iterdir():
            if child.is_dir() and (status := read_status(child)):
                found.append((child, status))

    def key(rs: tuple[Path, dict]) -> tuple[str, int]:
        # updated_at has millisecond resolution; the file's mtime breaks ties.
        return rs[1].get("updated_at") or "", (rs[0] / STATUS_FILE).stat().st_mtime_ns

    return sorted(found, key=key, reverse=True)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def progress_text(status: dict) -> str:
    if not status.get("stage_label"):
        return ""
    text = status["stage_label"]
    step, total = status.get("step"), status.get("total")
    if total:
        text += f" · {step or 0:,}/{total:,} ({(step or 0) / total:.0%})"
    if status.get("eta_s") is not None:
        text += f" · ETA {_duration(status['eta_s'])}"
    return text


STATE_MARKS = {
    RUNNING: "▶ running",
    DONE: "✓ done",
    FAILED: "✗ failed",
    INTERRUPTED: "■ interrupted",
    "stopped": "✗ stopped",
}


def format_status(run_dir: Path, status: dict) -> str:
    state = effective_state(status)
    lines = [str(run_dir), f"  Command:  trainjudge {status['command']}  ({STATE_MARKS[state]})"]
    if state == RUNNING:
        lines.append(f"  Now:      {progress_text(status)}")
        if status.get("message"):
            lines.append(f"            {status['message']}")
    elif state == "stopped":
        lines.append(f"  Stopped:  process {status.get('pid')} is gone, but the job never reported finishing")
        lines.append(f"            (last stage: {progress_text(status) or 'none'})")
    else:
        lines.append(f"  Result:   {status.get('message')}")
    lines.append(
        f"  Started:  {_duration(_age_s(status.get('started_at')))} ago · last update "
        f"{_duration(_age_s(status.get('updated_at')))} ago"
    )
    for s in status.get("stages", []):
        mark = "✓" if s["state"] == DONE else "✗"
        lines.append(f"    {mark} {s['label']} ({_duration(s['duration_s'])})")
    return "\n".join(lines)


def one_line(run_dir: Path, status: dict) -> str:
    state = effective_state(status)
    detail = progress_text(status) if state == RUNNING else (status.get("message") or "")
    return f"{run_dir.name:<34} {status['command']:<7} {STATE_MARKS[state]:<14} {detail}"
