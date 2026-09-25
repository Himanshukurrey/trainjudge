import json
import sys
from pathlib import Path

import pytest

from trainjudge import backends, mlx_backend, torch_backend, training
from trainjudge.backend_base import LoraConfig

DEMO = Path(__file__).parent.parent / "demo"
SQL = DEMO / "sql_generation" / "data.jsonl"


def test_resolve_prefers_mlx_then_torch(monkeypatch):
    monkeypatch.setattr(mlx_backend, "unavailable_reason", lambda: None)
    assert backends.resolve("auto") == "mlx"
    monkeypatch.setattr(mlx_backend, "unavailable_reason", lambda: "no mlx")
    monkeypatch.setattr(torch_backend, "unavailable_reason", lambda: None)
    assert backends.resolve("auto") == "torch"
    assert backends.resolve("mlx") == "mlx"
    with pytest.raises(ValueError):
        backends.resolve("tpu")


def test_resolve_with_nothing_installed_suggests_by_platform(monkeypatch):
    monkeypatch.setattr(mlx_backend, "unavailable_reason", lambda: "no mlx")
    monkeypatch.setattr(torch_backend, "unavailable_reason", lambda: "no torch")
    monkeypatch.setattr(sys, "platform", "win32")
    assert backends.resolve("auto") == "torch"


def test_torch_run_records_backend_and_config(tmp_path):
    prepared = training.prepare_run(SQL, "Qwen3-0.6B", tmp_path, iters=20, backend="torch", device="cpu")
    record = json.loads((prepared.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["backend"]["name"] == "torch" and record["backend"]["device"] == "cpu"
    assert "torch_version" in record["backend"]
    config = json.loads((prepared.run_dir / "torch_config.json").read_text(encoding="utf-8"))
    assert config["device"] == "cpu" and config["rank"] == 16 and config["mask_prompt"] is True
    assert config["data"] == str(prepared.run_dir / "data")
    assert prepared.command[1:3] == ["-m", "trainjudge.torch_train"]


def test_mlx_run_keeps_its_config(tmp_path):
    prepared = training.prepare_run(SQL, "Qwen3-0.6B", tmp_path, iters=20, backend="mlx")
    assert json.loads((prepared.run_dir / "run.json").read_text(encoding="utf-8"))["backend"]["name"] == "mlx"
    assert (prepared.run_dir / "mlx_config.yaml").exists()
    assert "--mask-prompt" in prepared.command


def test_adapter_ready_accepts_either_format(tmp_path):
    assert not backends.adapter_ready(tmp_path / "missing")
    for name in ("adapters.safetensors", "adapter_model.safetensors"):
        folder = tmp_path / name.split(".")[0]
        folder.mkdir()
        assert not backends.adapter_ready(folder)
        (folder / name).write_bytes(b"")
        assert backends.adapter_ready(folder)


def test_generator_follows_the_run():
    assert backends.generator({"backend": {"name": "mlx"}}) is mlx_backend.generate_outputs
    gen = backends.generator({"backend": {"name": "torch", "device": "cuda"}})
    assert gen.func is torch_backend.generate_outputs and gen.keywords == {"device": "cuda"}
    assert backends.generator({}) is mlx_backend.generate_outputs  # runs from before backends existed


def test_torch_command_and_config_file(tmp_path):
    cfg = LoraConfig(model="m", iters=5, rank=4, num_layers=2)
    path = torch_backend.write_config(cfg, tmp_path, "mps")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["device"] == "mps" and data["iters"] == 5 and data["num_layers"] == 2
    assert torch_backend.build_command(cfg, path, python="py") == ["py", "-m", "trainjudge.torch_train",
                                                                   "--config", str(path)]  # fmt: skip


def test_torch_training_failure_points_at_its_log(tmp_path):
    from trainjudge.backend_base import TrainingFailed

    script = tmp_path / "fail.py"
    script.write_text("import sys\nprint('CUDA out of memory', flush=True)\nsys.exit(1)\n", encoding="utf-8")
    with pytest.raises(TrainingFailed) as excinfo:
        torch_backend.train([sys.executable, str(script)], tmp_path)
    assert excinfo.value.log_path == tmp_path / "logs" / "torch.log"
    assert "CUDA out of memory" in excinfo.value.log_tail
    assert "PyTorch training exited" in str(excinfo.value)
