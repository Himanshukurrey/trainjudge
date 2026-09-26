import pytest

from trainjudge import runs, training
from trainjudge.regression_check import build_suite


@pytest.fixture
def trained_run_from(tmp_path):
    """Prepare a run from a dataset and mark it trained, with an empty adapter file."""

    def make(dataset, goal="the task", **options):
        prepared = training.prepare_run(dataset, "Qwen3-0.6B", tmp_path, goal=goal, **options)
        adapters = prepared.run_dir / "adapters"
        adapters.mkdir()
        (adapters / "adapters.safetensors").write_bytes(b"")
        record = runs.read_run_json(prepared.run_dir)
        record["status"] = "trained"
        record["training"] = {"train_loss_drop_pct": 90.0, "duration_s": 60}
        runs.write_run_json(prepared.run_dir, record)
        return prepared.run_dir

    return make


# An answer that passes each kind of regression check.
PASSING = {
    "bullets": "- a\n- b\n- c", "lowercase": "fine.", "max_words": "short.",
    "ends_with": "Ok. That is all.",
    "json_keys": '{"name": "x", "color": "y"}', "json_list": '["a", "b", "c", "d"]',
    "numbered": "1. a\n2. b\n3. c", "uncertain": "I don't know.",
}  # fmt: skip


@pytest.fixture
def fake_generator():
    """A fake backend: task prompts are answered by `answer(prompt, adapter_path)`, the
    regression suite always passes."""
    suite = {i.prompt: i for i in build_suite()}

    def make(answer):
        def generate(model, prompts, adapter_path=None, decoding=None, on_progress=None):
            out = []
            for p in prompts:
                item = suite.get(p)
                if item is None:
                    out.append(answer(p, adapter_path))
                else:
                    out.append(str(item.arg) if item.check == "number" else PASSING[item.check])
            return out

        return generate

    return make
