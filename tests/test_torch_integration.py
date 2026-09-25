"""Real training and generation with the PyTorch backend, on CPU with a tiny model.

Skipped unless TRAINJUDGE_TORCH_INTEGRATION=1 and torch/transformers/peft are
installed; CI runs it in its own job. It downloads a ~270 MB model the first time.
Everything else in the suite uses fake backends.
"""

import json
import os

import pytest

from trainjudge import backends, torch_backend, training

pytestmark = pytest.mark.skipif(
    os.environ.get("TRAINJUDGE_TORCH_INTEGRATION") != "1" or torch_backend.unavailable_reason() is not None,
    reason="set TRAINJUDGE_TORCH_INTEGRATION=1 with torch, transformers and peft installed",
)

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_train_and_generate_on_cpu(tmp_path):
    rows = [
        {"prompt": f"Tag: question {i} about {topic}", "completion": json.dumps({"topic": topic})}
        for i, topic in enumerate(["algebra", "genetics", "mechanics", "history"] * 15)
    ]
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    prepared = training.prepare_run(
        dataset, MODEL, tmp_path / "runs", iters=4, batch_size=2, rank=4, num_layers=2,
        backend="torch", device="cpu",
    )  # fmt: skip
    events = []
    summary = training.run_training(prepared, events.append)
    assert summary.iters_completed == 4
    assert {e["type"] for e in events} == {"train", "val"}
    assert backends.adapter_ready(prepared.run_dir / "adapters")
    assert (prepared.run_dir / "adapters" / "adapter_config.json").exists()

    outputs = torch_backend.generate_outputs(
        MODEL, ["Tag: question 1 about algebra", "Tag: question 2 about genetics"],
        adapter_path=prepared.run_dir / "adapters", device="cpu",
        decoding=torch_backend.DecodingConfig(max_tokens=16, batch_size=2),
    )  # fmt: skip
    assert len(outputs) == 2 and all(isinstance(o, str) for o in outputs)


def test_prompt_masking_matches_generation_prompt():
    from transformers import AutoTokenizer

    from trainjudge.torch_train import encode

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids, prompt_len = encode(tokenizer, {"prompt": "Hi", "completion": "Hello!"}, True, 512)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    assert ids[:prompt_len] == list(prompt_ids)  # the loss starts exactly at the answer
    assert "Hello!" in tokenizer.decode(ids[prompt_len:])


def test_generation_releases_accelerator_memory(monkeypatch):
    """Each generation pass must hand its memory back, or the next model load runs out (seen on a T4)."""
    released = []
    real = torch_backend.release_memory
    monkeypatch.setattr(
        torch_backend, "release_memory", lambda device: (released.append(device), real(device))
    )
    torch_backend.generate_outputs(
        MODEL, ["Hi"], device="cpu", decoding=torch_backend.DecodingConfig(max_tokens=4, batch_size=1)
    )
    assert released == ["cpu"]
