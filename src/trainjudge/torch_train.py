"""Training script for the PyTorch backend.

    python -m trainjudge.torch_train --config <run>/torch_config.json

Loads the base model with transformers, adds LoRA adapters with peft, trains
on <run>/data/train.jsonl with the loss on completions only, and saves the
adapter to <run>/adapters. Progress is printed in the shared format (see
backend_base), so TrainJudge's live progress, status and summary work the same
as with mlx-lm.

Deliberately small and dependency-light (no TRL or accelerate), so their API
changes can't break training. Prompts are rendered exactly like mlx-lm does:
the chat template over the full conversation, with the loss masked up to the
generation prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def log(message: str) -> None:
    print(message, flush=True)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def encode(tokenizer, row: dict, mask_prompt: bool, max_len: int) -> tuple[list[int], int]:
    """Token ids for one example and how many leading tokens are prompt (not trained on)."""
    if "text" in row and "prompt" not in row and "messages" not in row:
        ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
        return ids[:max_len], 0
    if "messages" in row:
        messages = row["messages"]
    else:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False))
    prompt_len = 0
    if mask_prompt:
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        prompt_len = len(prefix)
    return ids[:max_len], min(prompt_len, max_len)


def batches(examples: list, batch_size: int, rng: random.Random):
    """Endless shuffled batches, reshuffled every epoch."""
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1 or 1, batch_size):
            yield [examples[j] for j in order[i : i + batch_size]]


def collate(batch, pad_id: int, torch, device):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention = [], [], []
    for ids, prompt_len in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        labels.append([-100] * prompt_len + ids[prompt_len:] + [-100] * pad)
    as_tensor = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return as_tensor(input_ids), as_tensor(attention), as_tensor(labels)


def pick_device(requested: str, torch) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, torch):
    # bf16 on Ampere or newer NVIDIA GPUs; fp32 elsewhere (the T4 and MPS lack reliable bf16).
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32


def load_model(name: str, dtype, device: str):
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model.to(device)


def lora_targets(model, torch) -> list[str] | str:
    names = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    return ATTENTION_PROJECTIONS if set(ATTENTION_PROJECTIONS) <= names else "all-linear"


def peak_memory_gb(device: str, torch) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps" and hasattr(torch, "mps"):
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text(encoding="utf-8"))

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"], torch)
    dtype = pick_dtype(device, torch)
    name = torch.cuda.get_device_name(0) if device == "cuda" else device
    log(f"Loading pretrained model {cfg['model']} on {name} ({str(dtype).replace('torch.', '')})")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = load_model(cfg["model"], dtype, device)

    layers = model.config.num_hidden_layers
    targets = lora_targets(model, torch)
    lora_kwargs = dict(
        r=cfg["rank"],
        lora_alpha=cfg["scale"] * cfg["rank"],  # mlx-lm scales the LoRA output by `scale`
        lora_dropout=cfg["dropout"],
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    if 0 < cfg["num_layers"] < layers:
        lora_kwargs.update(layers_to_transform=list(range(layers - cfg["num_layers"], layers)))
        if targets == ATTENTION_PROJECTIONS:
            lora_kwargs.update(layers_pattern="layers")
    try:
        model = get_peft_model(model, LoraConfig(**lora_kwargs))
    except ImportError as e:
        if "torchao" in str(e):
            # peft refuses old torchao versions whenever torchao is installed (Colab ships one).
            # TrainJudge doesn't use torchao, so removing it is the simplest fix.
            raise SystemExit(
                f"peft can't load because of an incompatible torchao: {e}\n"
                "TrainJudge doesn't use torchao. Fix it with `pip uninstall torchao` "
                "(or `pip install -U torchao`), then train again."
            ) from e
        raise
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable parameters: {trainable / total:.3%} ({trainable / 1e6:.3f}M/{total / 1e6:.3f}M)")

    data = Path(cfg["data"])
    max_len = cfg["max_seq_length"]
    train = [encode(tokenizer, r, cfg["mask_prompt"], max_len) for r in load_rows(data / "train.jsonl")]
    valid = [encode(tokenizer, r, cfg["mask_prompt"], max_len) for r in load_rows(data / "valid.jsonl")]
    batch_size = max(1, min(cfg["batch_size"], len(train)))
    val_batches = cfg["val_batches"] if cfg["val_batches"] > 0 else len(valid)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])

    def validate(step: int) -> None:
        model.eval()
        start, loss_sum, tokens = time.monotonic(), 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(valid), val_batches * batch_size), batch_size):
                ids, attention, labels = collate(valid[i : i + batch_size], pad_id, torch, device)
                n = int((labels[:, 1:] != -100).sum())
                if n:
                    loss_sum += float(model(input_ids=ids, attention_mask=attention, labels=labels).loss) * n
                    tokens += n
        model.train()
        log(
            f"Iter {step}: Val loss {loss_sum / max(tokens, 1):.3f}, Val took {time.monotonic() - start:.3f}s"
        )

    log(f"Starting training..., iters: {cfg['iters']}")
    validate(1)
    model.train()
    stream = batches(train, batch_size, random.Random(cfg["seed"]))
    trained_tokens = 0
    window_losses, window_tokens, window_start, window_steps = [], 0, time.monotonic(), 0
    for step in range(1, cfg["iters"] + 1):
        ids, attention, labels = collate(next(stream), pad_id, torch, device)
        loss = model(input_ids=ids, attention_mask=attention, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        n = int((labels[:, 1:] != -100).sum())
        trained_tokens += n
        window_tokens += n
        window_steps += 1
        window_losses.append(float(loss))
        if step % cfg["steps_per_report"] == 0 or step == cfg["iters"]:
            elapsed = max(time.monotonic() - window_start, 1e-9)
            log(
                f"Iter {step}: Train loss {sum(window_losses) / len(window_losses):.3f}, "
                f"Learning Rate {cfg['learning_rate']:.3e}, It/sec {window_steps / elapsed:.3f}, "
                f"Tokens/sec {window_tokens / elapsed:.3f}, Trained Tokens {trained_tokens}, "
                f"Peak mem {peak_memory_gb(device, torch):.3f} GB"
            )
            window_losses, window_tokens, window_start, window_steps = [], 0, time.monotonic(), 0
        if step % cfg["steps_per_eval"] == 0 or step == cfg["iters"]:
            validate(step)

    adapters = Path(cfg["adapter_path"])
    adapters.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapters)
    log(f"Saved final weights to {adapters}.")


if __name__ == "__main__":
    main()
