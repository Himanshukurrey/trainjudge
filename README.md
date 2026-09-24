# TrainJudge

Before you fine-tune, TrainJudge tells you whether fine-tuning is even the right move.
After you fine-tune, it tells you whether it actually worked — on the task metric, not training loss.

> Status: pre-alpha (v0.1 in progress). Commands are stubs until their planned day lands.

## Install (dev)

```bash
pip install -e ".[dev,mlx]"   # mlx extra is Apple Silicon only
```

## Commands

```
trainjudge diagnose --dataset <path> --model <name> --goal "<text>"
trainjudge audit <path>
trainjudge train --dataset <path> --model <name> --method lora
trainjudge verify <run-dir>
```

## Known limitation

The diagnosis step is a heuristic classifier, not a guarantee. It can misclassify
mixed-goal tasks (partly knowledge, partly format).

## License

MIT
