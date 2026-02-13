# MaxRL vs GRPO Benchmark

Compares two advantage estimation methods (MaxRL and GRPO) on a Subset Sum task using Qwen models with LoRA fine-tuning. Based on the [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710) paper by Tajwar et al. (2026).

MaxRL normalizes advantages by `(r - mu) / mu`, which amplifies rare successes when mean reward is low. GRPO uses `(r - mu) / sigma`, which becomes unstable when variance is near zero. The paper shows MaxRL converges to maximum likelihood optimization and achieves up to 20x test-time scaling efficiency gains over GRPO.

## Files

- `max-rl.py` — Marimo notebook that runs a single training experiment. Works interactively or as a CLI script.
- `toy-simulation.py` — Marimo notebook comparing MaxRL vs GRPO on a toy binary reward simulation (no GPU needed).
- `grid.py` — Submits a grid of experiments to HuggingFace Jobs.

## Running a single job locally

```bash
# Interactive mode
uv run marimo edit max-rl.py

# Script mode with defaults
uv run max-rl.py

# Script mode with custom params
uv run max-rl.py -- --advantage-method grpo --group-size 16 --batch-size 4
```

## Running jobs on HuggingFace

### Prerequisites

- `hf` CLI installed (`pip install huggingface_hub[cli]`)
- Logged in via `hf login`
- A `.env` file in the parent directory with `WANDB_API_KEY=...`

### Submit a single job

```bash
hf jobs uv run \
  --detach \
  --flavor a10g-small \
  --env PYTHONUNBUFFERED=1 \
  --timeout 4h \
  --secrets-file ../.env \
  max-rl.py -- \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --advantage-method maxrl \
  --batch-size 4 \
  --group-size 16 \
  --epochs 20 \
  --wandb-run-name "Qwen2.5-0.5B-maxrl-n6-bs4-gs16"
```

Use `--flavor a100-large` for the 3B model.

### Submit the full grid

Edit the `grid` dict in `grid.py` to configure which combinations to run, then:

```bash
uv run grid.py
```

### Monitor jobs

```bash
hf jobs ps                    # List all jobs
hf jobs logs <job_id>         # View logs for a specific job
```

## Naming convention

Run names follow the pattern: `{model}-{method}-n{num_elements}-bs{batch_size}-gs{group_size}`

| Abbreviation | Parameter | Description |
|---|---|---|
| `n6` | `num_elements` | Number of integers in the subset sum problem |
| `bs4` | `batch_size` | Problems per training step |
| `gs16` | `group_size` | Rollouts per problem |

## Results

All runs log to the `maxrl-subset-sum` W&B project.
