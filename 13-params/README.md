# 13 Parameters Is All You Need

TinyLoRA training with GRPO on GSM8K math reasoning. Tracks experiments with WANDB.

## Running Jobs

### Local (Interactive)
```bash
cd 13-params/
uvx marimo edit 13-params-all-you-need.py
```

### Local (Script with custom params)
```bash
uv run 13-params-all-you-need.py -- \
  --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --learning-rate 5e-5 \
  --n-params 50 \
  --epochs 10
```

### HuggingFace Jobs

**Quick test (2 epochs, small dataset):**
```bash
uvx hf jobs uv run --flavor a10g-small --secrets-file .env \
  13-params-all-you-need.py -- \
    --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
    --epochs 2 \
    --train-samples 50 \
    --test-samples 20 \
    --wandb-run-name "test-run"
```

**With 7B model:**
```bash
uvx hf jobs uv run --flavor a10g-large --secrets-file .env \
  13-params-all-you-need.py -- \
    --model-id "Qwen/Qwen2.5-7B-Instruct" \
    --learning-rate 5e-5 \
    --n-params 50 \
    --epochs 10
```

**Grid search (submit multiple jobs):**
```bash
python run_grid.py
```
Edit the `grid` dict in `run_grid.py` to customize your hyperparameter sweep.

**Monitor jobs:**
```bash
uvx hf jobs ps              # List jobs
uvx hf jobs logs <job_id>   # View logs
uvx hf jobs cancel <job_id> # Cancel job
```

## Available Parameters

| Parameter | Default | Options |
|-----------|---------|---------|
| `--model-id` | Qwen/Qwen2.5-0.5B-Instruct | 0.5B or 7B |
| `--learning-rate` | 1e-5 | |
| `--n-params` | 13 | TinyLoRA parameter count |
| `--batch-size` | 8 | |
| `--group-size` | 8 | GRPO group size |
| `--epochs` | 20 | |
| `--temperature` | 0.9 | |
| `--train-samples` | 400 | |
| `--test-samples` | 100 | |
| `--wandb-run-name` | None | Custom name for run |

## GPU Requirements & Batch Sizing

**Memory Considerations:**
- The 0.5B model uses ~21GB in bfloat16, leaving limited room for batch processing
- The `model.generate()` step with `num_return_sequences=group_size` is memory-intensive
- Total sequences per batch = `batch_size × group_size`

**Recommended Configurations:**

| GPU | Memory | Model | batch_size | group_size | Sequences | Batches/Epoch (7473 samples) |
|-----|--------|-------|-----------|-----------|-----------|------------------------------|
| A10G | 24GB | 0.5B | 8 | 2 | 16 | 934 |
| A10G | 24GB | 0.5B | 4 | 4 | 16 | 1,868 |
| A100 | 80GB | 0.5B | 16-32 | 8 | 128-256 | 234-467 |
| A100 | 80GB | 7B | 8-16 | 4-8 | 32-128 | 467-934 |

**Memory Optimization:**
If you hit OOM errors, add the PyTorch memory defragmentation flag:
```bash
uvx hf jobs uv run --flavor a100-large -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --secrets-file .env 13-params-all-you-need.py -- \
    --batch-size 32 --group-size 8
```

## Training Time & Cost Estimates

**Full dataset (7,473 samples, 3 epochs):**

### 0.5B Model:

**A10G (batch_size=8, group_size=2):**
- 934 batches/epoch × 3 epochs = 2,802 batches
- ~5 sec/batch → **~4 hours**
- Cost: **$3.20-4.80** ($0.80-1.20/hr)

**A100 (batch_size=32, group_size=8):**
- 234 batches/epoch × 3 epochs = 702 batches
- ~10 sec/batch → **~2 hours**
- Cost: **$5.00** ($2.50/hr)

### 7B Model:

**A100 (batch_size=16, group_size=4):**
- 467 batches/epoch × 3 epochs = 1,401 batches
- ~25 sec/batch → **~10 hours**
- Cost: **$25.00** ($2.50/hr)

**Trade-offs:**
- **Cheapest**: A10G 0.5B (~$4, 4 hours)
- **Fastest**: A100 0.5B (~$5, 2 hours)
- **Best model**: A100 7B (~$25, 10 hours)

## Setup

Create a `.env` file with your WANDB API key:
```
WANDB_API_KEY=your_key_here
```

This works for both local runs and HF Jobs (passed via `--secrets-file .env`).
