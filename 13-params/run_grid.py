#!/usr/bin/env python3
"""
Submit a grid of HF Jobs for hyperparameter search.
Usage: python run_grid.py
"""
import subprocess
import itertools

# Define your grid
grid = {
    "model_id": ["Qwen/Qwen2.5-0.5B-Instruct"],
    "learning_rate": [1e-4],
    "n_params": [1, 5, 13],
    "batch_size": [8],
    "group_size": [2],
    "train_samples": [500, 1000],
    "test_samples": [500],
}

# Fixed params for all runs
fixed = {
    "epochs": 20,
}

def submit_job(params):
    """Submit a single HF job with given params."""
    # Build run name
    model_short = params["model_id"].split("/")[-1].replace("-Instruct", "")
    run_name = f"{model_short}-p{params['n_params']}-bs{params['batch_size']}-gs{params['group_size']}-ts{params['train_samples']}"

    # Choose flavor based on model size
    flavor = "a100-large"

    # Build command
    cmd = [
        "hf", "jobs", "uv", "run",
        "--flavor", flavor,
        "--env", "PYTHONUNBUFFERED=1",
        "--timeout", "2h",
        "--secrets-file", ".env",
        "13-params-all-you-need.py", "--",
        "--model-id", params["model_id"],
        "--learning-rate", str(params["learning_rate"]),
        "--n-params", str(params["n_params"]),
        "--batch-size", str(params["batch_size"]),
        "--group-size", str(params["group_size"]),
        "--epochs", str(fixed["epochs"]),
        "--train-samples", str(params["train_samples"]),
        "--test-samples", str(params["test_samples"]),
        "--wandb-run-name", run_name,
    ]

    print(f"Submitting: {run_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Job submitted: {result.stdout.strip()}")
        return result.stdout.strip()
    else:
        print(f"  ✗ Failed: {result.stderr.strip()}")
        return None

def main():
    # Generate all combinations
    keys = grid.keys()
    values = grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Submitting {len(combinations)} jobs...")
    print(f"Grid: {grid}")
    print(f"Fixed: {fixed}\n")

    job_ids = []
    for i, params in enumerate(combinations, 1):
        print(f"[{i}/{len(combinations)}]", end=" ")
        job_id = submit_job(params)
        if job_id:
            job_ids.append(job_id)
        print()

    print(f"\nSubmitted {len(job_ids)}/{len(combinations)} jobs successfully!")
    print("\nMonitor with:")
    print("  hf jobs ps")
    print("  hf jobs logs <job_id>")

if __name__ == "__main__":
    main()
