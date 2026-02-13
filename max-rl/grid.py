# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "huggingface_hub",
# ]
# ///

"""
Submit a grid of HF Jobs for MaxRL vs GRPO benchmark.
Usage: uv run grid.py
"""
import subprocess
import itertools

# Define your grid
grid = {
    "model_id": [
        "Qwen/Qwen2.5-0.5B-Instruct",
    ],
    "advantage_method": ["maxrl", "grpo"],
    "learning_rate": [1e-5],
    "num_elements": [6],
    "batch_size": [4],
    "group_size": [8, 16],
}

# Fixed params for all runs
fixed = {
    "epochs": 20,
    "max_value": 20,
}


def submit_job(params):
    """Submit a single HF job with given params."""
    model_short = params["model_id"].split("/")[-1].replace("-Instruct", "")
    run_name = f"{model_short}-{params['advantage_method']}-n{params['num_elements']}-bs{params['batch_size']}-gs{params['group_size']}"

    # Choose flavor based on model size
    if "3B" in params["model_id"]:
        flavor = "a100-large"
    else:
        flavor = "a10g-small"

    cmd = [
        "hf", "jobs", "uv", "run",
        "--detach",
        "--flavor", flavor,
        "--env", "PYTHONUNBUFFERED=1",
        "--timeout", "4h",
        "--secrets-file", "../.env",
        "max-rl.py", "--",
        "--model-id", params["model_id"],
        "--advantage-method", params["advantage_method"],
        "--learning-rate", str(params["learning_rate"]),
        "--num-elements", str(params["num_elements"]),
        "--batch-size", str(params["batch_size"]),
        "--group-size", str(params["group_size"]),
        "--epochs", str(fixed["epochs"]),
        "--max-value", str(fixed["max_value"]),
        "--wandb-run-name", run_name,
    ]

    print(f"Submitting: {run_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  Job submitted: {result.stdout.strip()}")
        return result.stdout.strip()
    else:
        print(f"  Failed: {result.stderr.strip()}")
        return None


def main():
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
