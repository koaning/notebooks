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

# T-shaped grid: vary difficulty (vertical) and group size (horizontal)
# All jobs use Qwen2.5-0.5B-Instruct, lr=1e-5
configs = [
    # Vertical arm: vary difficulty at gs=16
    {"advantage_method": "maxrl", "num_elements": 8,  "batch_size": 4, "group_size": 16},
    {"advantage_method": "grpo",  "num_elements": 8,  "batch_size": 4, "group_size": 16},
    {"advantage_method": "maxrl", "num_elements": 10, "batch_size": 4, "group_size": 16},
    {"advantage_method": "grpo",  "num_elements": 10, "batch_size": 4, "group_size": 16},
    # Horizontal arm: vary group size at n=10
    {"advantage_method": "maxrl", "num_elements": 10, "batch_size": 4, "group_size": 32},
    {"advantage_method": "grpo",  "num_elements": 10, "batch_size": 4, "group_size": 32},
    {"advantage_method": "maxrl", "num_elements": 10, "batch_size": 4, "group_size": 64},
    {"advantage_method": "grpo",  "num_elements": 10, "batch_size": 4, "group_size": 64},
    # n=12 at gs=32 (compensate for extreme sparsity)
    {"advantage_method": "maxrl", "num_elements": 12, "batch_size": 4, "group_size": 32},
    {"advantage_method": "grpo",  "num_elements": 12, "batch_size": 4, "group_size": 32},
]

# Fixed params for all runs
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
fixed = {
    "epochs": 20,
    "max_value": 20,
    "learning_rate": 1e-5,
}


def submit_job(params):
    """Submit a single HF job with given params."""
    model_short = model_id.split("/")[-1].replace("-Instruct", "")
    run_name = f"{model_short}-{params['advantage_method']}-n{params['num_elements']}-bs{params['batch_size']}-gs{params['group_size']}"

    # Use A100 for large group sizes or 3B models
    if "3B" in model_id or params["group_size"] >= 32:
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
        "--model-id", model_id,
        "--advantage-method", params["advantage_method"],
        "--learning-rate", str(fixed["learning_rate"]),
        "--num-elements", str(params["num_elements"]),
        "--batch-size", str(params["batch_size"]),
        "--group-size", str(params["group_size"]),
        "--epochs", str(fixed["epochs"]),
        "--max-value", str(fixed["max_value"]),
        "--wandb-run-name", run_name,
    ]

    print(f"Submitting: {run_name} ({flavor})")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  Job submitted: {result.stdout.strip()}")
        return result.stdout.strip()
    else:
        print(f"  Failed: {result.stderr.strip()}")
        return None


def main():
    print(f"Submitting {len(configs)} jobs...")
    print(f"Model: {model_id}")
    print(f"Fixed: {fixed}\n")

    job_ids = []
    for i, params in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", end=" ")
        job_id = submit_job(params)
        if job_id:
            job_ids.append(job_id)
        print()

    print(f"\nSubmitted {len(job_ids)}/{len(configs)} jobs successfully!")
    print("\nMonitor with:")
    print("  hf jobs ps")
    print("  hf jobs logs <job_id>")


if __name__ == "__main__":
    main()
