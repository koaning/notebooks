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

# Easy set: low max_value, fewer elements
# Hard set: high max_value, more elements
# Both vary group size to test MaxRL's advantage with more rollouts
configs = [
    # Easy: n=5, max_value=20
    {"advantage_method": "maxrl", "num_elements": 5,  "max_value": 20, "batch_size": 4, "group_size": 16},
    {"advantage_method": "grpo",  "num_elements": 5,  "max_value": 20, "batch_size": 4, "group_size": 16},
    {"advantage_method": "maxrl", "num_elements": 5,  "max_value": 20, "batch_size": 4, "group_size": 32},
    {"advantage_method": "grpo",  "num_elements": 5,  "max_value": 20, "batch_size": 4, "group_size": 32},
    {"advantage_method": "maxrl", "num_elements": 5,  "max_value": 20, "batch_size": 4, "group_size": 64},
    {"advantage_method": "grpo",  "num_elements": 5,  "max_value": 20, "batch_size": 4, "group_size": 64},
    # Hard: n=10, max_value=50
    {"advantage_method": "maxrl", "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 16},
    {"advantage_method": "grpo",  "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 16},
    {"advantage_method": "maxrl", "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 32},
    {"advantage_method": "grpo",  "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 32},
    {"advantage_method": "maxrl", "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 64},
    {"advantage_method": "grpo",  "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 64},
    {"advantage_method": "maxrl", "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 128},
    {"advantage_method": "grpo",  "num_elements": 10, "max_value": 50, "batch_size": 4, "group_size": 128},
]

# Fixed params for all runs
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
wandb_project = "maxrl-subset-sum-v2"
fixed = {
    "epochs": 20,
    "learning_rate": 1e-5,
}


def submit_job(params):
    """Submit a single HF job with given params."""
    model_short = model_id.split("/")[-1].replace("-Instruct", "")
    run_name = (
        f"{model_short}-{params['advantage_method']}"
        f"-n{params['num_elements']}-mv{params['max_value']}"
        f"-bs{params['batch_size']}-gs{params['group_size']}"
    )

    flavor = "a100-large"

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
        "--max-value", str(params["max_value"]),
        "--batch-size", str(params["batch_size"]),
        "--group-size", str(params["group_size"]),
        "--epochs", str(fixed["epochs"]),
        "--wandb-project", wandb_project,
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
    print(f"W&B project: {wandb_project}")
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
