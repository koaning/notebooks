# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=0.30",
#     "python-dotenv",
# ]
# ///

"""
Seed-range launcher for randopt-surface.py on Hugging Face Jobs.

Usage:
    uv run randopt-surface/run_grid.py
    uv run randopt-surface/run_grid.py --launch
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import run_uv_job

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
NOTEBOOK_PATH = PROJECT_DIR / "randopt-surface.py"

load_dotenv(ROOT_DIR / ".env")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SIGMA = 2e-3
FLAVOR = "l4x1"

FIXED = {
    "n_train_samples": "100",
    "max_number": "10",
    "group_size": "2",
    "data_seed": "0",
    "wandb_project": "randopt-surface",
}

SEEDS_PER_JOB = 25
N_JOBS = 10
SEED_RANGES = [
    (job_index * SEEDS_PER_JOB, (job_index + 1) * SEEDS_PER_JOB)
    for job_index in range(N_JOBS)
][1:]

TIMEOUT = "2h"


def build_runs(fixed):
    runs = []
    model_short_name = MODEL_ID.split("/")[-1].lower()
    for suffix in ("-instruct", "-it"):
        if model_short_name.endswith(suffix):
            model_short_name = model_short_name.removesuffix(suffix)

    wandb_group = f"{model_short_name}-s{SIGMA}-data{fixed['data_seed']}"
    for min_seed, max_seed in SEED_RANGES:
        params = {
            "model_id": MODEL_ID,
            "sigma": str(SIGMA),
            "min_seed": str(min_seed),
            "max_seed": str(max_seed),
            "wandb_group": wandb_group,
        }
        params.update(fixed)
        params["wandb_run_name"] = (
            f"{model_short_name}-s{SIGMA}-seeds{min_seed}-{max_seed}"
            f"-data{fixed['data_seed']}"
        )
        runs.append(params)
    return runs


def params_to_cli_args(params):
    args = []
    for key, value in params.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually launch the jobs (default is dry run)")
    args = parser.parse_args()

    secrets = {
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
    }

    runs = build_runs(FIXED)
    print(
        f"Grid has {len(runs)} runs "
        f"(seed ranges={SEED_RANGES}, flavor={FLAVOR}, timeout={TIMEOUT})\n"
    )

    for index, params in enumerate(runs, start=1):
        cli_args = params_to_cli_args(params)
        print(f"[{index}/{len(runs)}] {params['wandb_run_name']}")
        print(f"  flavor: {FLAVOR}")
        print(f"  args: {' '.join(cli_args)}")

        if args.launch:
            job = run_uv_job(
                str(NOTEBOOK_PATH),
                script_args=cli_args,
                flavor=FLAVOR,
                secrets=secrets,
                timeout=TIMEOUT,
            )
            print(f"  launched: {job.url}")
        else:
            print("  (dry run)")
        print()

    if not args.launch:
        print("Dry run complete. Pass --launch to submit jobs.")


if __name__ == "__main__":
    main()
