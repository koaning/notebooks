# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=0.30",
#     "python-dotenv",
# ]
# ///

"""
Grid search launcher for randopt-ensemble.py on Hugging Face Jobs.

Usage:
    # Dry run (print what would be launched)
    uv run randopt-ensemble/run_grid.py

    # Actually launch
    uv run randopt-ensemble/run_grid.py --launch
"""

import itertools
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import run_uv_job

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
NOTEBOOK_PATH = PROJECT_DIR / "randopt-ensemble.py"

load_dotenv(ROOT_DIR / ".env")

# --- Grid definition ---
GRID = {
    "model_id": [
        "Qwen/Qwen2.5-3B-Instruct",
    ],
    "sigma": [1e-3, 2 * 1e-3, 2 * 1e-4],
    "n_population": [100],
    "k_ensemble": [3, 4],
}

# --- Fixed params ---
FIXED = {
    "n_train_samples": "20",
    "n_test_samples": "20",
    "max_number": "10",
    "group_size": "2",
    "wandb_project": "randopt-ensemble",
}

SMALL_MODEL_FLAVOR = "t4-small"
LARGE_MODEL_FLAVOR = "l4x1"
TIMEOUT = "1h"


def pick_flavor(model_id):
    model_name = model_id.lower()
    if any(tag in model_name for tag in ("7b", "8b", "9b")):
        return LARGE_MODEL_FLAVOR
    return SMALL_MODEL_FLAVOR


def build_runs(grid, fixed):
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    runs = []
    for combo in combos:
        params = {k: str(v) for k, v in zip(keys, combo)}
        params.update(fixed)
        model_short_name = params["model_id"].split("/")[-1].lower()
        for suffix in ("-instruct", "-it"):
            if model_short_name.endswith(suffix):
                model_short_name = model_short_name.removesuffix(suffix)
        params["wandb_run_name"] = (
            f"{model_short_name}"
            f"-s{params['sigma']}"
            f"-p{params['n_population']}"
            f"-k{params['k_ensemble']}"
        )
        runs.append(params)
    return runs


def params_to_cli_args(params):
    args = []
    for k, v in params.items():
        args.extend([f"--{k.replace('_', '-')}", str(v)])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="Actually launch the jobs (default is dry run)")
    args = parser.parse_args()

    secrets = {
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
    }

    runs = build_runs(GRID, FIXED)
    print(
        f"Grid has {len(runs)} runs "
        f"(default flavor={SMALL_MODEL_FLAVOR}, large-model flavor={LARGE_MODEL_FLAVOR}, timeout={TIMEOUT})\n"
    )

    for i, params in enumerate(runs):
        flavor = pick_flavor(params["model_id"])
        cli_args = params_to_cli_args(params)
        print(f"[{i+1}/{len(runs)}] {params['wandb_run_name']}")
        print(f"  flavor: {flavor}")
        print(f"  args: {' '.join(cli_args)}")

        if args.launch:
            job = run_uv_job(
                str(NOTEBOOK_PATH),
                script_args=cli_args,
                flavor=flavor,
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
