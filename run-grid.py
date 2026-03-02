# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Grid search over models and repeat counts for ARC-Easy prompt repetition experiment."""

import subprocess

WANDB_BASE_URL = "https://api.inference.wandb.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

WANDB_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]

OPENROUTER_MODELS = [
    "google/gemma-3n-e4b-it",
    "qwen/qwen-2.5-7b-instruct",
]

REPEAT_COUNTS = [1, 2, 3]
PROMPT_ORDERS = ["question_first", "options_first"]
STRUCTURED_OUTPUT = [False, True]
N_EXAMPLES = 750
CONCURRENCY = 16


def main():
    runs = []
    for model in WANDB_MODELS:
        for repeat_count in REPEAT_COUNTS:
            for order in PROMPT_ORDERS:
                for structured in STRUCTURED_OUTPUT:
                    runs.append((WANDB_BASE_URL, model, repeat_count, order, structured))
    for model in OPENROUTER_MODELS:
        for repeat_count in REPEAT_COUNTS:
            for order in PROMPT_ORDERS:
                for structured in STRUCTURED_OUTPUT:
                    runs.append((OPENROUTER_BASE_URL, model, repeat_count, order, structured))

    for i, (base_url, model, repeat_count, order, structured) in enumerate(runs, 1):
        short_name = model.split("/")[-1]
        so_tag = "structured" if structured else "plain"
        order_tag = "qfirst" if order == "question_first" else "ofirst"
        run_name = f"{short_name}-repeat{repeat_count}-{order_tag}-{so_tag}"
        print(f"=== [{i}/{len(runs)}] {run_name} ===")
        cmd = [
            "uv", "run", "arc-easy-llm.py", "--",
            "--base-url", base_url,
            "--models", model,
            "--n-examples", str(N_EXAMPLES),
            "--concurrency", str(CONCURRENCY),
            "--repeat-count", str(repeat_count),
            "--prompt-order", order,
            "--wandb-run-name", run_name,
        ]
        if structured:
            cmd.extend(["--structured-output", "true"])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
        print()

    print("All runs complete.")


if __name__ == "__main__":
    main()
