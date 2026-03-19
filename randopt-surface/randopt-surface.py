# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "torch>=2.0.0",
#     "transformers",
#     "pydantic",
#     "wandb",
#     "python-dotenv",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import random
    import torch
    import wandb
    from pathlib import Path
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return AutoModelForCausalLM, AutoTokenizer, BaseModel, Field, mo, np, random, torch, wandb


@app.cell(hide_code=True)
def _(wandb):
    from wigglystuff import EnvConfig

    env_config = EnvConfig({
        "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
    })

    env_config
    return (env_config,)


@app.cell
def _(BaseModel, Field, env_config, mo):
    if mo.app_meta().mode != "script":
        env_config.require_valid()

    class ModelParams(BaseModel):
        model_id: str = Field(
            default="Qwen/Qwen2.5-3B-Instruct",
            description="HuggingFace model ID to use as base model.",
        )
        sigma: float = Field(
            default=2e-3,
            description="Noise scale for parameter perturbation.",
        )
        min_seed: int = Field(
            default=0,
            description="Inclusive lower bound for evaluated seeds.",
        )
        max_seed: int = Field(
            default=10,
            description="Exclusive upper bound for evaluated seeds.",
        )
        n_train_samples: int = Field(
            default=100,
            description="Number of samples in the fixed evaluation set.",
        )
        max_number: int = Field(
            default=10,
            description="Upper bound for random numbers in the pool.",
        )
        group_size: int = Field(
            default=2,
            description="How many numbers to pick for the target sum.",
        )
        data_seed: int = Field(
            default=0,
            description="Seed used to generate the train/test datasets.",
        )
        wandb_project: str = Field(
            default="randopt-surface",
            description="Weights & Biases project name.",
        )
        wandb_group: str | None = Field(
            default=None,
            description="Optional W&B group used to group per-seed runs.",
        )
        wandb_run_name: str | None = Field(
            default=None,
            description="Optional prefix used for per-seed W&B run names.",
        )

    return (ModelParams,)


@app.cell
def _(mo):
    el = mo.md("""
    {model_id}

    {sigma}

    {min_seed}

    {max_seed}

    {n_train_samples}

    {max_number}

    {group_size}

    {data_seed}

    {wandb_project}

    {wandb_group}

    {wandb_run_name}
    """).batch(
        model_id=mo.ui.dropdown(
            options=[
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "google/gemma-2-2b-it",
                "google/gemma-3-4b-it",
                "microsoft/Phi-4-mini-instruct",
            ],
            value="Qwen/Qwen2.5-3B-Instruct",
            label="Model ID",
        ),
        sigma=mo.ui.number(value=2e-3, start=1e-4, stop=1e-1, step=1e-4, label="Sigma"),
        min_seed=mo.ui.number(value=0, start=0, stop=10_000, step=1, label="Min seed"),
        max_seed=mo.ui.number(value=10, start=1, stop=10_000, step=1, label="Max seed (exclusive)"),
        n_train_samples=mo.ui.slider(10, 500, value=100, step=10, label="Samples"),
        max_number=mo.ui.slider(10, 200, value=10, step=10, label="Max number"),
        group_size=mo.ui.slider(2, 8, value=2, step=1, label="Group size"),
        data_seed=mo.ui.number(value=0, start=0, stop=10_000, step=1, label="Data seed"),
        wandb_project=mo.ui.text(value="randopt-surface", label="W&B Project"),
        wandb_group=mo.ui.text(value="", label="W&B Group"),
        wandb_run_name=mo.ui.text(value="", label="W&B Run Prefix"),
    ).form()
    el
    return (el,)


@app.cell
def _(ModelParams, el, mo):
    if mo.app_meta().mode == "script":
        cli_args = mo.cli_args()
        if "help" in cli_args or len(cli_args) == 0:
            print("Usage: uv run randopt-surface.py --model-id <id> [options]")
            print()
            for name, field in ModelParams.model_fields.items():
                if field.is_required():
                    default = " (required)"
                elif field.default is None:
                    default = " (optional)"
                else:
                    default = f" (default: {field.default})"
                print(f"  --{name.replace('_', '-'):20s} {field.description}{default}")
            raise SystemExit(0)
        model_params = ModelParams(
            **{k.replace("-", "_"): v for k, v in cli_args.items()}
        )
    else:
        mo.stop(el.value is None, mo.md("Submit the form to continue."))
        raw_values = {
            k: (v if v != "" else None)
            for k, v in el.value.items()
        }
        model_params = ModelParams(**raw_values)

    if model_params.max_seed <= model_params.min_seed:
        raise ValueError("max_seed must be greater than min_seed.")

    model_short_name = model_params.model_id.split("/")[-1].lower()
    for suffix in ("-instruct", "-it"):
        if model_short_name.endswith(suffix):
            model_short_name = model_short_name.removesuffix(suffix)

    updates = {}
    if not model_params.wandb_group:
        updates["wandb_group"] = (
            f"{model_short_name}-s{model_params.sigma}-data{model_params.data_seed}"
        )
    if not model_params.wandb_run_name:
        updates["wandb_run_name"] = (
            f"{model_short_name}-s{model_params.sigma}"
            f"-seeds{model_params.min_seed}-{model_params.max_seed}"
            f"-data{model_params.data_seed}"
        )
    if updates:
        model_params = model_params.model_copy(update=updates)

    n_seeds = model_params.max_seed - model_params.min_seed
    mo.md(f"""
    ### Active Parameters

    | Parameter | Value |
    |-----------|-------|
    | Model | `{model_params.model_id}` |
    | Sigma | `{model_params.sigma}` |
    | Min seed | `{model_params.min_seed}` |
    | Max seed | `{model_params.max_seed}` |
    | Seed count | `{n_seeds}` |
    | Samples | `{model_params.n_train_samples}` |
    | Max number | `{model_params.max_number}` |
    | Group size | `{model_params.group_size}` |
    | Data seed | `{model_params.data_seed}` |
    | W&B Project | `{model_params.wandb_project}` |
    | W&B Group | `{model_params.wandb_group}` |
    | W&B Run Prefix | `{model_params.wandb_run_name}` |
    | RNG backend | `numpy_pcg64_cpu` |
    """)
    return model_params, n_seeds


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, model_params, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_params.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_params.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.eval()
    return base_model, device, tokenizer


@app.cell
def _(random):
    def generate_subset_sum_data(n_samples=10, max_number=100, group_size=4, seed=0):
        data = []
        rng = random.Random(seed)
        pool_size = 2 * group_size
        for _ in range(n_samples):
            pool = rng.sample(range(1, max_number), pool_size)
            subset = rng.sample(pool, group_size)
            target = sum(subset)
            answer_str = ", ".join(map(str, sorted(subset)))
            data.append({
                "nums": pool,
                "target": target,
                "answer": answer_str,
            })
        return data

    return (generate_subset_sum_data,)


@app.cell
def _(device, model_params, tokenizer, torch):
    def get_prediction(model, sample):
        prompt = (
            f"Numbers: {sample['nums']}. Find EXACTLY {model_params.group_size} that sum to {sample['target']}. "
            f"Answer only with the {model_params.group_size} numbers separated by commas."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("commas.")[-1].strip()
        try:
            pred_nums = sorted([int(x.strip()) for x in response.replace(",", " ").split()])
            return ", ".join(map(str, pred_nums))
        except ValueError:
            return "invalid_format"

    def evaluate_acc(model, dataset):
        correct = 0
        for sample in dataset:
            if get_prediction(model, sample) == sample["answer"]:
                correct += 1
        return correct / len(dataset)

    return evaluate_acc, get_prediction


@app.cell
def _(device, np, torch):
    def perturb_model(base_model, seed, sigma, direction=1.0):
        rng = np.random.Generator(np.random.PCG64(seed))
        scale = sigma * direction
        with torch.no_grad():
            for param in base_model.parameters():
                noise = rng.standard_normal(param.shape, dtype=np.float32)
                noise_tensor = torch.from_numpy(noise).to(device=param.device, dtype=param.dtype)
                param.add_(noise_tensor, alpha=scale)

        if device == "cuda":
            torch.cuda.synchronize()

    return (perturb_model,)


@app.cell
def _(device, evaluate_acc, perturb_model, torch):
    def evaluate_seed(base_model, dataset, seed, sigma):
        perturb_model(base_model, seed, sigma, direction=1.0)
        try:
            return evaluate_acc(base_model, dataset)
        finally:
            perturb_model(base_model, seed, sigma, direction=-1.0)
            if device == "cuda":
                torch.cuda.empty_cache()

    return (evaluate_seed,)


@app.cell
def _(generate_subset_sum_data, model_params):
    eval_data = generate_subset_sum_data(
        model_params.n_train_samples,
        model_params.max_number,
        model_params.group_size,
        model_params.data_seed,
    )
    return (eval_data,)


@app.cell
def _(base_model, eval_data, evaluate_acc):
    base_acc = evaluate_acc(base_model, eval_data)
    return (base_acc,)


@app.cell
def _(
    base_model,
    base_acc,
    eval_data,
    evaluate_seed,
    model_params,
    n_seeds,
    wandb,
):
    results = []
    common_config = model_params.model_dump()
    common_config["perturbation_rng"] = "numpy_pcg64_cpu"
    common_config["seed_count"] = n_seeds

    for index, seed in enumerate(range(model_params.min_seed, model_params.max_seed), start=1):
        print(f"[seed {index}/{n_seeds}] evaluating seed={seed}", flush=True)
        seed_acc = evaluate_seed(base_model, eval_data, seed, model_params.sigma)
        uplift_abs = seed_acc - base_acc
        uplift_rel = (seed_acc / base_acc - 1.0) if base_acc else None
        run_name = f"{model_params.wandb_run_name}-seed{seed}"

        run_config = dict(common_config)
        run_config["seed"] = seed

        with wandb.init(
            project=model_params.wandb_project,
            group=model_params.wandb_group,
            name=run_name,
            config=run_config,
        ) as run:
            metrics = {
                "base_acc": base_acc,
                "seed_acc": seed_acc,
                "uplift_abs": uplift_abs,
                "seed": seed,
            }
            if uplift_rel is not None:
                metrics["uplift_rel"] = uplift_rel

            run.log(metrics)
            for key, value in run_config.items():
                run.summary[key] = value
            run.summary["base_acc"] = base_acc
            run.summary["seed_acc"] = seed_acc
            run.summary["uplift_abs"] = uplift_abs
            if uplift_rel is not None:
                run.summary["uplift_rel"] = uplift_rel

        results.append({
            "seed": seed,
            "seed_acc": seed_acc,
            "uplift_abs": uplift_abs,
            "uplift_rel": uplift_rel,
        })
        print(
            f"[seed {index}/{n_seeds}] seed={seed} "
            f"acc={seed_acc:.1%} uplift={uplift_abs:+.1%}",
            flush=True,
        )

    return common_config, results


@app.cell
def _(base_acc, mo, results):
    results_sorted = sorted(results, key=lambda row: row["seed_acc"], reverse=True)
    preview_rows = results_sorted[:10]
    row_lines = []
    for row in preview_rows:
        uplift_rel_display = f"{row['uplift_rel']:+.1%}" if row["uplift_rel"] is not None else "n/a"
        row_lines.append(
            f"| {row['seed']} | {row['seed_acc']:.1%} | "
            f"{row['uplift_abs']:+.1%} | {uplift_rel_display} |"
        )
    table_rows = "\n".join(row_lines)

    mo.md(f"""
    ### Seed Sweep Complete

    | Metric | Value |
    |--------|-------|
    | Base accuracy | {base_acc:.1%} |
    | Evaluated seeds | {len(results)} |

    Top seeds by accuracy:

    | Seed | Accuracy | Absolute Uplift | Relative Uplift |
    |------|----------|-----------------|-----------------|
    {table_rows}
    """)
    return (results_sorted,)


if __name__ == "__main__":
    app.run()
