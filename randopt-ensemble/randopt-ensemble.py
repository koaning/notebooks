# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
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
    import os
    import marimo as mo
    import torch
    import random
    import wandb
    from pathlib import Path
    from collections import Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pydantic import BaseModel, Field
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BaseModel,
        Counter,
        Field,
        mo,
        random,
        torch,
        wandb,
    )


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
            default="Qwen/Qwen2.5-0.5B-Instruct",
            description="HuggingFace model ID to use as base model.",
        )
        sigma: float = Field(
            default=1e-3,
            description="Noise scale for parameter perturbation.",
        )
        n_population: int = Field(
            default=10,
            description="Number of random seeds to search over.",
        )
        k_ensemble: int = Field(
            default=3,
            description="Number of top seeds to use in the ensemble.",
        )
        n_train_samples: int = Field(
            default=10,
            description="Number of training samples.",
        )
        n_test_samples: int = Field(
            default=10,
            description="Number of test samples.",
        )
        max_number: int = Field(
            default=100,
            description="Upper bound for random numbers in the pool.",
        )
        group_size: int = Field(
            default=4,
            description="How many numbers to pick for the target sum.",
        )
        wandb_project: str = Field(
            default="randopt-ensemble",
            description="Weights & Biases project name.",
        )
        wandb_run_name: str | None = Field(
            default=None,
            description="Optional Weights & Biases run name override.",
        )

    return (ModelParams,)


@app.cell
def _(mo):
    el = mo.md("""
    {model_id}

    {sigma}

    {n_population}

    {k_ensemble}

    {n_train_samples}

    {n_test_samples}

    {max_number}

    {group_size}

    {wandb_project}
    """).batch(
        model_id=mo.ui.dropdown(options=[
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-2-2b-it",
            "google/gemma-3-4b-it",
            "microsoft/Phi-4-mini-instruct",
        ], value="Qwen/Qwen2.5-3B-Instruct", label="Model ID"),
        sigma=mo.ui.slider(1e-4, 1e-1, value=1e-3, step=1e-4, label="Sigma (noise scale)"),
        n_population=mo.ui.slider(2, 50, value=10, step=1, label="Population size"),
        k_ensemble=mo.ui.slider(1, 10, value=3, step=1, label="Ensemble size (K)"),
        n_train_samples=mo.ui.slider(5, 100, value=15, step=5, label="Train samples"),
        n_test_samples=mo.ui.slider(5, 100, value=25, step=5, label="Test samples"),
        max_number=mo.ui.slider(10, 200, value=10, step=10, label="Max number"),
        group_size=mo.ui.slider(2, 8, value=3, step=1, label="Group size"),
        wandb_project=mo.ui.text(value="randopt-ensemble", label="W&B Project"),
    ).form()
    el
    return (el,)


@app.cell
def _(ModelParams, el, mo):
    if mo.app_meta().mode == "script":
        cli_args = mo.cli_args()
        if "help" in cli_args or len(cli_args) == 0:
            print("Usage: uv run randopt-ensemble.py --model-id <id> [options]")
            print()
            for name, field in ModelParams.model_fields.items():
                if field.is_required():
                    default = " (required)"
                elif field.default is None:
                    default = " (optional)"
                else:
                    default = f" (default: {field.default})"
                print(f"  --{name.replace('_', '-'):20s} {field.description}{default}")
            exit()
        model_params = ModelParams(
            **{k.replace("-", "_"): v for k, v in cli_args.items()}
        )
    else:
        mo.stop(el.value is None, mo.md("Submit the form to continue."))
        model_params = ModelParams(**el.value)

    if not model_params.wandb_run_name:
        model_short_name = model_params.model_id.split("/")[-1].lower()
        for suffix in ("-instruct", "-it"):
            if model_short_name.endswith(suffix):
                model_short_name = model_short_name.removesuffix(suffix)
        generated_run_name = (
            f"{model_short_name}"
            f"-s{model_params.sigma}"
            f"-p{model_params.n_population}"
            f"-k{model_params.k_ensemble}"
        )
        model_params = model_params.model_copy(
            update={"wandb_run_name": generated_run_name}
        )

    mo.md(f"""
    ### Active Parameters

    | Parameter | Value |
    |-----------|-------|
    | Model | `{model_params.model_id}` |
    | Sigma | `{model_params.sigma}` |
    | Population | `{model_params.n_population}` |
    | Ensemble K | `{model_params.k_ensemble}` |
    | Train samples | `{model_params.n_train_samples}` |
    | Test samples | `{model_params.n_test_samples}` |
    | Max number | `{model_params.max_number}` |
    | Group size | `{model_params.group_size}` |
    | W&B Project | `{model_params.wandb_project}` |
    | W&B Run | `{model_params.wandb_run_name}` |
    """)
    return (model_params,)


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
    def generate_4num_data(n_samples=10, max_number=100, group_size=4):
        data = []
        pool_size = 2 * group_size
        for _ in range(n_samples):
            pool = random.sample(range(1, max_number), pool_size)
            subset = random.sample(pool, group_size)
            target = sum(subset)
            answer_str = ", ".join(map(str, sorted(subset)))
            data.append({
                "nums": pool,
                "target": target,
                "answer": answer_str,
            })
        return data

    return (generate_4num_data,)


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
                **inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("commas.")[-1].strip()
        try:
            pred_nums = sorted([int(x.strip()) for x in response.replace(",", " ").split()])
            return ", ".join(map(str, pred_nums))
        except Exception:
            return "invalid_format"

    def evaluate_acc(model, dataset):
        correct = 0
        for s in dataset:
            if get_prediction(model, s) == s["answer"]:
                correct += 1
        return correct / len(dataset)

    def perturb_model(base_model, seed, sigma, direction=1.0):
        cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)

            scale = sigma * direction
            with torch.no_grad():
                for p in base_model.parameters():
                    p.add_(torch.randn_like(p) * scale)

    def evaluate_seed(base_model, dataset, seed, sigma):
        perturb_model(base_model, seed, sigma, direction=1.0)
        try:
            return evaluate_acc(base_model, dataset)
        finally:
            perturb_model(base_model, seed, sigma, direction=-1.0)
            if device == "cuda":
                torch.cuda.empty_cache()

    def predict_with_seed(base_model, dataset, seed, sigma):
        perturb_model(base_model, seed, sigma, direction=1.0)
        try:
            return [get_prediction(base_model, sample) for sample in dataset]
        finally:
            perturb_model(base_model, seed, sigma, direction=-1.0)
            if device == "cuda":
                torch.cuda.empty_cache()

    return evaluate_acc, evaluate_seed, predict_with_seed


@app.cell
def _(generate_4num_data, model_params):
    train_data = generate_4num_data(model_params.n_train_samples, model_params.max_number, model_params.group_size)
    test_data = generate_4num_data(model_params.n_test_samples, model_params.max_number, model_params.group_size)
    return test_data, train_data


@app.cell
def _(train_data):
    train_data
    return


@app.cell
def _(base_model, evaluate_seed, mo, model_params, train_data, wandb):
    wandb_config = model_params.model_dump()
    run = wandb.init(
        project=model_params.wandb_project,
        name=model_params.wandb_run_name,
        config=wandb_config,
    )
    wandb.config.update(wandb_config, allow_val_change=True)
    for _config_key, _config_value in wandb_config.items():
        wandb.summary[_config_key] = _config_value

    seed_results = []
    for _search_index, _seed in enumerate(range(model_params.n_population), start=1):
        print(
            f"[search {_search_index}/{model_params.n_population}] evaluating seed={_seed}",
            flush=True,
        )
        acc = evaluate_seed(base_model, train_data, _seed, model_params.sigma)
        seed_results.append({"seed": _seed, "train_acc": acc})
        wandb.log({"seed": _seed, "train_acc": acc})
        print(
            f"[search {_search_index}/{model_params.n_population}] seed={_seed} train_acc={acc:.1%}",
            flush=True,
        )

    seed_results_sorted = sorted(seed_results, key=lambda x: x["train_acc"], reverse=True)
    top_seeds = [r["seed"] for r in seed_results_sorted[: model_params.k_ensemble]]

    mo.md(f"""
    ### Parallel Search Complete

    Top seeds: `{top_seeds}`

    | Seed | Train Accuracy |
    |------|---------------|
    {"".join(f"| {r['seed']} | {r['train_acc']:.1%} |" + chr(10) for r in seed_results_sorted)}
    """)
    return (top_seeds,)


@app.cell
def _(
    Counter,
    base_model,
    evaluate_acc,
    mo,
    model_params,
    predict_with_seed,
    test_data,
    top_seeds,
    wandb,
):
    print("[base] evaluating base model on test set", flush=True)
    base_acc = evaluate_acc(base_model, test_data)
    print(f"[base] base_test_acc={base_acc:.1%}", flush=True)

    seed_predictions = {}
    for _ensemble_index, _seed in enumerate(top_seeds, start=1):
        print(
            f"[ensemble {_ensemble_index}/{len(top_seeds)}] generating predictions for seed={_seed}",
            flush=True,
        )
        seed_predictions[_seed] = predict_with_seed(
            base_model, test_data, _seed, model_params.sigma
        )

    votes_correct = 0
    test_predictions = []
    progress_interval = max(1, len(test_data) // 5)

    for idx, sample in enumerate(test_data):
        all_preds = [seed_predictions[_seed][idx] for _seed in top_seeds]

        final_answer = Counter(all_preds).most_common(1)[0][0]
        is_correct = final_answer == sample["answer"]
        if is_correct:
            votes_correct += 1

        test_predictions.append({
            "nums": str(sample["nums"]),
            "target": sample["target"],
            "answer": sample["answer"],
            "ensemble_prediction": final_answer,
            "correct": is_correct,
            "individual_predictions": str(all_preds),
        })

        if (idx + 1) % progress_interval == 0 or idx + 1 == len(test_data):
            print(
                f"[vote {idx + 1}/{len(test_data)}] running_ensemble_acc={votes_correct / (idx + 1):.1%}",
                flush=True,
            )

    ensemble_acc = votes_correct / len(test_data)
    uplift_abs = ensemble_acc - base_acc
    uplift_rel = (ensemble_acc / base_acc - 1.0) if base_acc else None
    uplift_rel_display = f"{uplift_rel:+.1%}" if uplift_rel is not None else "n/a"

    wandb.log({
        "base_test_acc": base_acc,
        "ensemble_test_acc": ensemble_acc,
        "uplift_abs": uplift_abs,
        "uplift_rel": uplift_rel,
    })
    wandb.summary["base_test_acc"] = base_acc
    wandb.summary["ensemble_test_acc"] = ensemble_acc
    wandb.summary["uplift_abs"] = uplift_abs
    if uplift_rel is not None:
        wandb.summary["uplift_rel"] = uplift_rel
    wandb.summary["top_seeds"] = str(top_seeds)

    mo.md(f"""
    ### Final Results (4-Number Task)

    | Metric | Value |
    |--------|-------|
    | Base Accuracy | {base_acc:.1%} |
    | Ensemble Accuracy | {ensemble_acc:.1%} |
    | Absolute Uplift | {uplift_abs:+.1%} |
    | Relative Uplift | {uplift_rel_display} |
    """)
    return


@app.cell
def _(mo, wandb):
    wandb.finish()
    mo.md("### Run complete. Data logged to W&B.")
    return


if __name__ == "__main__":
    app.run()
