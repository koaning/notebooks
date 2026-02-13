# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.8",
#     "matplotlib",
#     "numpy",
#     "polars",
#     "pydantic>=2.0.0",
#     "accelerate",
#     "torch==2.10.0",
#     "transformers==5.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import matplotlib.pyplot as plt
    import numpy as np
    import random


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    from pydantic import BaseModel, Field

    class EvalParams(BaseModel):
        model_id: str = Field(
            default="Qwen/Qwen2.5-0.5B-Instruct",
            description="Model to evaluate",
        )
        num_elements_min: int = Field(default=4, description="Min num_elements to sweep")
        num_elements_max: int = Field(default=10, description="Max num_elements to sweep")
        max_values: str = Field(
            default="10,20,50",
            description="Comma-separated max_value list to sweep",
        )
        n_problems: int = Field(default=50, description="Problems per parameter combination")
        max_new_tokens: int = Field(default=64, description="Max generation tokens")
        seed: int = Field(default=42, description="Random seed for reproducibility")

    return (EvalParams,)


@app.cell
def _(EvalParams, is_script_mode, mo):
    cli_args = {k.replace("-", "_"): v for k, v in mo.cli_args().items()}
    # mo.cli_args() auto-parses "10" as int, but max_values should stay a string
    if "max_values" in cli_args:
        cli_args["max_values"] = str(cli_args["max_values"])
    eval_params = EvalParams(**cli_args)

    if is_script_mode:
        print("=" * 60)
        print("Subset Sum Baseline Evaluation")
        print("=" * 60)
        print(f"\nConfiguration:")
        for key, value in eval_params.model_dump().items():
            print(f"  {key}: {value}")
        print()
    return (eval_params,)


@app.cell
def _(eval_params, mo):
    available_models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]

    model_multiselect = mo.ui.multiselect(
        options=available_models,
        value=[eval_params.model_id],
        label="Models to evaluate",
    )

    num_elements_range_slider = mo.ui.range_slider(
        start=3, stop=12, step=1,
        value=[eval_params.num_elements_min, eval_params.num_elements_max],
        label="num_elements range",
    )

    max_values_input = mo.ui.text(
        value=eval_params.max_values,
        label="max_value list (comma-separated)",
    )

    n_problems_slider = mo.ui.slider(
        start=1, stop=200, step=1,
        value=eval_params.n_problems,
        label="Problems per setting",
    )

    mo.vstack([
        mo.md("## Evaluation Configuration"),
        model_multiselect,
        num_elements_range_slider,
        max_values_input,
        n_problems_slider,
    ])
    return (
        max_values_input,
        model_multiselect,
        n_problems_slider,
        num_elements_range_slider,
    )


@app.cell
def _(
    eval_params,
    is_script_mode,
    max_values_input,
    model_multiselect,
    n_problems_slider,
    num_elements_range_slider,
):
    if is_script_mode:
        model_ids = [eval_params.model_id]
        num_elements_list = list(range(
            eval_params.num_elements_min,
            eval_params.num_elements_max + 1,
        ))
        max_value_list = [int(x.strip()) for x in eval_params.max_values.split(",")]
        n_problems = eval_params.n_problems
        seed = eval_params.seed
    else:
        model_ids = model_multiselect.value
        lo, hi = num_elements_range_slider.value
        num_elements_list = list(range(lo, hi + 1))
        max_value_list = [int(x.strip()) for x in max_values_input.value.split(",")]
        n_problems = n_problems_slider.value
        seed = eval_params.seed
    return max_value_list, model_ids, n_problems, num_elements_list, seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Subset Sum: Base Model Difficulty Landscape

    How hard is the subset sum task for untrained LLMs? This notebook sweeps across
    `num_elements` (how many integers in the set) and `max_value` (upper bound of each integer)
    to measure greedy accuracy of base Qwen models.

    Harder problems have more elements and larger values. Results help choose
    good training parameters for the MaxRL/GRPO benchmark.
    """)
    return


@app.cell
def _():
    def generate_subset_sum_problem(num_elements=6, max_value=20):
        numbers = [random.randint(1, max_value) for _ in range(num_elements)]
        # Pick a random subset to guarantee at least one solution
        k = random.randint(1, max(1, num_elements // 2))
        subset_indices = random.sample(range(num_elements), k)
        target = sum(numbers[i] for i in subset_indices)
        return numbers, target

    def format_prompt(numbers, target):
        nums_str = ", ".join(str(n) for n in numbers)
        return (
            f"Given the numbers [{nums_str}] and target {target}, "
            f"select a subset that sums exactly to {target}. "
            f"Reply with ONLY the selected numbers separated by commas."
        )

    def check_answer(response_text, numbers, target):
        import re
        first_line = response_text.strip().split("\n")[0].strip()
        selected = [int(m) for m in re.findall(r'\b\d+\b', first_line)]
        if not selected:
            return 0.0
        remaining = list(numbers)
        for s in selected:
            if s in remaining:
                remaining.remove(s)
            else:
                return 0.0
        return 1.0 if sum(selected) == target else 0.0

    return check_answer, format_prompt, generate_subset_sum_problem


@app.cell
def _():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    return device, dtype


@app.cell
def _(check_answer, format_prompt, generate_subset_sum_problem):
    def evaluate_single_model(
        model_id, device, dtype, num_elements_list, max_value_list, n_problems, max_new_tokens, seed
    ):
        """Load a model, sweep all (num_elements, max_value) combos, return results list."""
        import gc

        print(f"\nLoading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map={"": device},
        )
        model.eval()

        model_short = model_id.split("/")[-1]
        results = []
        total_combos = len(num_elements_list) * len(max_value_list)

        _idx = 0
        for _n_elem in num_elements_list:
            for _max_val in max_value_list:
                random.seed(seed)
                _correct = 0

                for _i in range(n_problems):
                    _nums, _tgt = generate_subset_sum_problem(_n_elem, _max_val)
                    _prompt = format_prompt(_nums, _tgt)
                    _messages = [{"role": "user", "content": _prompt}]
                    _formatted = tokenizer.apply_chat_template(
                        _messages, tokenize=False, add_generation_prompt=True
                    )
                    _inputs = tokenizer(_formatted, return_tensors="pt").to(device)
                    with torch.no_grad():
                        _out = model.generate(
                            **_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    _text = tokenizer.decode(
                        _out[0, _inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    _correct += check_answer(_text, _nums, _tgt)

                _accuracy = _correct / n_problems
                results.append({
                    "model": model_short,
                    "num_elements": _n_elem,
                    "max_value": _max_val,
                    "accuracy": _accuracy,
                    "n_problems": n_problems,
                })
                _idx += 1
                print(
                    f"  [{_idx}/{total_combos}] n={_n_elem}, max_val={_max_val}: "
                    f"{_accuracy:.1%}"
                )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return results

    return (evaluate_single_model,)


@app.cell
def _(
    device,
    dtype,
    eval_params,
    evaluate_single_model,
    is_script_mode,
    max_value_list,
    model_ids,
    n_problems,
    num_elements_list,
    seed,
):
    all_results = []

    for _model_id in model_ids:
        _model_results = evaluate_single_model(
            _model_id,
            device,
            dtype,
            num_elements_list,
            max_value_list,
            n_problems,
            eval_params.max_new_tokens,
            seed,
        )
        all_results.extend(_model_results)

    if is_script_mode:
        print(f"\nEvaluation complete. {len(all_results)} data points collected.")
    return (all_results,)


@app.cell
def _(all_results):
    import polars as pl

    results_df = pl.DataFrame(all_results)
    results_df
    return


if __name__ == "__main__":
    app.run()
