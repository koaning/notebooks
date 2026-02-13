# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.8",
#     "matplotlib",
#     "numpy",
#     "peft",
#     "polars",
#     "pydantic>=2.0.0",
#     "python-dotenv",
#     "torch==2.10.0",
#     "transformers==5.1.0",
#     "wandb",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


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

    class ExperimentParams(BaseModel):
        # Model
        model_id: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
        advantage_method: str = Field(default="maxrl", description="'maxrl' or 'grpo'")

        # LoRA
        lora_r: int = Field(default=8, description="LoRA rank")
        lora_alpha: int = Field(default=16, description="LoRA alpha")

        # Subset Sum problem
        num_elements: int = Field(default=6, description="Number of integers in the set")
        max_value: int = Field(default=20, description="Max value for each integer")

        # Training
        batch_size: int = Field(default=4, description="Problems per training step")
        group_size: int = Field(default=16, description="Rollouts per problem (G)")
        epochs: int = Field(default=20, description="Number of training epochs")
        learning_rate: float = Field(default=1e-5, description="Learning rate")
        max_new_tokens: int = Field(default=64, description="Max generation tokens")
        temperature: float = Field(default=0.8, description="Sampling temperature")

        # W&B
        wandb_project: str = Field(default="maxrl-subset-sum")
        wandb_run_name: str | None = Field(default=None)

    return (ExperimentParams,)


@app.cell
def _(ExperimentParams, is_script_mode, mo):
    import time

    start_time = time.time()

    cli_args = {k.replace("-", "_"): v for k, v in mo.cli_args().items()}
    model_params = ExperimentParams(**cli_args)

    if is_script_mode:
        print("=" * 60)
        print("MaxRL vs GRPO Benchmark: Subset Sum")
        print("=" * 60)
        print(f"\nConfiguration:")
        for key, value in model_params.model_dump().items():
            print(f"  {key}: {value}")
        print()
    return model_params, start_time, time


@app.cell
def _(is_script_mode, mo, model_params):
    available_models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]

    model_dropdown = mo.ui.dropdown(
        options=available_models,
        value=model_params.model_id,
        label="Model",
    )

    advantage_dropdown = mo.ui.dropdown(
        options=["maxrl", "grpo"],
        value=model_params.advantage_method,
        label="Advantage Method",
    )

    batch_size_slider = mo.ui.slider(
        value=model_params.batch_size,
        start=1,
        stop=32,
        step=1,
        label="Batch Size",
    )

    group_size_slider = mo.ui.slider(
        value=model_params.group_size,
        start=2,
        stop=32,
        step=2,
        label="Group Size",
    )

    epochs_slider = mo.ui.slider(
        value=model_params.epochs,
        start=1,
        stop=50,
        step=1,
        label="Epochs",
    )

    learning_rate_input = mo.ui.number(
        value=model_params.learning_rate,
        start=1e-6,
        stop=1e-3,
        step=1e-6,
        label="Learning Rate",
    )

    num_elements_slider = mo.ui.slider(
        value=model_params.num_elements,
        start=4,
        stop=12,
        step=1,
        label="Number of Elements (N)",
    )

    temperature_slider = mo.ui.slider(
        value=model_params.temperature,
        start=0.1,
        stop=2.0,
        step=0.1,
        label="Sampling Temperature",
    )

    lora_r_slider = mo.ui.slider(
        value=model_params.lora_r,
        start=1,
        stop=32,
        step=1,
        label="LoRA Rank",
    )

    _ui_output = None
    if not is_script_mode:
        _ui_output = mo.vstack([
            mo.md("## Training Configuration"),
            model_dropdown,
            advantage_dropdown,
            batch_size_slider,
            group_size_slider,
            epochs_slider,
            learning_rate_input,
            num_elements_slider,
            temperature_slider,
            lora_r_slider,
        ])

    _ui_output
    return (
        advantage_dropdown,
        batch_size_slider,
        epochs_slider,
        group_size_slider,
        learning_rate_input,
        lora_r_slider,
        model_dropdown,
        num_elements_slider,
        temperature_slider,
    )


@app.cell
def _(
    ExperimentParams,
    advantage_dropdown,
    batch_size_slider,
    epochs_slider,
    group_size_slider,
    is_script_mode,
    learning_rate_input,
    lora_r_slider,
    model_dropdown,
    model_params,
    num_elements_slider,
    temperature_slider,
):
    if is_script_mode:
        config = model_params
    else:
        config = ExperimentParams(
            model_id=model_dropdown.value,
            advantage_method=advantage_dropdown.value,
            lora_r=lora_r_slider.value,
            lora_alpha=model_params.lora_alpha,
            num_elements=num_elements_slider.value,
            max_value=model_params.max_value,
            batch_size=batch_size_slider.value,
            group_size=group_size_slider.value,
            epochs=epochs_slider.value,
            learning_rate=learning_rate_input.value,
            max_new_tokens=model_params.max_new_tokens,
            temperature=temperature_slider.value,
            wandb_project=model_params.wandb_project,
            wandb_run_name=model_params.wandb_run_name,
        )

    if not config.wandb_run_name:
        model_short = config.model_id.split("/")[-1].replace("-Instruct", "")
        config = ExperimentParams(
            **{**config.model_dump(),
               "wandb_run_name": f"{model_short}-{config.advantage_method}-n{config.num_elements}-bs{config.batch_size}-gs{config.group_size}"}
        )
    return (config,)


@app.cell
def _():
    from pathlib import Path
    from dotenv import load_dotenv

    if Path(".env").exists():
        load_dotenv(".env")
    return


@app.cell
def _():
    import wandb
    from wigglystuff import EnvConfig

    env_config = EnvConfig({
        "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
    })
    env_config
    return (env_config, wandb)


@app.cell
def _(config, env_config, is_script_mode, wandb):
    env_config.require_valid()

    if is_script_mode:
        print("Initializing WANDB...")

    import os

    import torch as _torch

    wandb_config = config.model_dump()
    # Try common env vars for HF Job ID
    hf_job_id = os.environ.get("HF_JOB_ID") or os.environ.get("HOSTNAME")
    if hf_job_id:
        wandb_config["hf_job_id"] = hf_job_id
    if _torch.cuda.is_available():
        wandb_config["gpu"] = _torch.cuda.get_device_name(0)

    wandb_run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=wandb_config,
        tags=[config.advantage_method, config.model_id.split("/")[-1]],
    )

    if is_script_mode:
        print(f"WANDB initialized: {wandb_run.url}\n")
    return (wandb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MaxRL vs GRPO: Subset Sum Benchmark

    This notebook benchmarks two advantage computation methods for reinforcement learning on a **Subset Sum** task,
    based on the [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710) paper
    by Tajwar, Zeng, Zhou, Song, Arora, Jiang, Schneider, Salakhutdinov, Feng & Zanette (2026).

    | Method | Formula | When success is rare |
    |--------|---------|---------------------|
    | **MaxRL** | $(r - \mu) / \mu$ | $1/\mu$ is large, strongly amplifying rare successes |
    | **GRPO** | $(r - \mu) / \sigma$ | $\sigma \approx 0$ when $\mu \approx 0$, poorly defined |

    MaxRL provides an unbiased policy-gradient estimator that converges to maximum likelihood optimization
    as more sampling compute is allocated, achieving up to 20x test-time scaling efficiency gains over GRPO.

    The task: given N integers and a target, select a subset summing exactly to the target. Binary reward (0 or 1).
    """)
    return


@app.cell
def _():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import numpy as np
    import random

    return AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, np, random, torch


@app.cell
def _(random):
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
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    LoraConfig,
    config,
    get_peft_model,
    is_script_mode,
    torch,
):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    if is_script_mode:
        print(f"Device: {device}")
        print(f"Loading model: {config.model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map={"": device},
    )

    if is_script_mode:
        print("Model loaded, applying LoRA...")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    return device, model, optimizer, tokenizer


@app.cell
def _(check_answer, format_prompt, generate_subset_sum_problem, random, torch):
    def evaluate_model(model, tokenizer, device, num_elements, max_value, n_eval=50, seed=42):
        model.eval()
        _correct = 0
        random.seed(seed)
        for _i in range(n_eval):
            _nums, _tgt = generate_subset_sum_problem(num_elements, max_value)
            _prompt = format_prompt(_nums, _tgt)
            _messages = [{"role": "user", "content": _prompt}]
            _formatted = tokenizer.apply_chat_template(
                _messages, tokenize=False, add_generation_prompt=True
            )
            _inputs = tokenizer(_formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                _out = model.generate(
                    **_inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            _text = tokenizer.decode(_out[0, _inputs.input_ids.shape[1]:], skip_special_tokens=True)
            _correct += check_answer(_text, _nums, _tgt)
        return _correct / n_eval

    return (evaluate_model,)


@app.cell
def _(config, device, evaluate_model, is_script_mode, model, tokenizer, wandb):
    print("Evaluating initial accuracy (before training)...")
    initial_accuracy = evaluate_model(model, tokenizer, device, config.num_elements, config.max_value)
    wandb.log({"eval/initial_accuracy": initial_accuracy, "global_step": 0})
    print(f"Initial accuracy: {initial_accuracy:.1%}")
    return (initial_accuracy,)


@app.cell
def _(torch):
    def compute_advantages(rewards, method="maxrl"):
        mu = rewards.mean(dim=1, keepdim=True)
        if method == "maxrl":
            return (rewards - mu) / (mu + 1e-8)
        else:
            std = rewards.std(dim=1, keepdim=True)
            return (rewards - mu) / (std + 1e-8)

    return (compute_advantages,)


@app.cell
def _(
    check_answer,
    compute_advantages,
    config,
    device,
    evaluate_model,
    format_prompt,
    generate_subset_sum_problem,
    initial_accuracy,
    is_script_mode,
    model,
    np,
    optimizer,
    start_time,
    time,
    tokenizer,
    torch,
    wandb,
):
    import gc

    model.train()
    data = []
    global_step = 0

    # In the subset sum task, we generate infinite fresh problems.
    # Each epoch processes batch_size * 10 problems (like a virtual dataset).
    steps_per_epoch = 10

    if is_script_mode:
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Epochs: {config.epochs} | Batch size: {config.batch_size} | Group size: {config.group_size}")
        print(f"Steps per epoch: {steps_per_epoch} | Advantage: {config.advantage_method}")
        print("=" * 60)
        print()

    for _epoch in range(config.epochs):
        _epoch_rewards = []
        _epoch_losses = []
        _epoch_times = []

        for _step_in_epoch in range(steps_per_epoch):
            _tic = time.time()
            optimizer.zero_grad()

            # 1. Generate batch of problems
            _problems = [
                generate_subset_sum_problem(config.num_elements, config.max_value)
                for _ in range(config.batch_size)
            ]
            _prompts = [format_prompt(nums, tgt) for nums, tgt in _problems]

            # 2. Format as chat prompts
            _messages_batch = [[{"role": "user", "content": p}] for p in _prompts]
            _formatted = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in _messages_batch
            ]
            _inputs = tokenizer(_formatted, return_tensors="pt", padding=True).to(device)
            _prompt_len = _inputs.input_ids.shape[1]

            # 3. Sample G rollouts per problem
            with torch.no_grad():
                _outputs = model.generate(
                    **_inputs,
                    max_new_tokens=config.max_new_tokens,
                    num_return_sequences=config.group_size,
                    do_sample=True,
                    temperature=config.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # 4. Score rollouts
            _rewards_list = []
            for _i, (_nums, _tgt) in enumerate(_problems):
                _group_rewards = []
                for _j in range(config.group_size):
                    _idx = _i * config.group_size + _j
                    _gen_ids = _outputs[_idx, _prompt_len:]
                    _text = tokenizer.decode(_gen_ids, skip_special_tokens=True)
                    _reward = check_answer(_text, _nums, _tgt)
                    _group_rewards.append(_reward)
                _rewards_list.append(_group_rewards)

            _rewards = torch.tensor(_rewards_list, device=device, dtype=torch.float32)
            _success_rate = _rewards.mean().item()
            _has_any_success = (_rewards.sum() > 0).item()

            # 5. Compute advantages and update (skip if all rewards are zero)
            _loss_val = None
            _grad_norm_val = 0.0

            if _has_any_success:
                _advantages = compute_advantages(_rewards, method=config.advantage_method)
                _flat_advantages = _advantages.view(-1)

                # 6. Differentiable forward pass on generated sequences
                _model_inputs = {
                    "input_ids": _outputs,
                    "attention_mask": (_outputs != tokenizer.pad_token_id).long(),
                }
                _full_logits = model(**_model_inputs).logits

                # 7. Policy gradient loss
                _shift_logits = _full_logits[:, _prompt_len - 1:-1, :]
                _shift_labels = _outputs[:, _prompt_len:]
                _log_probs = torch.log_softmax(_shift_logits, dim=-1)
                _per_token = torch.gather(
                    _log_probs, 2, _shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                _completion_mask = (_shift_labels != tokenizer.pad_token_id).float()
                _seq_log_probs = (_per_token * _completion_mask).sum(dim=1)
                _loss = -(_seq_log_probs * _flat_advantages).mean()

                _loss.backward()
                _grad_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()

                _loss_val = _loss.item()
                _grad_norm_val = _grad_norm.item() if hasattr(_grad_norm, "item") else _grad_norm

                del _full_logits, _loss
            del _outputs
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            _batch_time = time.time() - _tic
            _epoch_rewards.append(_success_rate)
            if _loss_val is not None:
                _epoch_losses.append(_loss_val)
            _epoch_times.append(_batch_time)

            # Log per-step metrics
            _log_dict = {
                "batch/success_rate": _success_rate,
                "batch/time": _batch_time,
                "epoch": _epoch,
                "global_step": global_step,
            }
            if _loss_val is not None:
                _log_dict["batch/loss"] = _loss_val
                _log_dict["batch/grad_norm"] = _grad_norm_val
            wandb.log(_log_dict)

            data.append({
                "epoch": _epoch,
                "step": global_step,
                "success_rate": _success_rate,
                "loss": _loss_val,
                "time": _batch_time,
            })

            _loss_str = f"{_loss_val:.4f}" if _loss_val is not None else "N/A"
            print(
                f"[{_epoch+1}/{config.epochs}][{_step_in_epoch+1}/{steps_per_epoch}] "
                f"success={_success_rate:.3f} "
                f"loss={_loss_str:>8} "
                f"time={_batch_time:.1f}s"
            )

            global_step += 1

        # Epoch-level eval
        _eval_acc = evaluate_model(model, tokenizer, device, config.num_elements, config.max_value)
        model.train()

        # Epoch-level logging
        _epoch_log = {
            "epoch/train_success": np.mean(_epoch_rewards),
            "epoch/eval_accuracy": _eval_acc,
            "epoch/time_total": np.sum(_epoch_times),
            "epoch": _epoch + 1,
        }
        if _epoch_losses:
            _epoch_log["epoch/loss_mean"] = np.mean(_epoch_losses)
        wandb.log(_epoch_log)

        print(f"Epoch {_epoch+1} done | train_success={np.mean(_epoch_rewards):.3f} | eval_accuracy={_eval_acc:.3f}\n")

    _total_time = time.time() - start_time
    print(f"Training complete in {_total_time/60:.1f} minutes")

    return (data,)


@app.cell
def _(data, is_script_mode):
    if not is_script_mode:
        import polars as pl
        pl.DataFrame(data).plot.line(x="step", y="success_rate")
    return


@app.cell
def _(
    config,
    data,
    device,
    evaluate_model,
    initial_accuracy,
    mo,
    model,
    start_time,
    time,
    tokenizer,
    wandb,
):
    print("Evaluating final accuracy (after training)...")
    final_accuracy = evaluate_model(model, tokenizer, device, config.num_elements, config.max_value)

    wandb.log({"eval/final_accuracy": final_accuracy})
    wandb.run.summary["initial_accuracy"] = initial_accuracy
    wandb.run.summary["final_accuracy"] = final_accuracy
    wandb.run.summary["accuracy_improvement"] = final_accuracy - initial_accuracy
    wandb.run.summary["total_time_minutes"] = (time.time() - start_time) / 60
    wandb.finish()

    print(f"Initial accuracy: {initial_accuracy:.1%}")
    print(f"Final accuracy:   {final_accuracy:.1%}")
    print(f"Improvement:      {final_accuracy - initial_accuracy:+.1%}")

    mo.md(f"""
    **Evaluation (greedy, 50 problems)**

    | Stage | Accuracy |
    |-------|----------|
    | Initial | {initial_accuracy:.1%} |
    | Final | {final_accuracy:.1%} |
    | Improvement | {final_accuracy - initial_accuracy:+.1%} |
    """)
    return


if __name__ == "__main__":
    app.run()
