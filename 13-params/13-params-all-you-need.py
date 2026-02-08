# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.12.0",
#     "datasets==4.5.0",
#     "marimo>=0.19.8",
#     "polars",
#     "pydantic>=2.0.0",
#     "python-dotenv",
#     "torch==2.10.0",
#     "transformers==5.1.0",
#     "wandb",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    # Detect script mode
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    from pydantic import BaseModel, Field

    class ModelParams(BaseModel):
        # Model settings
        model_id: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
        n_params: int = Field(default=13, description="Number of TinyLoRA parameters")

        # Training hyperparameters
        batch_size: int = Field(default=8)
        group_size: int = Field(default=8)
        epochs: int = Field(default=20)
        learning_rate: float = Field(default=1e-5)

        # Generation settings
        max_new_tokens: int = Field(default=64)
        temperature: float = Field(default=0.9)

        # Data settings
        train_samples: int = Field(default=400, ge=1, le=7473, description="Training samples (max 7473)")
        test_samples: int = Field(default=100, ge=1, le=1319, description="Test samples (max 1319)")

        # WANDB settings
        wandb_project: str = Field(default="tiny-lora-gsm8k")
        wandb_run_name: str | None = Field(default=None)

    return (ModelParams,)


@app.cell
def _(ModelParams, is_script_mode, mo):
    import time

    # Start total timer
    start_time = time.time()

    # Parse CLI args (if any) into ModelParams
    # CLI args use dashes (--learning-rate) but ModelParams uses underscores (learning_rate)
    cli_args = {k.replace("-", "_"): v for k, v in mo.cli_args().items()}
    model_params = ModelParams(**cli_args)

    if is_script_mode:
        print("=" * 60)
        print("13 PARAMETERS IS ALL YOU NEED - Starting Training")
        print("=" * 60)
        print(f"\n📋 Configuration:")
        for key, value in model_params.model_dump().items():
            print(f"  {key}: {value}")
        print()
    return model_params, start_time, time


@app.cell
def _(is_script_mode, mo, model_params):
    # Available models
    available_models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]

    # Create UI controls with values from model_params
    model_dropdown = mo.ui.dropdown(
        options=available_models,
        value=model_params.model_id,
        label="Model"
    )

    learning_rate_input = mo.ui.number(
        value=model_params.learning_rate,
        start=1e-6,
        stop=1e-3,
        step=1e-6,
        label="Learning Rate"
    )

    batch_size_slider = mo.ui.slider(
        value=model_params.batch_size,
        start=1,
        stop=256,
        step=1,
        label="Batch Size"
    )

    group_size_slider = mo.ui.slider(
        value=model_params.group_size,
        start=1,
        stop=16,
        step=1,
        label="Group Size"
    )

    epochs_slider = mo.ui.slider(
        value=model_params.epochs,
        start=1,
        stop=50,
        step=1,
        label="Epochs"
    )

    n_params_input = mo.ui.number(
        value=model_params.n_params,
        start=1,
        stop=100,
        step=1,
        label="Number of TinyLoRA Parameters"
    )

    temperature_slider = mo.ui.slider(
        value=model_params.temperature,
        start=0.1,
        stop=2.0,
        step=0.1,
        label="Sampling Temperature"
    )

    # Display the controls in interactive mode
    _ui_output = None
    if not is_script_mode:
        _ui_output = mo.vstack([
            mo.md("## Training Configuration"),
            model_dropdown,
            learning_rate_input,
            batch_size_slider,
            group_size_slider,
            epochs_slider,
            n_params_input,
            temperature_slider
        ])

    _ui_output
    return (
        batch_size_slider,
        epochs_slider,
        group_size_slider,
        learning_rate_input,
        model_dropdown,
        n_params_input,
        temperature_slider,
    )


@app.cell
def _(
    ModelParams,
    batch_size_slider,
    epochs_slider,
    group_size_slider,
    is_script_mode,
    learning_rate_input,
    model_dropdown,
    model_params,
    n_params_input,
    temperature_slider,
):
    # Build final config from UI widgets or CLI args
    if is_script_mode:
        # Use CLI args directly in script mode
        config = model_params
    else:
        # Use UI widget values in interactive mode
        config = ModelParams(
            model_id=model_dropdown.value,
            learning_rate=learning_rate_input.value,
            batch_size=batch_size_slider.value,
            group_size=group_size_slider.value,
            epochs=epochs_slider.value,
            n_params=n_params_input.value,
            temperature=temperature_slider.value,
            # Keep other params from model_params
            max_new_tokens=model_params.max_new_tokens,
            train_samples=model_params.train_samples,
            test_samples=model_params.test_samples,
            wandb_project=model_params.wandb_project,
            wandb_run_name=model_params.wandb_run_name,
        )

    # Auto-generate run name if not provided
    if not config.wandb_run_name:
        model_short = config.model_id.split("/")[-1].replace("-Instruct", "")
        config = ModelParams(
            **{**config.model_dump(),
               "wandb_run_name": f"{model_short}-{config.n_params}p-lr{config.learning_rate}-bs{config.batch_size}-train{config.train_samples}"}
        )
    return (config,)


@app.cell
def _(config, is_script_mode):
    import wandb
    import os
    from pathlib import Path

    if is_script_mode:
        print("🔄 Initializing WANDB...")

    # Load .env for local runs (HF Jobs passes secrets as env vars)
    if Path(".env").exists():
        from dotenv import load_dotenv
        load_dotenv(".env")

    # Check WANDB_API_KEY is available
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError(
            "WANDB_API_KEY not found!\n"
            "Local: Create a .env file with WANDB_API_KEY=your_key\n"
            "HF Jobs: Pass --secrets-file .env to hf jobs uv run"
        )

    # Initialize WANDB (system metrics logged automatically)
    wandb_run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.model_dump(),
        tags=["tiny-lora", f"{config.n_params}-params"]
    )

    if is_script_mode:
        print(f"✅ WANDB initialized: {wandb_run.url}\n")
    return (wandb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 13 parameters is all you need [tm]

    This is a brief implementation inspired by the [Learning to Reason in 13 Parameters
    ](https://www.alphaxiv.org/abs/2602.04118) paper by John X. Morris, Niloofar Mireshghallah, Mark Ibrahim and Saeed Mahloujifar. It demonstrates a really cool idea: that you can finetune Qwen2.5 by finetuning only very few parameters! You can even get quite far with 13 parameter, but can we make a dent with just a single one?

    But how?! The trick is to re-use those parameters in a bunch of places in the architecture and to share them. Even better: you can use SVD to figure out the most imporant directions in the parameter space and your parameters can be used to merely scale those.

    ## 1. TinyLoRA Implementation
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.optim import AdamW
    from torch.utils.data import DataLoader


    class TinyLoRALayer(nn.Module):
        def __init__(self, original_layer, shared_vector, r=1):
            super().__init__()
            self.original_layer = original_layer
            self.tiny_weight = shared_vector # Now a vector of size u
            u = shared_vector.shape[0]

            # Freeze and SVD (use bfloat16 to reduce memory)
            W = self.original_layer.weight.data
            U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            U, S, Vh = U.to(W.dtype), S.to(W.dtype), Vh.to(W.dtype)

            self.register_buffer('U', U[:, :r].to(original_layer.weight.dtype))
            self.register_buffer('S_Vh', (torch.diag(S[:r]) @ Vh[:r, :]).to(original_layer.weight.dtype))

            # TinyLoRA's fixed random basis P: size (u, r, r)
            # For r=1, this is just a (u, 1, 1) tensor
            P = torch.randn(u, r, r)
            self.register_buffer('P', P.to(original_layer.weight.dtype))

        def forward(self, x):
            # 1. Project input to rank-r: (batch, seq, r)
            low_rank_x = x @ self.S_Vh.T.to(x.dtype)

            # 2. Compute the learned update matrix R = sum(v_i * P_i)
            # Fix: 'urk' to represent the random basis matrices
            # We sum over 'u' to get a single (r, r) update matrix
            R = torch.einsum('u,urk->rk', self.tiny_weight.to(x.dtype), self.P.to(x.dtype))

            # 3. Apply R and project back: (batch, seq, dim)
            update = (low_rank_x @ R) @ self.U.T.to(x.dtype)

            return self.original_layer(x) + update


    def apply_tiny_lora(model, n_params=13):
        # Just ONE parameter for the whole model
        shared_scalar = nn.Parameter(torch.zeros(n_params))

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(p in name for p in ["q_proj", "v_proj", "up_proj", "down_proj"]):
                parent = model.get_submodule(name.rsplit('.', 1)[0])
                setattr(parent, name.rsplit('.', 1)[1], TinyLoRALayer(module, shared_scalar))
        return model

    return (
        AdamW,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataLoader,
        apply_tiny_lora,
        load_dataset,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. General Setup
    """)
    return


@app.cell
def _(
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    apply_tiny_lora,
    config,
    is_script_mode,
    torch,
):
    # 2. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_script_mode:
        print(f"🔧 Device: {device}")
        print(f"📦 Loading model: {config.model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = 'left'

    # Qwen doesn't have a pad_token by default, which causes generation issues
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Load model directly to GPU so SVD runs on device
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if is_script_mode:
        print("✅ Model loaded")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    if is_script_mode:
        print(f"🔬 Applying TinyLoRA ({config.n_params} parameters)...")

    # Except the few parameters
    model = apply_tiny_lora(model, n_params=config.n_params)
    model.to(device=device, dtype=torch.bfloat16)

    # A relatively low learning rate seems wise here.
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    first_tiny_param = next(p for n, p in model.named_parameters() if "tiny_weight" in n)

    if is_script_mode:
        print(f"✅ TinyLoRA applied (device: {first_tiny_param.device}, dtype: {first_tiny_param.dtype})\n")
    else:
        print(f"TinyLoRA parameter device: {first_tiny_param.device}, dtype: {first_tiny_param.dtype}")
    return model, optimizer, tokenizer


@app.cell
def _(
    config,
    evaluate_accuracy,
    is_script_mode,
    model,
    test_dataset,
    tokenizer,
    wandb,
):
    def check_trainable_params(model):
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"📊 Total trainable parameters: {trainable_count}")
        if not is_script_mode:
            print(f"Trainable parameter names: {trainable_params}")
            print(list(filter(lambda p: p.requires_grad, model.parameters())))

    check_trainable_params(model)

    # Evaluate initial accuracy
    print("\n🎯 Evaluating initial accuracy...")
    initial_accuracy = evaluate_accuracy(
        model,
        tokenizer,
        test_dataset,
        num_samples=config.test_samples
    )

    wandb.log({
        "accuracy/initial": initial_accuracy,
        "epoch": 0
    })

    if is_script_mode:
        print(f"✅ Initial accuracy: {initial_accuracy:.2f}%\n")
    return (initial_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Data
    """)
    return


@app.cell
def _(config, is_script_mode, load_dataset):
    if is_script_mode:
        print(f"📚 Loading training data ({config.train_samples} samples)...")
    dataset = load_dataset("openai/gsm8k", "main", split=f"train[:{config.train_samples}]")
    if is_script_mode:
        print(f"✅ Training data loaded: {len(dataset)} examples\n")
    return (dataset,)


@app.cell
def _(config, is_script_mode, load_dataset):
    if is_script_mode:
        print(f"📚 Loading test data ({config.test_samples} samples)...")
    test_dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{config.test_samples}]")
    if is_script_mode:
        print(f"✅ Test data loaded: {len(test_dataset)} examples\n")
    return (test_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This dataset contains math-y reasoning. The benefit of this is that we get a nice "yes"/"no" answer out at the end that tells us if we are correct.
    """)
    return


@app.cell
def _(dataset):
    dataset[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Minimal GRPO Training Loop

    A few notes on the loop below.

    ### Groups vs. Batches

    In the context of the training loop we've built, **Batch Size** and **Group Size** serve two different purposes, though they both impact how many sequences the model processes at once.

    The **Batch Size** refers to the number of *unique questions* (prompts) the model looks at in one step. If your batch size is 2, the model is trying to learn from two different math problems at the same time. This is standard across almost all machine learning training. <alphaxiv-paper-citation title="Training Setup" page="5" first="We train on" last="batch size of 64." />

    The **Group Size** is specific to the RL algorithm (GRPO) used in this paper. For *each* unique question in your batch, the model generates multiple different attempts or "completions." If your group size is 4, the model will generate 4 different answers for every single question. This allows the model to compare its own outputs against each other:
    *   It looks at all 4 answers for Question A.
    *   It sees that 1 was correct and 3 were wrong.
    *   It calculates an "advantage" for the correct one by comparing it to the average performance of the group. <alphaxiv-paper-citation title="GRPO Method" page="1" first="Modern language models" last="verifiable rewards (RLVR)" />

    The relationship between them determines the total number of sequences processed:
    *   **Total Sequences** = `Batch Size` $\times$ `Group Size`.
    *   If you have a batch size of 2 and a group size of 4, the model processes **8 total sequences** in one forward pass.

    The paper finds that using a group-based approach is essential for TinyLoRA because it provides a clear signal for those 13 parameters to follow without needing a separate, memory-heavy "critic" model that standard RL usually requires.
    """)
    return


@app.cell
def _(torch):
    import gc

    def cleanup():
        # Helper to clear memory between reruns
        gc.collect()
        torch.cuda.empty_cache()

    cleanup()
    return


@app.cell
def _(
    DataLoader,
    config,
    dataset,
    evaluate_accuracy,
    initial_accuracy,
    is_script_mode,
    model,
    optimizer,
    start_time,
    test_dataset,
    time,
    tokenizer,
    torch,
    wandb,
):
    import numpy as np

    model.train()
    batch_size = config.batch_size
    group_size = config.group_size
    epochs = config.epochs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if is_script_mode:
        print("=" * 60)
        print("🚀 STARTING TRAINING")
        print("=" * 60)
        print(f"Epochs: {epochs} | Batch size: {batch_size} | Group size: {group_size}")
        print(f"Total batches per epoch: {len(dataloader)}")
        print("=" * 60)
        print()

    data = []

    for epoch in range(epochs):
        epoch_losses = []
        epoch_rewards = []
        epoch_times = []

        for batch_i, batch in enumerate(dataloader):
            tic = time.time()
            optimizer.zero_grad()

            # 1. Prepare Prompts
            prompts = [f"Question: {q}\nAnswer:" for q in batch['question']]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            # 2. Sample completions (Non-differentiable)
            with torch.no_grad():
                # Generate multiple completions per prompt for GRPO variance reduction
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    num_return_sequences=group_size,
                    do_sample=True,
                    temperature=config.temperature,
                    pad_token_id=tokenizer.pad_token_id
                ) # Shape: [batch_size * group_size, seq_len]

            # 3. Calculate Rewards and Group-Relative Advantages
            ground_truths = batch['answer']
            rewards_list = []

            for i in range(len(ground_truths)):
                gt_num = ground_truths[i].split("####")[-1].strip()
                group_rewards = []
                for j in range(group_size):
                    idx = i * group_size + j
                    text = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                    # Simple math verification reward
                    group_rewards.append(1.0 if gt_num in text else 0.0)
                rewards_list.append(group_rewards)

            rewards_tensor = torch.tensor(rewards_list, device=model.device, dtype=torch.bfloat16)
            mean = rewards_tensor.mean(dim=1, keepdim=True)
            std = rewards_tensor.std(dim=1, keepdim=True)
            advantages = (rewards_tensor - mean) / (std + 1e-8)
            flat_advantages = advantages.view(-1) # [batch_size * group_size]

            # 4. Differentiable Forward Pass
            # We re-run the model on the generated sequences to get logits for the sampled tokens
            model_inputs = {"input_ids": outputs, "attention_mask": (outputs != tokenizer.pad_token_id).long()}
            full_logits = model(**model_inputs).logits

            # 5. Policy Gradient Loss (GRPO)
            # Align logits with the tokens actually sampled
            # Shift so that logits at index t predict token at index t+1
            shift_logits = full_logits[:, prompt_len-1:-1, :]
            shift_labels = outputs[:, prompt_len:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            # Gather log_prob of the specific token that was sampled
            per_token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # Sum log_probs across the generated completion
            sequence_log_probs = per_token_log_probs.sum(dim=1)

            # Final objective: maximize (Advantage * LogProb)
            loss = -(sequence_log_probs * flat_advantages).mean()

            # 6. Backward and Step
            if loss.requires_grad:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()

                batch_time = time.time() - tic

                # Accumulate for epoch metrics
                epoch_losses.append(loss.item())
                epoch_rewards.append(rewards_tensor.mean().item())
                epoch_times.append(batch_time)

                # Log per-batch metrics to WANDB
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/grad_norm": grad_norm.item(),
                    "batch/avg_reward": rewards_tensor.mean().item(),
                    "batch/time": batch_time,
                    "epoch": epoch,
                    "batch": batch_i,
                    "global_step": epoch * len(dataloader) + batch_i
                })

                print(f"Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f} | Avg Reward: {rewards_tensor.mean():.2f} | time {batch_time:.2f}")
                data.append({"loss": loss.item(), "epoch": epoch, "batch": batch_i, "reward": rewards_tensor.mean().item(), "time": batch_time})

            del full_logits, outputs, loss, per_token_log_probs, model_inputs
            torch.cuda.empty_cache()

        # Log epoch-level aggregates
        wandb.log({
            "epoch/loss_mean": np.mean(epoch_losses),
            "epoch/loss_std": np.std(epoch_losses),
            "epoch/reward_mean": np.mean(epoch_rewards),
            "epoch/time_total": np.sum(epoch_times),
            "epoch": epoch + 1
        })

        print(f"Epoch {epoch} done | Avg Loss: {np.mean(epoch_losses):.4f} | Avg Reward: {np.mean(epoch_rewards):.2f}")

    # Evaluate final accuracy
    print("Evaluating final accuracy...")
    final_accuracy = evaluate_accuracy(
        model,
        tokenizer,
        test_dataset,
        num_samples=config.test_samples
    )

    wandb.log({
        "accuracy/final": final_accuracy,
        "accuracy/improvement": final_accuracy - initial_accuracy,
        "epoch": config.epochs
    })

    # Calculate total time
    total_time = time.time() - start_time
    total_minutes = total_time / 60

    # Summary metrics
    wandb.run.summary["best_accuracy"] = final_accuracy
    wandb.run.summary["accuracy_gain"] = final_accuracy - initial_accuracy
    wandb.run.summary["total_time_seconds"] = total_time
    wandb.run.summary["total_time_minutes"] = total_minutes

    # Log total time
    wandb.log({
        "timing/total_seconds": total_time,
        "timing/total_minutes": total_minutes,
    })

    print(f"\n=== Training Complete ===")
    print(f"Initial Accuracy: {initial_accuracy:.2f}%")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Improvement: {final_accuracy - initial_accuracy:.2f}%")
    print(f"Total Time: {total_minutes:.2f} minutes ({total_time:.1f} seconds)")

    wandb.finish()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utilities
    """)
    return


@app.cell
def _(data, is_script_mode):
    # Skip plotting in script mode (requires altair which isn't installed)
    if not is_script_mode:
        import polars as pl
        pl.DataFrame(data).plot.scatter(x="epoch", y="reward")
    return


@app.cell
def _(torch, wandb):
    def evaluate_accuracy(model, tokenizer, dataset, num_samples=100, batch_size=50, log_samples=False):
        """
        Evaluates Pass@1 accuracy with optional WANDB sample logging.
        """
        model.eval()
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        correct = 0
        total = 0

        # Select subset
        num_to_eval = min(num_samples, len(dataset))
        eval_subset = dataset.select(range(num_to_eval))

        # Prepare WANDB table if logging samples
        sample_table = None
        if log_samples and wandb.run is not None:
            sample_table = wandb.Table(columns=["question", "ground_truth", "prediction", "correct"])

        for i in range(0, num_to_eval, batch_size):
            # Correctly get a list of examples from the dataset slice
            batch_indices = range(i, min(i + batch_size, num_to_eval))
            batch = [eval_subset[idx] for idx in batch_indices]

            prompts = [f"Question: {ex['question']}\nAnswer:" for ex in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True
                    )

            # Decode only the generated part
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for j, response in enumerate(responses):
                gt_answer = batch[j]['answer'].split("####")[-1].strip()
                is_correct = gt_answer in response

                # Simple check for the answer in response
                if is_correct:
                    correct += 1
                total += 1

                # Log first 10 examples to WANDB table
                if sample_table is not None and total <= 10:
                    sample_table.add_data(
                        batch[j]['question'],
                        gt_answer,
                        response,
                        is_correct
                    )

            print(f"Evaluated {total}/{num_to_eval}... Current Accuracy: {100 * correct / total:.2f}%")

        accuracy = 100 * correct / total
        print(f"\nFinal Pass@1 Accuracy: {accuracy:.2f}%")

        # Log sample table to WANDB
        if sample_table is not None:
            wandb.log({"predictions": sample_table})

        return accuracy


    return (evaluate_accuracy,)


@app.cell
def _():
    # def evaluate_accuracy(model, tokenizer, dataset, num_samples=100):
    #     """
    #     Evaluates Pass@1 accuracy on a subset of the dataset.
    #     """
    #     model.eval()
    #     correct = 0
    #     total = 0

    #     # We use a small subset for a quick check
    #     eval_subset = dataset.select(range(min(num_samples, len(dataset))))

    #     for i, example in enumerate(eval_subset):
    #         prompt = f"Question: {example['question']}\nAnswer:"
    #         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    #         with torch.no_grad():
    #             # Greedy decoding for stable evaluation
    #             output_ids = model.generate(
    #                 **inputs, 
    #                 max_new_tokens=256, 
    #                 do_sample=False, 
    #                 pad_token_id=tokenizer.pad_token_id
    #             )

    #         response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    #         # Extract the final number from the ground truth (format: "#### 42")
    #         gt_answer = example['answer'].split("####")[-1].strip()

    #         if gt_answer in response:
    #             correct += 1
    #         total += 1

    #         if (i + 1) % 10 == 0:
    #             print(f"Evaluated {i+1}/{num_samples}... Current Accuracy: {100 * correct / total:.2f}%")

    #     accuracy = 100 * correct / total
    #     print(f"\nFinal Pass@1 Accuracy: {accuracy:.2f}%")
    #     return accuracy
    return


if __name__ == "__main__":
    app.run()
