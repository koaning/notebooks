# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "trl",
#     "ipython==9.10.0",
#     "datasets",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learning to Reason in 13 Parameters

    This notebook reproduces the key findings from the **TinyLoRA** paper, which shows that
    just **13 trainable parameters** (26 bytes in bf16) can unlock reasoning in an 8B language model.

    The trick is **extreme weight tying**: a single set of scalars controls rank-decomposed updates
    across every transformer layer, trained with **Group Relative Policy Optimization (GRPO)**
    using verifiable reward signals.

    We cover two challenges:
    - **GSM8K** — grade school math problems
    - **Logic Grid Puzzles** — Zebra-style deduction puzzles
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The 13-Parameter Breakdown

    The paper assigns different ranks to different projection types, all **tied across every transformer layer**:

    | Category | Targets | Rank | Params per target | Count | Subtotal |
    |---|---|---|---|---|---|
    | Attention projections | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 2 | 2 scalars | 4 | **8** |
    | MLP projections | `gate_proj`, `up_proj`, `down_proj` | 1 | 1 scalar | 3 | **3** |
    | LayerNorms | `input_layernorm`, `post_attention_layernorm` | — | 1 scalar | 2 | **2** |
    | **Total** | | | | | **13** |

    Each attention projection gets a **rank-2** update: two scalars, each scaling an independent
    random rank-1 matrix ($U \cdot v_r \cdot V^\top$). MLP projections get **rank-1** updates.
    LayerNorms get a single trainable scaling factor applied identically across all layers.

    The sliders below let you experiment with different rank configurations.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn

    return nn, torch


@app.cell
def _(nn, torch):
    class TinyLoRAController(nn.Module):
        """The 'remote control' for the model. With default settings, holds exactly 13 trainable scalars."""

        def __init__(self, attn_rank=2, mlp_rank=1, train_layernorms=True, bf16=True):
            super().__init__()
            self.attn_rank = attn_rank
            self.mlp_rank = mlp_rank
            self.train_layernorms = train_layernorms
            self.dtype = torch.bfloat16 if bf16 else torch.float32

            # Attention: 4 projections (q, k, v, o) x attn_rank scalars each
            self.attn_v = nn.Parameter(torch.zeros(4, attn_rank))
            # MLP: 3 projections (gate, up, down) x mlp_rank scalars each
            self.mlp_v = nn.Parameter(torch.zeros(3, mlp_rank))
            # LayerNorms: 2 scaling factors (input_layernorm, post_attention_layernorm)
            if train_layernorms:
                self.norm_v = nn.Parameter(torch.zeros(2))

        @property
        def num_params(self):
            return 4 * self.attn_rank + 3 * self.mlp_rank + (2 if self.train_layernorms else 0)

        def get_projection_update(self, proj_type, proj_idx, d_out, d_in, device):
            """Generate a rank-r weight update for a linear projection.

            proj_type: 'attn' or 'mlp'
            proj_idx: index within the type (0-3 for attn, 0-2 for mlp)
            """
            v = self.attn_v[proj_idx] if proj_type == "attn" else self.mlp_v[proj_idx]
            rank = self.attn_rank if proj_type == "attn" else self.mlp_rank
            delta_W = torch.zeros(d_out, d_in, device=device, dtype=self.dtype)
            for r in range(rank):
                seed = hash((proj_type, proj_idx, r)) % (2**31)
                gen = torch.Generator(device=device).manual_seed(seed)
                U = torch.randn(d_out, 1, generator=gen, device=device, dtype=self.dtype)
                V = torch.randn(1, d_in, generator=gen, device=device, dtype=self.dtype)
                delta_W = delta_W + v[r] * (U @ V)
            return delta_W

        def get_norm_scale(self, norm_idx):
            """Return multiplicative scale for a LayerNorm (applied as 1 + v)."""
            return 1.0 + self.norm_v[norm_idx]

    return (TinyLoRAController,)


@app.cell
def _(torch):
    ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
    MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]
    NORM_NAMES = ["input_layernorm", "post_attention_layernorm"]

    def patch_model_with_tinylora(model, controller):
        """Freeze all base model params, then attach TinyLoRA hooks to every transformer layer.

        Returns a list of hook handles (call handle.remove() to undo patching).
        """
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        handles = []

        for layer in model.model.layers:
            # Attention projection hooks
            for idx, proj_name in enumerate(ATTN_PROJS):
                linear = getattr(layer.self_attn, proj_name)
                d_out, d_in = linear.weight.shape

                def make_attn_hook(proj_idx, d_out, d_in):
                    def hook(module, input, output):
                        delta_W = controller.get_projection_update("attn", proj_idx, d_out, d_in, output.device)
                        return output + torch.nn.functional.linear(input[0], delta_W)
                    return hook

                h = linear.register_forward_hook(make_attn_hook(idx, d_out, d_in))
                handles.append(h)

            # MLP projection hooks
            for idx, proj_name in enumerate(MLP_PROJS):
                linear = getattr(layer.mlp, proj_name)
                d_out, d_in = linear.weight.shape

                def make_mlp_hook(proj_idx, d_out, d_in):
                    def hook(module, input, output):
                        delta_W = controller.get_projection_update("mlp", proj_idx, d_out, d_in, output.device)
                        return output + torch.nn.functional.linear(input[0], delta_W)
                    return hook

                h = linear.register_forward_hook(make_mlp_hook(idx, d_out, d_in))
                handles.append(h)

            # LayerNorm hooks
            if controller.train_layernorms:
                for idx, norm_name in enumerate(NORM_NAMES):
                    norm_module = getattr(layer, norm_name)

                    def make_norm_hook(norm_idx):
                        def hook(module, input, output):
                            return output * controller.get_norm_scale(norm_idx)
                        return hook

                    h = norm_module.register_forward_hook(make_norm_hook(idx))
                    handles.append(h)

        return handles

    return (patch_model_with_tinylora,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    task_selector = mo.ui.dropdown(
        options={"GSM8K (Math)": "gsm8k", "Logic Grid Puzzles": "logic"},
        value="GSM8K (Math)",
        label="Task",
    )
    attn_rank_slider = mo.ui.slider(
        start=1, stop=4, step=1, value=2, label="Attention rank"
    )
    mlp_rank_slider = mo.ui.slider(
        start=1, stop=4, step=1, value=1, label="MLP rank"
    )
    train_norms_switch = mo.ui.switch(value=True, label="Train LayerNorms")
    num_steps_slider = mo.ui.slider(
        start=100, stop=5000, step=100, value=2000, label="Training steps"
    )
    lr_slider = mo.ui.slider(
        start=-5, stop=-2, step=0.25, value=-4, label="Log₁₀ learning rate"
    )
    group_size_slider = mo.ui.slider(
        start=2, stop=16, step=2, value=8, label="GRPO group size"
    )
    max_tokens_slider = mo.ui.slider(
        start=64, stop=2048, step=64, value=512, label="Max completion tokens"
    )
    train_button = mo.ui.run_button(label="Train")
    return (
        attn_rank_slider,
        group_size_slider,
        lr_slider,
        max_tokens_slider,
        mlp_rank_slider,
        num_steps_slider,
        task_selector,
        train_button,
        train_norms_switch,
    )


@app.cell
def _():
    import ipython

    return


@app.cell
def _(
    attn_rank_slider,
    group_size_slider,
    lr_slider,
    max_tokens_slider,
    mlp_rank_slider,
    mo,
    num_steps_slider,
    task_selector,
    train_button,
    train_norms_switch,
):
    _total = (
        4 * attn_rank_slider.value
        + 3 * mlp_rank_slider.value
        + (2 if train_norms_switch.value else 0)
    )
    _lr_display = 10 ** lr_slider.value

    mo.vstack([
        mo.hstack([task_selector, train_norms_switch], justify="start", gap=1),
        mo.hstack([attn_rank_slider, mlp_rank_slider], justify="start", gap=1),
        mo.md(f"**Total trainable parameters: {_total}** ({_total * 2} bytes in bf16)"),
        mo.hstack([num_steps_slider, lr_slider], justify="start", gap=1),
        mo.md(f"Learning rate: `{_lr_display:.2e}`"),
        mo.hstack([group_size_slider, max_tokens_slider], justify="start", gap=1),
        train_button,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data
    """)
    return


@app.cell
def _(task_selector):
    import re
    from datasets import load_dataset

    def extract_gsm8k_answer(answer_text):
        """Extract the numeric answer after #### in GSM8K."""
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return ""

    def load_gsm8k():
        ds = load_dataset("openai/gsm8k", "main")
        train_data = []
        for ex in ds["train"]:
            train_data.append({
                "prompt": ex["question"],
                "ground_truth": extract_gsm8k_answer(ex["answer"]),
            })
        test_data = []
        for ex in ds["test"]:
            test_data.append({
                "prompt": ex["question"],
                "ground_truth": extract_gsm8k_answer(ex["answer"]),
            })
        return train_data, test_data

    LOGIC_PUZZLES_TRAIN = [
        {
            "prompt": (
                "There are 3 houses in a row: red, blue, green. "
                "The cat lives in the red house. The dog does not live in the blue house. "
                "The bird lives in the house next to the cat. "
                "Who lives in the green house?"
            ),
            "ground_truth": "dog",
        },
        {
            "prompt": (
                "Three friends — Alice, Bob, and Carol — each have a different pet: a cat, a dog, and a fish. "
                "Alice does not have the cat. Bob does not have the dog. Carol has the fish. "
                "Who has the cat?"
            ),
            "ground_truth": "Bob",
        },
        {
            "prompt": (
                "Four students sit in a row. Emma is not next to Finn. "
                "Grace is at one end. Henry is next to Grace. "
                "Finn is not at the other end. "
                "What is the order from left to right?"
            ),
            "ground_truth": "Grace, Henry, Finn, Emma",
        },
        {
            "prompt": (
                "There are 3 boxes: A, B, C. One has gold, one has silver, one is empty. "
                "Box A says 'The gold is in here.' Box B says 'This box is empty.' "
                "Box C says 'The gold is in Box B.' Exactly one statement is true. "
                "Which box has the gold?"
            ),
            "ground_truth": "B",
        },
        {
            "prompt": (
                "Five people — P, Q, R, S, T — are ranked 1st to 5th. "
                "P is ranked higher than Q. R is ranked immediately below S. "
                "T is ranked 3rd. Q is not ranked last. "
                "What is the ranking from 1st to 5th?"
            ),
            "ground_truth": "P, S, T, R, Q",
        },
    ]

    # For logic puzzles, we replicate the small set to have enough training data
    LOGIC_PUZZLES_TEST = [
        {
            "prompt": (
                "Three coworkers — Xena, Yuri, and Zara — each drink a different beverage: coffee, tea, or juice. "
                "Xena does not drink coffee. Yuri drinks tea. "
                "What does Xena drink?"
            ),
            "ground_truth": "juice",
        },
        {
            "prompt": (
                "There are 4 lockers numbered 1–4. Each stores one item: a book, a hat, a phone, or keys. "
                "The book is in an even-numbered locker. The hat is not in locker 1 or 4. "
                "The phone is in locker 1. "
                "Where are the keys?"
            ),
            "ground_truth": "locker 4",
        },
    ]

    selected_task = task_selector.value
    return (
        LOGIC_PUZZLES_TEST,
        LOGIC_PUZZLES_TRAIN,
        load_gsm8k,
        re,
        selected_task,
    )


@app.cell
def _(
    LOGIC_PUZZLES_TEST,
    LOGIC_PUZZLES_TRAIN,
    is_script_mode,
    load_gsm8k,
    selected_task,
    system_prompt,
):
    from datasets import Dataset

    if selected_task == "gsm8k":
        _train_list, _test_list = load_gsm8k()
    else:
        # Replicate logic puzzles to have more training examples
        _train_list = LOGIC_PUZZLES_TRAIN * 200
        _test_list = LOGIC_PUZZLES_TEST

    if is_script_mode:
        # Use a tiny subset for CI testing
        _train_list = _train_list[:4]
        _test_list = _test_list[:2]

    # Format prompts as chat messages with system prompt for trl
    def _format_as_chat(examples):
        return {
            "prompt": [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p},
                ]
                for p in examples["prompt"]
            ]
        }

    train_dataset = Dataset.from_list(_train_list).map(_format_as_chat, batched=True)
    test_dataset = Dataset.from_list(_test_list).map(_format_as_chat, batched=True)
    return test_dataset, train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reward Function

    The reward is binary: **1.0** if the model's answer matches the gold answer, **0.0** otherwise.
    We extract the answer from inside `\boxed{}` in the model's completion.

    - **Math (GSM8K):** Numeric comparison after stripping commas/whitespace.
    - **Logic puzzles:** Case-insensitive string match.
    """)
    return


@app.cell
def _(re, selected_task):
    def extract_boxed(text):
        """Extract content inside \\boxed{} from a completion."""
        match = re.search(r"\\boxed\{(.*?)\}", text)
        if match:
            return match.group(1).strip()
        return ""

    def verify_math(pred, gold):
        """Check if two numeric answers match."""
        pred_clean = pred.replace(",", "").replace(" ", "").replace("$", "")
        gold_clean = gold.replace(",", "").replace(" ", "").replace("$", "")
        return pred_clean == gold_clean

    def reward_fn(completions, ground_truth, **kwargs):
        """Reward function compatible with trl's GRPOTrainer.

        Returns 1.0 for correct answers, 0.0 for incorrect.
        """
        rewards = []
        for completion, gold in zip(completions, ground_truth):
            # Handle conversational format (list of message dicts)
            if isinstance(completion, list):
                completion = completion[-1]["content"] if completion else ""
            pred = extract_boxed(completion)
            if not pred:
                rewards.append(0.0)
                continue
            if selected_task == "gsm8k":
                rewards.append(1.0 if verify_math(pred, gold) else 0.0)
            else:
                rewards.append(1.0 if pred.lower() == gold.lower() else 0.0)
        return rewards

    return extract_boxed, reward_fn, verify_math


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Model & Training
    """)
    return


@app.cell
def _(selected_task):
    SYSTEM_PROMPTS = {
        "gsm8k": "Think step by step. Put your final answer in \\boxed{}.",
        "logic": "Analyze the constraints step by step. Put your final answer in \\boxed{}.",
    }
    system_prompt = SYSTEM_PROMPTS[selected_task]
    return (system_prompt,)


@app.cell
def _(
    TinyLoRAController,
    attn_rank_slider,
    is_script_mode,
    mlp_rank_slider,
    mo,
    patch_model_with_tinylora,
    torch,
    train_norms_switch,
):
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _model_name = "Qwen/Qwen2.5-7B-Instruct"

    if is_script_mode:
        # In script mode we don't actually load the model
        model = None
        tokenizer = None
        controller = TinyLoRAController(
            attn_rank=attn_rank_slider.value,
            mlp_rank=mlp_rank_slider.value,
            train_layernorms=train_norms_switch.value,
        )
    else:
        from huggingface_hub import snapshot_download

        with mo.status.spinner(f"Downloading {_model_name} (skip if cached)..."):
            snapshot_download(_model_name)

        with mo.status.spinner(f"Loading {_model_name} into memory..."):
            tokenizer = AutoTokenizer.from_pretrained(_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(
                _model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        controller = TinyLoRAController(
            attn_rank=attn_rank_slider.value,
            mlp_rank=mlp_rank_slider.value,
            train_layernorms=train_norms_switch.value,
        ).to(model.device)

        hook_handles = patch_model_with_tinylora(model, controller)

    mo.output.append(mo.md(f"Controller: **{controller.num_params} trainable parameters**"))
    return controller, model, tokenizer


@app.cell
def _(
    controller,
    group_size_slider,
    is_script_mode,
    lr_slider,
    max_tokens_slider,
    mo,
    model,
    num_steps_slider,
    reward_fn,
    tokenizer,
    train_button,
    train_dataset,
):
    from pathlib import Path
    import tempfile
    import time

    training_log = None

    if is_script_mode:
        mo.output.append(mo.md("*Script mode: skipping training (requires GPU).*"))
    else:
        mo.stop(not train_button.value, mo.md("*Click **Train** to start GRPO training.*"))

        from trl import GRPOTrainer, GRPOConfig
        from transformers import TrainerCallback

        _output_dir = Path(tempfile.mkdtemp()) / "tinylora-grpo"
        _lr = 10 ** lr_slider.value
        _max_steps = num_steps_slider.value
        _max_tokens = max_tokens_slider.value

        _config = GRPOConfig(
            output_dir=str(_output_dir),
            num_generations=group_size_slider.value,
            max_completion_length=_max_tokens,
            max_steps=_max_steps,
            per_device_train_batch_size=group_size_slider.value,
            gradient_accumulation_steps=1,
            learning_rate=_lr,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        # Live feedback callback — appends metrics to cell output during training
        class _MarimoProgressCallback(TrainerCallback):
            def __init__(self):
                self._step_start = None

            def on_step_begin(self, args, state, control, **kwargs):
                self._step_start = time.time()
                mo.output.append(mo.md(
                    f"`Step {state.global_step + 1}/{_max_steps} — generating {group_size_slider.value} "
                    f"completions (max {_max_tokens} tokens each)...`"
                ))

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                _step = state.global_step
                _elapsed = time.time() - self._step_start if self._step_start else 0
                _reward = logs.get("reward", None)
                _length = logs.get("completions/mean_length", None)
                _parts = [f"Step {_step}/{_max_steps}", f"{_elapsed:.1f}s"]
                if _reward is not None:
                    _parts.append(f"reward: {_reward:.3f}")
                if _length is not None:
                    _parts.append(f"mean length: {_length:.0f}")
                mo.output.append(mo.md(f"`{' | '.join(_parts)}`"))

        # Build a custom optimizer that only trains the controller parameters
        import torch as _torch
        _optimizer = _torch.optim.AdamW(controller.parameters(), lr=_lr)

        _trainer = GRPOTrainer(
            model=model,
            args=_config,
            reward_funcs=reward_fn,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            optimizers=(_optimizer, None),
            callbacks=[_MarimoProgressCallback()],
        )

        mo.output.append(mo.md(f"**Training started** — {_max_steps} steps, lr={_lr:.2e}"))
        _trainer.train()
        mo.output.append(mo.md("**Training complete!**"))
        training_log = _trainer.state.log_history
    return (training_log,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training Curves
    """)
    return


@app.cell
def _(mo, training_log):
    import matplotlib.pyplot as plt
    import numpy as np

    mo.stop(training_log is None, mo.md("*No training data yet.*"))

    _reward_steps = []
    _rewards = []
    _length_steps = []
    _lengths = []

    for entry in training_log:
        if "reward" in entry and "step" in entry:
            _reward_steps.append(entry["step"])
            _rewards.append(entry["reward"])
        if "completions/mean_length" in entry and "step" in entry:
            _length_steps.append(entry["step"])
            _lengths.append(entry["completions/mean_length"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(_reward_steps, _rewards, "b-", alpha=0.7)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Reward over Training")
    ax1.grid(True, alpha=0.3)

    ax2.plot(_length_steps, _lengths, "r-", alpha=0.7)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Completion Length")
    ax2.set_title("Response Length over Training")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation

    We evaluate the trained controller on held-out test data. For comparison, we also
    evaluate the **baseline** (controller parameters zeroed out) to measure the accuracy
    gain from the 13 learned parameters.
    """)
    return


@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="Evaluate")
    eval_samples_slider = mo.ui.slider(
        start=10, stop=500, step=10, value=200, label="Eval samples"
    )
    mo.hstack([eval_samples_slider, eval_button], justify="start", gap=1)
    return eval_button, eval_samples_slider


@app.cell
def _(
    controller,
    eval_button,
    eval_samples_slider,
    extract_boxed,
    is_script_mode,
    mo,
    model,
    selected_task,
    system_prompt,
    test_dataset,
    tokenizer,
    torch,
    verify_math,
):
    eval_results = None

    if is_script_mode:
        mo.output.append(mo.md("*Script mode: skipping evaluation.*"))
    else:
        mo.stop(not eval_button.value, mo.md("*Click **Evaluate** to run evaluation.*"))

        def _evaluate(zero_controller=False):
            """Run greedy evaluation. If zero_controller=True, temporarily zero out params for baseline."""
            _saved_params = {}
            if zero_controller:
                for name, param in controller.named_parameters():
                    _saved_params[name] = param.data.clone()
                    param.data.zero_()

            _n = min(eval_samples_slider.value, len(test_dataset))
            _results = []

            for _i in range(_n):
                _example = test_dataset[_i]
                _prompt = _example["prompt"]
                _gold = _example["ground_truth"]

                _messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _prompt},
                ]
                _input_text = tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
                _inputs = tokenizer(_input_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    _output_ids = model.generate(
                        **_inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                    )
                _completion = tokenizer.decode(_output_ids[0][_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                _pred = extract_boxed(_completion)
                if selected_task == "gsm8k":
                    _correct = verify_math(_pred, _gold) if _pred else False
                else:
                    _correct = (_pred.lower() == _gold.lower()) if _pred else False

                _results.append({
                    "prompt": _prompt[:100] + "..." if len(_prompt) > 100 else _prompt,
                    "prediction": _pred,
                    "gold": _gold,
                    "correct": _correct,
                    "completion_length": len(_completion),
                    "completion": _completion,
                })

            if zero_controller:
                for name, param in controller.named_parameters():
                    param.data.copy_(_saved_params[name])

            return _results

        with mo.status.spinner("Evaluating baseline (controller zeroed)..."):
            _baseline_results = _evaluate(zero_controller=True)

        with mo.status.spinner("Evaluating trained controller..."):
            _trained_results = _evaluate(zero_controller=False)

        eval_results = {
            "baseline": _baseline_results,
            "trained": _trained_results,
        }
    return (eval_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluation Results
    """)
    return


@app.cell
def _(eval_results, mo, np):
    mo.stop(eval_results is None, mo.md("*No evaluation results yet.*"))

    def _summarize(results):
        _correct = sum(1 for r in results if r["correct"])
        _total = len(results)
        _acc = _correct / _total * 100 if _total > 0 else 0
        _mean_len = np.mean([r["completion_length"] for r in results])
        return _acc, _mean_len, _correct, _total

    _b_acc, _b_len, _b_correct, _b_total = _summarize(eval_results["baseline"])
    _t_acc, _t_len, _t_correct, _t_total = _summarize(eval_results["trained"])

    mo.md(f"""
    | | Baseline (zeroed) | Trained |
    |---|---|---|
    | **Accuracy** | {_b_acc:.1f}% ({_b_correct}/{_b_total}) | {_t_acc:.1f}% ({_t_correct}/{_t_total}) |
    | **Mean response length** | {_b_len:.0f} tokens | {_t_len:.0f} tokens |
    | **Improvement** | — | **+{_t_acc - _b_acc:.1f}%** |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sample Outputs
    """)
    return


@app.cell
def _(eval_results, mo):
    mo.stop(eval_results is None, mo.md("*No evaluation results yet.*"))

    _samples = []
    for _i in range(min(5, len(eval_results["trained"]))):
        _t = eval_results["trained"][_i]
        _b = eval_results["baseline"][_i]
        _samples.append({
            "Prompt": _t["prompt"],
            "Gold": _t["gold"],
            "Baseline pred": _b["prediction"],
            "Baseline correct": _b["correct"],
            "Trained pred": _t["prediction"],
            "Trained correct": _t["correct"],
        })

    mo.ui.table(_samples, label="Sample predictions (first 5)")
    return


if __name__ == "__main__":
    app.run()
