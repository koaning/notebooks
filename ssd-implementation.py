# /// script
# dependencies = [
#     "accelerate==1.13.0",
#     "marimo",
#     "pydantic==2.13.0",
#     "python-dotenv==1.2.2",
#     "rich==15.0.0",
#     "torch==2.11.0",
#     "transformers==5.5.4",
#     "wandb==0.26.0",
#     "wigglystuff==0.3.2",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv

    load_dotenv(".env")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment Keys
    """)
    return


@app.cell
def _(env_config, is_script_mode, mo):
    if not is_script_mode and env_config is not None:
        mo.vstack(
            [
                mo.md(
                    "If W&B is already configured locally, this can be ignored."
                ),
                env_config,
            ]
        )
    else:
        None
    return


@app.cell
def _(ModelParams, mo):
    import wandb
    import sys

    is_script_mode = mo.app_meta().mode == "script"

    if is_script_mode and not mo.cli_args():
        from rich.console import Console
        from rich.table import Table

        table = Table(title="CLI Options")
        table.add_column("Flag", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Default", style="yellow")
        table.add_column("Description")

        for name, field in ModelParams.model_fields.items():
            flag = f"--{name.replace('_', '-')}"
            type_name = (
                field.annotation.__name__
                if hasattr(field.annotation, "__name__")
                else str(field.annotation)
            )
            table.add_row(
                flag, type_name, str(field.default), field.description or ""
            )

        Console().print(table)
        sys.exit(0)

    env_config = None
    if not is_script_mode:
        from wigglystuff import EnvConfig

        env_config = mo.ui.anywidget(
            EnvConfig(
                {
                    "WANDB_API_KEY": lambda key: wandb.login(key=key, verify=True),
                }
            )
        )
    return env_config, is_script_mode, wandb


@app.cell
def _():
    import hashlib
    import json
    from typing import Literal
    from pydantic import computed_field, BaseModel, Field

    MODEL_CHOICES = Literal[
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]

    class ModelParams(BaseModel):
        difficulty: int = Field(default=2, description="Task difficulty level (1, 2, or 3).")
        model_id: MODEL_CHOICES = Field(default="Qwen/Qwen2.5-0.5B-Instruct", description="HuggingFace model ID.")
        num_train: int = Field(default=150, description="Number of training tasks.")
        num_test: int = Field(default=100, description="Number of test tasks.")
        learning_rate: float = Field(default=1e-5, description="Learning rate for AdamW.")
        epochs: int = Field(default=8, description="Number of fine-tuning epochs.")
        temperature: float = Field(default=1.2, description="Sampling temperature for SSD.")
        seed: int = Field(default=42, description="Random seed for reproducibility.")
        wandb_project: str = Field(default="ssd-benchmark", description="W&B project name.")

        @computed_field
        @property
        def run_name(self) -> str:
            parts = [
                f"d{self.difficulty}",
                f"e{self.epochs}",
                f"lr{self.learning_rate:.0e}",
                f"t{self.temperature}",
            ]
            params_dict = {
                "difficulty": self.difficulty,
                "model_id": self.model_id,
                "num_train": self.num_train,
                "num_test": self.num_test,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "temperature": self.temperature,
                "seed": self.seed,
            }
            h = hashlib.md5(json.dumps(params_dict, sort_keys=True).encode()).hexdigest()[:6]
            return "-".join(parts) + f"-{h}"

    return (ModelParams,)


@app.cell
def _(mo):
    params_form = mo.md("""
    ## Model parameters

    {difficulty}
    {model_id}
    {num_train}
    {num_test}
    {learning_rate}
    {epochs}
    {temperature}
    {seed}
    {wandb_project}
    """).batch(
        difficulty=mo.ui.slider(1, 3, value=2, step=1, label="Difficulty"),
        model_id=mo.ui.dropdown(options=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"], value="Qwen/Qwen2.5-0.5B-Instruct", label="Model ID"),
        num_train=mo.ui.slider(10, 500, value=150, step=10, label="Num train"),
        num_test=mo.ui.slider(10, 500, value=100, step=10, label="Num test"),
        learning_rate=mo.ui.slider(1e-6, 1e-3, value=1e-5, step=1e-6, label="Learning rate"),
        epochs=mo.ui.slider(1, 30, value=8, step=1, label="Epochs"),
        temperature=mo.ui.slider(0.1, 2.0, value=1.2, step=0.1, label="Temperature"),
        seed=mo.ui.number(value=42, start=0, stop=9999, label="Seed"),
        wandb_project=mo.ui.text(value="ssd-benchmark", label="W&B project"),
    ).form()
    return (params_form,)


@app.cell
def _(is_script_mode, params_form):
    params_form if not is_script_mode else None
    return


@app.cell
def _(ModelParams, is_script_mode, mo, params_form):
    if is_script_mode:
        model_params = ModelParams(
            **{k.replace("-", "_"): v for k, v in mo.cli_args().items()}
        )
    else:
        mo.stop(
            params_form.value is None,
            mo.md("Submit the form to start the run."),
        )
        model_params = ModelParams(**params_form.value)
    return (model_params,)


@app.cell
def _(generate_math_tasks, is_script_mode, mo, model_params, wandb):
    import random as _random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.optim import AdamW
    import re

    _random.seed(model_params.seed)
    torch.manual_seed(model_params.seed)

    login_error = None
    try:
        logged_in = wandb.login(verify=True)
    except Exception as exc:
        logged_in = False
        login_error = exc

    if logged_in is False:
        message = (
            "W&B authentication failed. Configure W&B locally or provide "
            "WANDB_API_KEY above, then submit the form again."
        )
        if is_script_mode:
            print(message)
            if login_error is not None:
                raise login_error
            raise RuntimeError(message)

        details = f"\n\nError: `{login_error}`" if login_error is not None else ""
        mo.stop(True, mo.md(f"{message}{details}"))

    run = wandb.init(
        project=model_params.wandb_project,
        name=model_params.run_name,
        config=model_params.model_dump(),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_params.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_params.model_id, torch_dtype="auto", device_map="auto"
    )

    train_tasks = generate_math_tasks(
        model_params.num_train, difficulty=model_params.difficulty
    )
    test_tasks = generate_math_tasks(
        model_params.num_test, difficulty=model_params.difficulty
    )

    template = "Input: {} \nOutput:"


    def run_eval(model, tasks, label="Evaluation"):
        correct = 0
        model.eval()
        print(f"\n--- {label} (Difficulty {model_params.difficulty}) ---")
        for expr, gold in tasks:
            prompt = template.format(expr)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False
                )
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                pred_text = response.split("Output:")[-1].strip()
                prediction = re.findall(r"-?\d+", pred_text)

                is_correct = (prediction[0] == gold) if prediction else False
                if is_correct:
                    correct += 1
                print(
                    f"Expr: {expr.ljust(18)} | Pred: {prediction[0] if prediction else 'N/A'} | "
                    f"Gold: {gold.ljust(3)} | {'✅' if is_correct else '❌'}"
                )

        acc = (correct / len(tasks)) * 100
        print(f"Accuracy: {acc}%")
        return acc


    # 1. Baseline
    base_acc = run_eval(model, test_tasks, "Baseline")
    wandb.log({"baseline_accuracy": base_acc})

    # 2. SSD Sampling
    print("\nSampling synthetic data...")
    synthetic_data = []
    for expr, _ in train_tasks:
        prompt = template.format(expr)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=True,
            temperature=model_params.temperature,
        )
        synthetic_data.append(
            tokenizer.decode(output[0], skip_special_tokens=True)
        )

    # 3. SSD Fine-tuning
    print("Training...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=model_params.learning_rate)
    for epoch in range(model_params.epochs):
        epoch_loss = 0.0
        for text in synthetic_data:
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
            loss = model(**inputs, labels=inputs.input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(synthetic_data)
        wandb.log({"epoch": epoch, "train_loss": avg_loss})
        print(f"Epoch {epoch + 1}/{model_params.epochs} - Loss: {avg_loss:.4f}")

    # 4. Final SSD Result
    ssd_acc = run_eval(model, test_tasks, "SSD Result")
    wandb.log({"ssd_accuracy": ssd_acc, "improvement": ssd_acc - base_acc})
    print(
        f"\nSummary for Difficulty {model_params.difficulty}: {base_acc}% -> {ssd_acc}%"
    )

    wandb.finish()
    return


@app.cell
def _():
    import random as _random


    def generate_math_tasks(num_tasks=10, difficulty=1):
        """
        difficulty 1: basic 3-term addition/subtraction (a + b + c)
        difficulty 2: nested brackets with two operations (a + (b * c))
        difficulty 3: double nested brackets ((a + b) * (c - d))
        """
        tasks = []
        for _ in range(num_tasks):
            a, b, c, d = [_random.randint(1, 10) for _ in range(4)]

            if difficulty == 1:
                ops = [_random.choice(["+", "-"]) for _ in range(2)]
                expr = f"{a} {ops[0]} {b} {ops[1]} {c}"

            elif difficulty == 2:
                op1 = _random.choice(["+", "-"])
                op2 = _random.choice(["*", "+"])
                if _random.random() > 0.5:
                    expr = f"({a} {op2} {b}) {op1} {c}"
                else:
                    expr = f"{a} {op1} ({b} {op2} {c})"

            elif difficulty == 3:
                op1 = _random.choice(["*", "+"])
                op2 = _random.choice(["+", "-"])
                op3 = _random.choice(["+", "-"])
                expr = f"({a} {op2} {b}) {op1} ({c} {op3} {d})"

            gold = str(eval(expr))
            tasks.append((expr, gold))
        return tasks

    return (generate_math_tasks,)


if __name__ == "__main__":
    app.run()
