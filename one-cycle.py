# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "altair==5.5.0",
#     "hastyplot==0.4.1",
#     "marimo",
#     "matplotlib==3.10.7",
#     "mnist1d==0.0.2.post1",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "torch==2.12.0",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return alt, device, mo, nn, np, pd, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The cycle learning-rate policy

    Let's see why the learning rate matters. We'll fit the simplest possible model — a line $y = x\,b_1 + b_0$ — with plain gradient descent and watch what happens as you crank the learning rate up.
    """)
    return


@app.cell
def _(np):
    # Synthetic line: y = x * b1_true + b0_true + noise
    b0_true, b1_true = 1.0, 2.0
    rng = np.random.default_rng(0)
    x_data = np.linspace(-2, 2, 30)
    y_data = b1_true * x_data + b0_true + rng.normal(0, 0.5, x_data.shape)

    def mse(b0, b1):
        pred = b1 * x_data[:, None, None] + b0
        return ((pred - y_data[:, None, None]) ** 2).mean(axis=0)

    def grad(b0, b1):
        pred = b1 * x_data + b0
        err = pred - y_data
        return 2 * err.mean(), 2 * (err * x_data).mean()

    return b0_true, b1_true, grad, mse


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        0.01, 1.2, value=0.1, step=0.01, label="learning rate", show_value=True
    )
    steps_slider = mo.ui.slider(
        2, 60, value=20, step=1, label="GD steps", show_value=True
    )
    elev_slider = mo.ui.slider(
        0, 90, value=35, step=1, label="elevation", show_value=True
    )
    azim_slider = mo.ui.slider(
        -180, 180, value=-60, step=5, label="azimuth", show_value=True
    )
    mo.vstack(
        [
            mo.hstack([lr_slider, steps_slider], justify="start"),
            mo.hstack([elev_slider, azim_slider], justify="start"),
        ]
    )
    return azim_slider, elev_slider, lr_slider, steps_slider


@app.cell
def _(grad, lr_slider, np, steps_slider):
    # Plain gradient descent from a fixed bad start.
    b0, b1 = -4.0, -3.0
    traj = [(b0, b1)]
    for _ in range(steps_slider.value):
        g0, g1 = grad(b0, b1)
        b0 = b0 - lr_slider.value * g0
        b1 = b1 - lr_slider.value * g1
        b0 = float(np.clip(b0, -8, 8))
        b1 = float(np.clip(b1, -8, 8))
        traj.append((b0, b1))
    traj = np.array(traj)
    return (traj,)


@app.cell(hide_code=True)
def _(azim_slider, b0_true, b1_true, elev_slider, mse, np, plt, traj):
    # 3D loss surface + descent trajectory.
    gb0 = np.linspace(-8, 8, 80)
    gb1 = np.linspace(-8, 8, 80)
    B0, B1 = np.meshgrid(gb0, gb1)
    Z = mse(B0, B1)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(B0, B1, Z, cmap="viridis", alpha=0.6, linewidth=0)

    tb0, tb1 = traj[:, 0], traj[:, 1]
    tz = np.array([mse(np.array([[a]]), np.array([[b]]))[0, 0] for a, b in traj])
    ax.plot(tb0, tb1, tz, color="red", marker="o", markersize=3, linewidth=1.5)
    ax.scatter([b0_true], [b1_true], [0.0], color="black", s=40)

    ax.set_xlabel("b0 (intercept)")
    ax.set_ylabel("b1 (slope)")
    ax.set_zlabel("MSE")
    ax.set_title("Gradient descent on the loss bowl")
    ax.view_init(elev=elev_slider.value, azim=azim_slider.value)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Low learning rate → the red path crawls and never reaches the bowl's
    floor in the budget.
    Too high → it overshoots and bounces up the walls
    (or diverges).

    There's a band that converges. A *schedule*
    lets you exploit a hidgh rate while still keeping the risk of an overshoot low.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Schedules matter: the 1cycle policy

    The **1cycle** policy (Leslie Smith) runs a single cycle: the learning rate
    ramps *up* to a peak, then back *down* well below where it started. The high
    middle phase moves fast across the loss landscape; the cool-down at the end
    settles into a minimum. Smith calls the speed-up *super-convergence*.

    Below we train the same Conv1d net on MNIST-1D several ways with SGD+momentum,
    all sharing the **same peak learning rate** and epoch budget, and compare test
    accuracy per epoch:

    - **constant** — sits at the peak LR the whole time
    - **step-decay** — starts at the peak LR, halved at 50% and 75%
    - **1cycle** — ramps *up* to the peak then anneals back down
    - **2cycle / 3cycle** — the same up-down ramp repeated 2× / 3× (warm restarts)

    Push `max_lr` high enough and the constant run becomes unstable while the
    cyclic schedules ride the same rate to a better minimum — that's
    super-convergence. The LR curves themselves are plotted below the accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(SCHEDULES, X_train, alt, epochs_slider, max_lr_slider, pd, schedule_lr):
    from hastyplot import qplot 

    lr_n_epochs = epochs_slider.value
    lr_max = max_lr_slider.value
    lr_bs = 128
    lr_spe = (X_train.shape[0] + lr_bs - 1) // lr_bs
    lr_total = lr_n_epochs * lr_spe

    lr_records = []
    for sched_name in SCHEDULES:
        for step in range(lr_total):
            lr_records.append(
                {
                    "epoch": (step + 1) / lr_spe,
                    "lr": schedule_lr(sched_name, step, lr_total, lr_max),
                    "schedule": sched_name,
                }
            )
    lr_df = pd.DataFrame(lr_records)

    lr_chart = (
        alt.Chart(lr_df)
        .mark_line()
        .encode(
            x=alt.X("epoch:Q", title="epoch"),
            y=alt.Y("lr:Q", title="learning rate"),
            color=alt.Color("schedule:N", title="schedule"),
        )
        .properties(width=560, height=160, title="Learning rate schedule")
    )

    qplot(
        lr_df, 
        "epoch", 
        "lr", 
        width=460, 
        height=160,
        color="schedule", 
        mark="line", 
        title="Learning rates", 
        subtitle="How much might this influence convergence?")
    return (lr_chart,)


@app.cell
def _(np):
    from mnist1d.data import make_dataset, get_dataset_args

    args = get_dataset_args()
    data = make_dataset(args)
    X_train, y_train = data["x"], data["y"]
    X_test, y_test = data["x_test"], data["y_test"]
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(np):
    def schedule_lr(name, step, total_steps, max_lr):
        # One source of truth for the LR at a given step, shared by training and
        # the schedule plot so they can never drift apart.
        if name == "constant":
            return max_lr
        if name == "step-decay":
            frac = step / total_steps
            factor = 1.0
            for milestone in (0.5, 0.75):
                if frac >= milestone:
                    factor *= 0.5
            return max_lr * factor
        # "Ncycle": N back-to-back 1cycle ramps (cosine, like torch OneCycleLR).
        n_cycles = int(name[0])
        init_lr = max_lr / 25
        min_lr = init_lr / 1e4
        pct_up = 0.3
        cycle_frac = (step / total_steps * n_cycles) % 1.0
        if cycle_frac < pct_up:
            p = cycle_frac / pct_up
            return max_lr + (init_lr - max_lr) / 2 * (1 + np.cos(np.pi * p))
        p = (cycle_frac - pct_up) / (1 - pct_up)
        return min_lr + (max_lr - min_lr) / 2 * (1 + np.cos(np.pi * p))

    return (schedule_lr,)


@app.cell
def _(mo, np):
    lr_steps = [round(float(v), 4) for v in np.logspace(np.log10(0.001), np.log10(1.0), 25)]
    max_lr_slider = mo.ui.slider(
        steps=lr_steps,
        value=min(lr_steps, key=lambda v: abs(v - 0.3)),
        label="max learning rate",
        show_value=True,
    )
    epochs_slider = mo.ui.slider(
        10, 100, value=40, step=5, label="epochs", show_value=True
    )
    sigmoid_toggle = mo.ui.switch(value=False, label="use sigmoid (off = ReLU)")
    run_button = mo.ui.run_button(label="Train all schedules")

    mo.vstack([max_lr_slider, epochs_slider, sigmoid_toggle, run_button])
    return epochs_slider, max_lr_slider, run_button, sigmoid_toggle


@app.cell
def _():
    SCHEDULES = ["constant", "step-decay", "1cycle", "2cycle", "3cycle"]
    return (SCHEDULES,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(
    SCHEDULES,
    X_test,
    X_train,
    device,
    epochs_slider,
    max_lr_slider,
    mo,
    nn,
    pd,
    run_button,
    schedule_lr,
    sigmoid_toggle,
    torch,
    y_test,
    y_train,
):
    mo.stop(
        not run_button.value,
        mo.md("*Set the knobs, then click **Train all schedules**.*"),
    )

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)

    n_epochs = epochs_slider.value
    max_lr = max_lr_slider.value
    batch_size = 128
    n_train = Xtr.shape[0]
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    total_steps = n_epochs * steps_per_epoch

    act_layer = nn.Sigmoid if sigmoid_toggle.value else nn.ReLU

    def build_model():
        # Conv1d frontend: MNIST-1D has translation structure an MLP throws away.
        torch.manual_seed(0)
        return nn.Sequential(
            nn.Unflatten(1, (1, 40)),
            nn.Conv1d(1, 32, 3, padding=1),
            act_layer(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            act_layer(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * 10, 10),
        ).to(device)

    def train(name):
        torch.manual_seed(0)
        model = build_model()
        # LR is set by hand each step from the shared schedule_lr function.
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        records = []
        global_step = 0
        for ep in range(n_epochs):
            perm = torch.randperm(n_train, device=device)
            model.train()
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                lr = schedule_lr(name, global_step, total_steps, max_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad()
                loss_fn(model(Xtr[idx]), ytr[idx]).backward()
                optimizer.step()
                global_step += 1
            model.eval()
            with torch.no_grad():
                tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
                te_acc = (model(Xte).argmax(1) == yte).float().mean().item()
            records.append(
                {"epoch": ep + 1, "split": "train", "accuracy": tr_acc, "schedule": name}
            )
            records.append(
                {"epoch": ep + 1, "split": "test", "accuracy": te_acc, "schedule": name}
            )
        return records

    all_records = []
    with mo.status.progress_bar(total=len(SCHEDULES)) as bar:
        for sched in SCHEDULES:
            all_records.extend(train(sched))
            bar.update()
    bench_df = pd.DataFrame(all_records)
    return bench_df, build_model


@app.cell(hide_code=True)
def _(alt, bench_df, lr_chart):
    acc_base = alt.Chart(bench_df).mark_line().encode(
        x=alt.X("epoch:Q", title=None),
        y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("schedule:N", title="schedule"),
    )

    train_chart = acc_base.transform_filter(
        alt.datum.split == "train"
    ).properties(width=560, height=200, title="Train accuracy per epoch")

    test_chart = acc_base.transform_filter(
        alt.datum.split == "test"
    ).properties(width=560, height=200, title="Test accuracy per epoch")

    alt.vconcat(train_chart, test_chart, lr_chart).resolve_scale(
        x="shared", color="shared"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Intersting lesson here. You can get into a dead zone (probably because all the ReLU point to zero) that you never ever really recover from. Notice how red suffers first? Then Orange. Then Blue?
    """)
    return


@app.cell
def _(build_model):
    build_model()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
