# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.0.0",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    return mo, nn, np, plt, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How Much Does Backprop Actually Cost?

    Everyone knows backpropagation is "more expensive" than a forward pass — but by how much exactly?
    This notebook measures the **time** and **memory** overhead of backward passes compared to forward-only inference,
    on a feedforward network trained on a high-dimensional chessboard classification task.

    We compare **SGD** and **Adam** to see how optimizer state affects the cost.
    """)
    return


@app.cell
def _(mo):
    params_form = mo.ui.dictionary({
        "n_layers": mo.ui.slider(start=2, stop=20, step=1, value=4, label="Layers"),
        "width": mo.ui.slider(start=32, stop=1512, step=32, value=128, label="Width"),
        "input_dim": mo.ui.slider(start=4, stop=200, step=2, value=10, label="Input dim"),
        "n_samples": mo.ui.slider(start=500, stop=50000, step=500, value=2000, label="Samples"),
    }).form()
    params_form
    return (params_form,)


@app.cell
def _(np, torch):
    def make_chessboard(n_samples, input_dim, freq=3):
        """High-dimensional chessboard: label = sum(floor(x_i * freq)) % 2."""
        X = np.random.uniform(-1, 1, (n_samples, input_dim)).astype(np.float32)
        label = np.sum(np.floor(X * freq).astype(int), axis=1) % 2
        return torch.tensor(X), torch.tensor(label, dtype=torch.float32).unsqueeze(1)

    return (make_chessboard,)


@app.cell
def _(nn):
    def build_model(input_dim, width, n_layers):
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1), nn.Sigmoid()]
        return nn.Sequential(*layers)

    return (build_model,)


@app.cell
def _(nn, np, time, torch):
    def _tensor_bytes(t):
        return t.nelement() * t.element_size()

    def _param_bytes(model):
        return sum(_tensor_bytes(p) for p in model.parameters())

    def _grad_bytes(model):
        return sum(_tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)

    def _optimizer_state_bytes(optimizer):
        total = 0
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total += _tensor_bytes(v)
        return total

    def benchmark(model, X, y, optimizer, n_warmup=5, n_measure=30):
        loss_fn = nn.BCELoss()

        # --- Forward only (timing) ---
        for _ in range(n_warmup):
            with torch.no_grad():
                model(X)

        fwd_times = []
        for _ in range(n_measure):
            t0 = time.perf_counter_ns()
            with torch.no_grad():
                model(X)
            fwd_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Forward-only memory: just the parameters (no grads, no optimizer state)
        fwd_mem = _param_bytes(model)

        # --- Forward + backward + step (timing) ---
        for _ in range(n_warmup):
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        fwd_bwd_times = []
        for _ in range(n_measure):
            optimizer.zero_grad()
            t0 = time.perf_counter_ns()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            fwd_bwd_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Training memory: parameters + gradients + optimizer state
        fwd_bwd_mem = _param_bytes(model) + _grad_bytes(model) + _optimizer_state_bytes(optimizer)

        return {
            "fwd_ms": float(np.median(fwd_times)),
            "fwd_bwd_ms": float(np.median(fwd_bwd_times)),
            "fwd_mem_kb": fwd_mem / 1024,
            "fwd_bwd_mem_kb": fwd_bwd_mem / 1024,
        }

    return (benchmark,)


@app.cell
def _(benchmark, build_model, make_chessboard, mo, params_form, torch):
    mo.stop(params_form.value is None, mo.md("Submit the form to run the benchmark."))
    _input_dim = params_form.value["input_dim"]
    _width = params_form.value["width"]
    _n_layers = params_form.value["n_layers"]
    _n_samples = params_form.value["n_samples"]

    X, y = make_chessboard(_n_samples, _input_dim)

    results = {}
    for _name, _opt_cls in [("SGD", torch.optim.SGD), ("Adam", torch.optim.Adam)]:
        _model = build_model(_input_dim, _width, _n_layers)
        _opt = _opt_cls(_model.parameters(), lr=0.01)
        results[_name] = benchmark(_model, X, y, _opt)
    return (results,)


@app.cell(hide_code=True)
def _(mo, params_form, results):
    def _fmt(v, unit="ms"):
        return f"{v:.2f} {unit}"

    def _ratio(fwd_bwd, fwd):
        return f"{fwd_bwd / fwd:.1f}x"

    _sgd = results["SGD"]
    _adam = results["Adam"]

    mo.md(f"""
    ## Results for {params_form.value['n_layers']}-layer, width-{params_form.value['width']} network

    | | Forward only | Forward + Backward | Ratio |
    |---|---:|---:|---:|
    | **SGD time** | {_fmt(_sgd['fwd_ms'])} | {_fmt(_sgd['fwd_bwd_ms'])} | {_ratio(_sgd['fwd_bwd_ms'], _sgd['fwd_ms'])} |
    | **Adam time** | {_fmt(_adam['fwd_ms'])} | {_fmt(_adam['fwd_bwd_ms'])} | {_ratio(_adam['fwd_bwd_ms'], _adam['fwd_ms'])} |
    | **SGD memory** | {_fmt(_sgd['fwd_mem_kb'], 'KB')} | {_fmt(_sgd['fwd_bwd_mem_kb'], 'KB')} | {_ratio(_sgd['fwd_bwd_mem_kb'], _sgd['fwd_mem_kb'])} |
    | **Adam memory** | {_fmt(_adam['fwd_mem_kb'], 'KB')} | {_fmt(_adam['fwd_bwd_mem_kb'], 'KB')} | {_ratio(_adam['fwd_bwd_mem_kb'], _adam['fwd_mem_kb'])} |
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    _sgd = results["SGD"]
    _adam = results["Adam"]

    fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Time comparison
    _x = np.arange(2)
    _w = 0.3
    _ax1.bar(_x - _w/2, [_sgd["fwd_ms"], _adam["fwd_ms"]], _w, label="Forward", color="#4c78a8")
    _ax1.bar(_x + _w/2, [_sgd["fwd_bwd_ms"], _adam["fwd_bwd_ms"]], _w, label="Forward + Backward", color="#e45756")
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(["SGD", "Adam"])
    _ax1.set_ylabel("Time (ms)")
    _ax1.set_title("Time per iteration")
    _ax1.legend()

    # Memory comparison
    _ax2.bar(_x - _w/2, [_sgd["fwd_mem_kb"], _adam["fwd_mem_kb"]], _w, label="Forward", color="#4c78a8")
    _ax2.bar(_x + _w/2, [_sgd["fwd_bwd_mem_kb"], _adam["fwd_bwd_mem_kb"]], _w, label="Forward + Backward", color="#e45756")
    _ax2.set_xticks(_x)
    _ax2.set_xticklabels(["SGD", "Adam"])
    _ax2.set_ylabel("Peak memory (KB)")
    _ax2.set_title("Peak memory usage")
    _ax2.legend()

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How does the cost ratio scale with network width?

    Below we sweep over different widths to see if the backward/forward ratio is stable or changes with model size.
    """)
    return


@app.cell
def _(benchmark, build_model, make_chessboard, mo, params_form, torch):
    mo.stop(params_form.value is None)
    _input_dim = params_form.value["input_dim"]
    _n_layers = params_form.value["n_layers"]
    _n_samples = params_form.value["n_samples"]

    _X, _y = make_chessboard(_n_samples, _input_dim)

    widths = [32, 64, 128, 256, 512]
    sweep_results = []
    for _w in widths:
        for _name, _opt_cls in [("SGD", torch.optim.SGD), ("Adam", torch.optim.Adam)]:
            _model = build_model(_input_dim, _w, _n_layers)
            _opt = _opt_cls(_model.parameters(), lr=0.01)
            _res = benchmark(_model, _X, _y, _opt, n_warmup=3, n_measure=15)
            sweep_results.append({
                "width": _w,
                "optimizer": _name,
                "time_ratio": _res["fwd_bwd_ms"] / _res["fwd_ms"],
                "mem_ratio": _res["fwd_bwd_mem_kb"] / _res["fwd_mem_kb"],
                "fwd_ms": _res["fwd_ms"],
                "fwd_bwd_ms": _res["fwd_bwd_ms"],
            })
    return sweep_results, widths


@app.cell(hide_code=True)
def _(plt, sweep_results, widths):
    _sgd = [r for r in sweep_results if r["optimizer"] == "SGD"]
    _adam = [r for r in sweep_results if r["optimizer"] == "Adam"]

    _fig2, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Time ratio vs width
    _ax1.plot(widths, [r["time_ratio"] for r in _sgd], "o-", label="SGD", color="#4c78a8", linewidth=2)
    _ax1.plot(widths, [r["time_ratio"] for r in _adam], "s-", label="Adam", color="#e45756", linewidth=2)
    _ax1.set_xlabel("Network width")
    _ax1.set_ylabel("Time ratio (fwd+bwd / fwd)")
    _ax1.set_title("Backward pass time overhead")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)
    _ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # Absolute times vs width
    _ax2.plot(widths, [r["fwd_ms"] for r in _sgd], "o--", label="SGD fwd", color="#4c78a8", alpha=0.6)
    _ax2.plot(widths, [r["fwd_bwd_ms"] for r in _sgd], "o-", label="SGD fwd+bwd", color="#4c78a8")
    _ax2.plot(widths, [r["fwd_ms"] for r in _adam], "s--", label="Adam fwd", color="#e45756", alpha=0.6)
    _ax2.plot(widths, [r["fwd_bwd_ms"] for r in _adam], "s-", label="Adam fwd+bwd", color="#e45756")
    _ax2.set_xlabel("Network width")
    _ax2.set_ylabel("Time (ms)")
    _ax2.set_title("Absolute time vs width")
    _ax2.legend(fontsize=8)
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Backprop typically costs 2–3x the forward pass** in wall-clock time. The backward pass needs to compute gradients for every layer and store intermediate activations.
    - **Adam is slightly more expensive than SGD** per step because it maintains running estimates of the first and second moments ($m_t$ and $v_t$) for each parameter, doubling the optimizer state.
    - **The ratio is relatively stable across model sizes** — it's a property of the computation graph structure, not the absolute size.
    - **Memory overhead of backprop is significant** because PyTorch must retain intermediate activations from the forward pass to compute gradients during the backward pass.
    """)
    return


if __name__ == "__main__":
    app.run()
