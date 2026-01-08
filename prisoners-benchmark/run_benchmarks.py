# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "max",
#     "numba",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import time
    import sys
    import subprocess
    import shutil
    import os
    import numpy as np
    return mo, np, os, sys, time


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Prisoner Simulation Benchmark

    Comparing different implementations of the prisoner simulation:
    - **Pure Python** - baseline with Python lists
    - **NumPy** - using NumPy arrays
    - **Numba** - JIT-compiled with `@njit`
    - **Rust** - native extension via PyO3
    - **Mojo** - Modular's high-performance language
    """)
    return


@app.cell
def _():
    from bench_python import simulate_python, simulate_numpy
    from bench_numba import simulate_numba, warmup as warmup_numba
    return simulate_numba, simulate_numpy, simulate_python, warmup_numba


@app.cell
def _():
    # Try to import Rust implementation
    try:
        from bench_rust import simulate_rust
        RUST_AVAILABLE = True
    except ImportError:
        simulate_rust = None
        RUST_AVAILABLE = False
    return RUST_AVAILABLE, simulate_rust


@app.cell
def _(os, sys, time):
    # Mojo implementation - direct import via mojo.importer
    import mojo.importer

    # Add the prisoners-benchmark directory to sys.path if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from bench_mojo import simulate_mojo as simulate_mojo_direct

    def simulate_mojo_wrapper(n_prisoners: int, n_sims: int) -> tuple[float, list[int]]:
        """Run Mojo simulation. Returns (elapsed_time, results)."""
        start = time.perf_counter()
        results = simulate_mojo_direct(n_prisoners, n_sims)
        elapsed = time.perf_counter() - start
        return elapsed, list(results)
    return (simulate_mojo_wrapper,)


@app.cell(hide_code=True)
def _(RUST_AVAILABLE, mo):
    _notes = []
    if not RUST_AVAILABLE:
        _notes.append("**Rust** not available. Build with: `cd bench_rust && maturin develop --release`")

    mo.md("> " + "\n>\n> ".join(_notes)) if _notes else None
    return


@app.cell
def _(np, time):
    def benchmark(func, n_prisoners: int, n_sims: int, name: str) -> tuple[float, list]:
        """Benchmark a simulation function. Returns (elapsed_time, results)."""
        start = time.perf_counter()
        results = func(n_prisoners, n_sims)
        elapsed = time.perf_counter() - start
        return elapsed, results


    def verify_distribution(results1: list, results2: list, name1: str, name2: str, tolerance: float = 0.05) -> bool:
        """Check that two result distributions are similar (statistical sanity check)."""
        mean1, mean2 = np.mean(results1), np.mean(results2)
        print(mean1, mean2)
        relative_diff = abs(mean1 - mean2) / max(mean1, mean2)
        return relative_diff < tolerance
    return benchmark, verify_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    n_sims_input = mo.ui.slider(
        value=10_000, start=1_000, stop=100_000, step=1_000, label="Number of simulations"
    )
    n_sims_input
    return (n_sims_input,)


@app.cell
def _(n_sims_input):
    N_PRISONERS = 100
    N_SIMS = n_sims_input.value
    return N_PRISONERS, N_SIMS


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Run Benchmark
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.button(label="🚀 Run Benchmarks")
    run_button
    return (run_button,)


@app.cell
def _(
    N_PRISONERS,
    N_SIMS,
    RUST_AVAILABLE,
    benchmark,
    run_button,
    simulate_mojo_wrapper,
    simulate_numba,
    simulate_numpy,
    simulate_python,
    simulate_rust,
    warmup_numba,
):
    # Trigger on button click
    run_button

    # Warm up Numba
    warmup_numba()

    # Run benchmarks
    results_dict = {}

    # Pure Python
    elapsed, results = benchmark(simulate_python, N_PRISONERS, N_SIMS, "Pure Python")
    results_dict["Pure Python"] = (elapsed, results)

    # NumPy
    elapsed, results = benchmark(simulate_numpy, N_PRISONERS, N_SIMS, "NumPy")
    results_dict["NumPy"] = (elapsed, results)

    # Numba
    elapsed, results = benchmark(simulate_numba, N_PRISONERS, N_SIMS, "Numba")
    results_dict["Numba"] = (elapsed, results)

    # Rust (if available)
    if RUST_AVAILABLE:
        elapsed, results = benchmark(simulate_rust, N_PRISONERS, N_SIMS, "Rust")
        results_dict["Rust"] = (elapsed, results)

    # Mojo
    elapsed, results = simulate_mojo_wrapper(N_PRISONERS, N_SIMS)
    results_dict["Mojo"] = (elapsed, results)

    {k: v[0] for k, v in results_dict.items()}
    return (results_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Results
    """)
    return


@app.cell
def _(N_PRISONERS, N_SIMS, mo, results_dict):
    # Calculate speedups
    _baseline = results_dict["Pure Python"][0]

    _rows = []
    for _name, (_elapsed, _) in results_dict.items():
        _speedup = _baseline / _elapsed
        _rows.append({
            "Implementation": _name,
            "Time (s)": f"{_elapsed:.3f}",
            "Sims/sec": f"{N_SIMS/_elapsed:,.0f}",
            "Speedup": f"{_speedup:.1f}x",
        })

    import pandas as pd
    _results_df = pd.DataFrame(_rows)

    mo.md(f"""
    **Parameters:** {N_PRISONERS} prisoners, {N_SIMS:,} simulations

    {_results_df.to_markdown(index=False)}
    """)

    print(_results_df.to_markdown(index=False))
    return


@app.cell
def _():
    _results_df.to_markdown(index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Distribution Verification
    """)
    return


@app.cell
def _(results_dict):
    results_dict
    return


@app.cell
def _(results_dict, verify_distribution):
    _baseline_results = results_dict["Pure Python"][1]
    _verification = []
    for _name, (_, _results) in results_dict.items():
        if _name == "Pure Python":
            continue
        _ok = verify_distribution(_baseline_results, _results, "Pure Python", _name)
        _status = "✓" if _ok else "✗"
        _verification.append(f"- {_name}: {_status}")

    _all_ok = all("✓" in v for v in _verification)
    _summary = "All distributions match!" if _all_ok else "Warning: Some distributions differ"
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
