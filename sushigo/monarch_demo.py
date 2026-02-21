# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.4.2",
#     "altair==6.0.0",
#     "polars==1.38.1",
#     "torchmonarch",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import polars as pl

    return alt, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parallel Sushi Go GA with Monarch Actors

    The [existing notebook](./demo.py) evolves Sushi Go players with a genetic algorithm,
    but fitness evaluation is sequential — each agent plays its games one at a time.
    That's the bottleneck: `pop_size × games_per_eval` games every generation.

    [**Monarch**](https://github.com/pytorch/monarch) lets us parallelize this by spawning
    **Evaluator actors** across worker processes. Each actor gets a batch of agents to
    evaluate independently, then we collect results and run the GA as before.

    ```
    ┌─────────────┐     chromosomes      ┌──────────────┐
    │             │ ──── batch 0 ──────▸ │  Evaluator 0 │ ──▸ [0.7, 0.5, ...]
    │   GA Loop   │ ──── batch 1 ──────▸ │  Evaluator 1 │ ──▸ [0.6, 0.8, ...]
    │             │ ──── batch 2 ──────▸ │  Evaluator 2 │ ──▸ [0.4, 0.7, ...]
    └─────────────┘     (parallel)       └──────────────┘     ──▸ combine
    ```
    """)
    return


@app.cell
def _(np):
    from pathlib import Path
    from monarch.actor import Actor, endpoint, this_host
    from game import SushiGoGame
    from agent import NeuralAgent, RandomAgent, CHROMOSOME_SIZE
    from ga import GeneticAlgorithm
    import game as _game_mod

    # Capture the directory containing game.py so worker processes can find it
    sushigo_dir = str(Path(_game_mod.__file__).resolve().parent)

    return (
        Actor, CHROMOSOME_SIZE, GeneticAlgorithm, NeuralAgent, RandomAgent,
        SushiGoGame, endpoint, sushigo_dir, this_host,
    )


@app.cell
def _(Actor, endpoint, np):
    class Evaluator(Actor):
        """Monarch actor that evaluates a batch of agent chromosomes."""

        def __init__(self, games_per_eval: int, seed: int, src_dir: str):
            import sys

            # Worker processes need the sushigo dir on sys.path
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)

            from game import SushiGoGame
            from agent import RandomAgent

            self.games_per_eval = games_per_eval
            self.rng = np.random.default_rng(seed)
            self.game = SushiGoGame(rng=self.rng)
            self.opponent = RandomAgent(rng=self.rng)

        @endpoint
        def evaluate_batch(self, chromosomes: list) -> list:
            """Score a list of weight vectors. Returns list of win-rate floats."""
            from agent import NeuralAgent

            results = []
            for weights in chromosomes:
                agent = NeuralAgent(weights=np.array(weights), rng=self.rng)
                wins = 0.0
                for _ in range(self.games_per_eval):
                    s1, s2 = self.game.play(agent, self.opponent)
                    if s1 > s2:
                        wins += 1.0
                    elif s1 == s2:
                        wins += 0.5
                results.append(wins / self.games_per_eval)
            return results

    return (Evaluator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings

    Tune the GA and parallelism parameters, then hit **Run**.
    The notebook will run the GA twice — once with Monarch workers, once sequentially —
    so you can compare wall-clock times.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_workers_slider = mo.ui.slider(1, 8, value=4, step=1, label="Workers")
    pop_size_slider = mo.ui.slider(20, 200, value=80, step=10, label="Population size")
    generations_slider = mo.ui.slider(10, 100, value=30, step=5, label="Generations")
    games_slider = mo.ui.slider(10, 50, value=20, step=5, label="Games per eval")
    seed_input = mo.ui.number(value=42, start=0, stop=9999, label="Seed")

    controls = mo.vstack([
        mo.hstack([n_workers_slider, pop_size_slider, generations_slider], justify="start"),
        mo.hstack([games_slider, seed_input], justify="start"),
    ])
    controls
    return (
        games_slider, generations_slider, n_workers_slider,
        pop_size_slider, seed_input,
    )


@app.cell
def _(generations_slider, mo, n_workers_slider, pop_size_slider):
    run_btn = mo.ui.run_button(label="Run GA (parallel + sequential)")
    mo.hstack([
        run_btn,
        mo.md(
            f"**{pop_size_slider.value}** agents × **{generations_slider.value}** generations "
            f"on **{n_workers_slider.value}** Monarch workers"
        ),
    ], justify="start")
    return (run_btn,)


@app.cell
async def _(
    CHROMOSOME_SIZE, Evaluator, GeneticAlgorithm, NeuralAgent,
    games_slider, generations_slider, mo, n_workers_slider, np,
    pop_size_slider, run_btn, seed_input, sushigo_dir, this_host,
):
    import time

    mo.stop(not run_btn.value, mo.md("*Click the button to start.*"))

    # --- Config ---
    n_workers = n_workers_slider.value
    pop_size = pop_size_slider.value
    n_gen = generations_slider.value
    games_per_eval = games_slider.value
    seed = seed_input.value

    # --- Parallel GA with Monarch ---
    rng = np.random.default_rng(seed)
    population = [NeuralAgent(rng=rng) for _ in range(pop_size)]

    ga = GeneticAlgorithm(
        pop_size=pop_size,
        elite_count=max(1, pop_size // 20),
        tournament_k=3,
        mutation_rate=0.1,
        mutation_sigma=0.3,
        games_per_eval=games_per_eval,
        seed=seed,
    )

    mesh = this_host().spawn_procs(per_host={"w": n_workers})
    evaluators = mesh.spawn(
        "eval", Evaluator, games_per_eval, seed, sushigo_dir,
    )

    par_history = {"generation": [], "best": [], "mean": [], "worst": []}
    t0 = time.perf_counter()

    for _gen in range(n_gen):
        # Serialize chromosomes as plain lists for Monarch transport
        _chroms = [agent.chromosome.tolist() for agent in population]

        # Split into batches and dispatch to workers in parallel
        _batches = np.array_split(range(len(_chroms)), n_workers)
        _futures = []
        for _i, _idx_batch in enumerate(_batches):
            _batch = [_chroms[j] for j in _idx_batch]
            _f = evaluators.slice(w=_i).evaluate_batch.call_one(_batch)
            _futures.append(_f)

        # Collect results
        _fitness_parts = [f.get() for f in _futures]
        _fitness = np.concatenate(_fitness_parts)

        par_history["generation"].append(_gen)
        par_history["best"].append(float(np.max(_fitness)))
        par_history["mean"].append(float(np.mean(_fitness)))
        par_history["worst"].append(float(np.min(_fitness)))

        population = ga.evolve_step(population, _fitness)

    par_time = time.perf_counter() - t0
    mesh.stop().get()

    # --- Sequential baseline ---
    t0 = time.perf_counter()
    seq_ga = GeneticAlgorithm(
        pop_size=pop_size,
        elite_count=max(1, pop_size // 20),
        tournament_k=3,
        mutation_rate=0.1,
        mutation_sigma=0.3,
        games_per_eval=games_per_eval,
        seed=seed,
    )
    seq_result = seq_ga.run(generations=n_gen)
    seq_time = time.perf_counter() - t0
    seq_history = seq_result["history"]

    return par_history, par_time, seq_history, seq_time


@app.cell(hide_code=True)
def _(mo, n_workers_slider, par_time, seq_time):
    speedup = seq_time / par_time if par_time > 0 else float("inf")
    mo.md(f"""
    ## Timing

    | | Wall time | Speedup |
    |---|---|---|
    | **Sequential** | {seq_time:.1f}s | 1.0× |
    | **Monarch ({n_workers_slider.value} workers)** | {par_time:.1f}s | **{speedup:.2f}×** |
    """)
    return (speedup,)


@app.cell(hide_code=True)
def _(alt, par_time, pl, seq_time, speedup):
    timing_df = pl.DataFrame({
        "method": ["Sequential", "Monarch"],
        "seconds": [seq_time, par_time],
        "label": [f"{seq_time:.1f}s", f"{par_time:.1f}s ({speedup:.1f}×)"],
    })

    timing_chart = (
        alt.Chart(timing_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("method:N", title=None, sort=["Sequential", "Monarch"]),
            y=alt.Y("seconds:Q", title="Wall time (s)"),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=["Sequential", "Monarch"],
                    range=["#95a5a6", "#e67e22"],
                ),
                legend=None,
            ),
        )
    )
    timing_text = (
        alt.Chart(timing_df)
        .mark_text(dy=-10, fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("method:N", sort=["Sequential", "Monarch"]),
            y=alt.Y("seconds:Q"),
            text="label:N",
        )
    )
    (timing_chart + timing_text).properties(width=300, height=300, title="GA Wall Time")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fitness Over Generations
    """)
    return


@app.cell(hide_code=True)
def _(alt, par_history, pl, seq_history):
    def _make_df(history, method):
        return pl.DataFrame({
            "generation": history["generation"],
            "best": history["best"],
            "mean": history["mean"],
            "worst": history["worst"],
        }).unpivot(
            index="generation",
            on=["best", "mean", "worst"],
            variable_name="metric",
            value_name="fitness",
        ).with_columns(pl.lit(method).alias("method"))

    combined = pl.concat([
        _make_df(par_history, "Monarch"),
        _make_df(seq_history, "Sequential"),
    ])

    fitness_chart = (
        alt.Chart(combined)
        .mark_line()
        .encode(
            x=alt.X("generation:Q", title="Generation"),
            y=alt.Y("fitness:Q", title="Win rate vs random"),
            color=alt.Color("metric:N", title="Metric"),
            strokeDash=alt.StrokeDash("method:N", title="Method"),
        )
        .properties(width=600, height=350)
    )
    fitness_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How It Works

    1. **Spawn a process mesh** with `this_host().spawn_procs(per_host={"w": n_workers})` —
       creates `n_workers` OS processes on the local machine.
    2. **Spawn `Evaluator` actors** on the mesh — each process gets its own game engine
       and RNG, ready to play Sushi Go games independently.
    3. **Each generation**, the GA serializes all chromosomes as plain Python lists
       (Monarch pickles them across the process boundary), splits them into batches,
       and dispatches one batch per worker via `evaluators.slice(w=i).evaluate_batch.call_one(batch)`.
    4. **Futures resolve in parallel** — the workers play games simultaneously.
    5. Results are concatenated into a single fitness array and the standard GA
       operators (elitism, tournament selection, crossover, mutation) produce the next generation.

    The GA logic is identical to the sequential version — only the fitness evaluation is distributed.
    """)
    return


if __name__ == "__main__":
    app.run()
