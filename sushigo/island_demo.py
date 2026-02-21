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

    # CLI args override UI defaults: uv run island_demo.py -- --islands 4 --pop 120 ...
    args = mo.cli_args()

    return alt, args, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Island Model GA: Monarch Actors vs Multiprocessing

    A standard parallel GA just farms out **fitness evaluation** — any process pool can do that
    (see the [parallel evaluator notebook](./monarch_demo.py)).

    The **Island Model** is where actors truly shine. Each island is a **stateful worker** that:

    - **owns** its own population
    - **evolves** independently for several generations
    - periodically **migrates** its best agents to neighbouring islands

    With **multiprocessing**, this requires Queues, a command protocol, and manual process
    lifecycle management. With **Monarch actors**, it's just endpoint calls on stateful objects.

    ```
    ┌──────────┐  migrate  ┌──────────┐  migrate  ┌──────────┐
    │ Island 0 │ ────────▸ │ Island 1 │ ────────▸ │ Island 2 │
    │ pop=30   │ ◂──────── │ pop=30   │ ◂──────── │ pop=30   │
    │ evolve() │           │ evolve() │           │ evolve() │
    └──────────┘           └──────────┘           └──────────┘
         │                      │                      │
         └──────────── migrate ─┘──────── migrate ─────┘
                        (ring topology)
    ```
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from monarch.actor import Actor, endpoint, this_host
    from game import SushiGoGame
    from agent import NeuralAgent, RandomAgent, CHROMOSOME_SIZE
    from ga import GeneticAlgorithm
    import game as _game_mod

    sushigo_dir = str(Path(_game_mod.__file__).resolve().parent)

    return (
        Actor, CHROMOSOME_SIZE, GeneticAlgorithm, NeuralAgent, Path,
        RandomAgent, SushiGoGame, endpoint, sushigo_dir, this_host,
    )


@app.cell
def _(Actor, endpoint):
    class IslandActor(Actor):
        """Monarch actor: a self-contained GA island with its own population."""

        def __init__(self, island_id, pop_size, games_per_eval, elite_count,
                     tournament_k, mutation_rate, mutation_sigma, seed, src_dir):
            import sys
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)

            from ga import GeneticAlgorithm
            import numpy as _np

            self._np = _np
            self.island_id = island_id
            self.ga = GeneticAlgorithm(
                pop_size=pop_size, elite_count=elite_count,
                tournament_k=tournament_k, mutation_rate=mutation_rate,
                mutation_sigma=mutation_sigma, games_per_eval=games_per_eval,
                seed=seed + island_id,
            )
            self.population = self.ga._make_population()
            self.fitness = None

        @endpoint
        def evolve(self, n_gens):
            stats = []
            for _ in range(n_gens):
                self.fitness = self.ga.evaluate_fitness(self.population)
                stats.append({
                    "best": float(self._np.max(self.fitness)),
                    "mean": float(self._np.mean(self.fitness)),
                    "worst": float(self._np.min(self.fitness)),
                })
                self.population = self.ga.evolve_step(self.population, self.fitness)
            return stats

        @endpoint
        def get_best(self, k):
            if self.fitness is None:
                self.fitness = self.ga.evaluate_fitness(self.population)
            top_idx = self._np.argsort(self.fitness)[-k:]
            return [self.population[i].chromosome.tolist() for i in top_idx]

        @endpoint
        def receive_migrants(self, chromosomes):
            from agent import NeuralAgent
            if self.fitness is None:
                self.fitness = self.ga.evaluate_fitness(self.population)
            worst_idx = self._np.argsort(self.fitness)[:len(chromosomes)]
            for idx, chrom in zip(worst_idx, chromosomes):
                self.population[idx] = NeuralAgent(
                    weights=self._np.array(chrom), rng=self.ga.rng
                )
            self.fitness = None

        @endpoint
        def get_global_best(self):
            if self.fitness is None:
                self.fitness = self.ga.evaluate_fitness(self.population)
            best_idx = int(self._np.argmax(self.fitness))
            return {
                "chromosome": self.population[best_idx].chromosome.tolist(),
                "fitness": float(self.fitness[best_idx]),
            }

    return (IslandActor,)


@app.cell(hide_code=True)
def _(args, mo):
    # CLI args override slider defaults:
    #   uv run island_demo.py -- --islands 4 --pop 120 --gens 50 \
    #       --interval 10 --migrants 2 --games 30 --seed 42
    _def = lambda key, fallback: int(args.get(key, fallback))

    n_islands_slider = mo.ui.slider(2, 8, value=_def("islands", 4), step=1, label="Islands")
    total_pop_slider = mo.ui.slider(40, 400, value=_def("pop", 80), step=20, label="Total population")
    generations_slider = mo.ui.slider(20, 200, value=_def("gens", 40), step=10, label="Generations")
    migration_interval_slider = mo.ui.slider(
        2, 20, value=_def("interval", 5), step=1, label="Migration interval (gens)"
    )
    n_migrants_slider = mo.ui.slider(1, 5, value=_def("migrants", 2), step=1, label="Migrants per round")
    games_slider = mo.ui.slider(10, 100, value=_def("games", 20), step=5, label="Games per eval")
    seed_input = mo.ui.number(value=_def("seed", 42), start=0, stop=9999, label="Seed")

    mo.md("## Settings")

    controls = mo.vstack([
        mo.hstack(
            [n_islands_slider, total_pop_slider, generations_slider],
            justify="start",
        ),
        mo.hstack(
            [migration_interval_slider, n_migrants_slider, games_slider, seed_input],
            justify="start",
        ),
    ])
    controls
    return (
        games_slider, generations_slider, migration_interval_slider,
        n_islands_slider, n_migrants_slider, seed_input, total_pop_slider,
    )


@app.cell
def _(
    games_slider, generations_slider, migration_interval_slider, mo,
    n_islands_slider, n_migrants_slider, total_pop_slider,
):
    run_btn = mo.ui.run_button(label="Run Island Model GA")
    _island_pop = total_pop_slider.value // n_islands_slider.value
    _n_rounds = generations_slider.value // migration_interval_slider.value
    mo.hstack([
        run_btn,
        mo.md(
            f"**{n_islands_slider.value}** islands × "
            f"**{_island_pop}** agents each · "
            f"**{generations_slider.value}** generations · "
            f"migrate every **{migration_interval_slider.value}** gens "
            f"(**{n_migrants_slider.value}** migrants) · "
            f"**{games_slider.value}** games/eval · "
            f"**{_n_rounds}** migration rounds"
        ),
    ], justify="start")
    return (run_btn,)


@app.cell
async def _(
    GeneticAlgorithm, IslandActor, args, games_slider, generations_slider,
    migration_interval_slider, mo, n_islands_slider, n_migrants_slider,
    np, run_btn, seed_input, sushigo_dir, this_host, total_pop_slider,
):
    import time

    # Auto-run when invoked with CLI args; otherwise wait for button
    if not args:
        mo.stop(not run_btn.value, mo.md("*Click **Run Island Model GA** above to start.*"))

    # --- Unpack settings ---
    n_islands = n_islands_slider.value
    total_pop_size = total_pop_slider.value
    island_pop_size = total_pop_size // n_islands
    n_generations = generations_slider.value
    migration_interval = migration_interval_slider.value
    n_migrants = n_migrants_slider.value
    games_per_eval = games_slider.value
    seed = seed_input.value
    elite_count = max(1, island_pop_size // 20)
    tournament_k = 3
    mutation_rate = 0.1
    mutation_sigma = 0.3
    n_migration_rounds = n_generations // migration_interval

    # ==================================================================
    # Part A: Monarch Island Model
    # ==================================================================
    mesh = this_host().spawn_procs(per_host={"island": n_islands})

    islands = []
    for _i in range(n_islands):
        _isl = mesh.slice(island=_i).spawn(
            f"island_{_i}", IslandActor,
            _i, island_pop_size, games_per_eval, elite_count,
            tournament_k, mutation_rate, mutation_sigma, seed, sushigo_dir,
        )
        islands.append(_isl)

    monarch_history = {"generation": [], "best": [], "mean": []}
    t0 = time.perf_counter()

    for round_idx in range(n_migration_rounds):
        # Evolve all islands in parallel
        futures = [isl.evolve.call_one(migration_interval) for isl in islands]
        all_stats = [f.get() for f in futures]

        # Record per-generation global stats
        for gen_offset in range(migration_interval):
            gen = round_idx * migration_interval + gen_offset
            best_across = max(s[gen_offset]["best"] for s in all_stats)
            mean_across = np.mean([s[gen_offset]["mean"] for s in all_stats])
            monarch_history["generation"].append(gen)
            monarch_history["best"].append(best_across)
            monarch_history["mean"].append(float(mean_across))

        # Migration (ring topology): island i receives best from island (i-1)
        if round_idx < n_migration_rounds - 1:
            best_futures = [isl.get_best.call_one(n_migrants) for isl in islands]
            all_best = [f.get() for f in best_futures]

            migrate_futures = []
            for _i, isl in enumerate(islands):
                src = (_i - 1) % n_islands
                migrate_futures.append(isl.receive_migrants.call_one(all_best[src]))
            for f in migrate_futures:
                f.get()

    monarch_time = time.perf_counter() - t0
    mesh.stop().get()

    # ==================================================================
    # Part B: Multiprocessing Island Model
    # ==================================================================
    import multiprocessing as _mp
    from island_worker import island_worker

    cmd_queues = []
    result_queues = []
    processes = []

    t0 = time.perf_counter()

    for _i in range(n_islands):
        cmd_q = _mp.Queue()
        res_q = _mp.Queue()
        p = _mp.Process(target=island_worker, args=(
            _i, island_pop_size, games_per_eval, elite_count, tournament_k,
            mutation_rate, mutation_sigma, seed, sushigo_dir,
            cmd_q, res_q,
        ))
        p.start()
        cmd_queues.append(cmd_q)
        result_queues.append(res_q)
        processes.append(p)

    mp_history = {"generation": [], "best": [], "mean": []}

    for round_idx in range(n_migration_rounds):
        # Send evolve command to all islands
        for q in cmd_queues:
            q.put(("evolve", migration_interval))

        # Collect results
        all_stats = []
        for q in result_queues:
            _, stats = q.get()
            all_stats.append(stats)

        # Record per-generation global stats
        for gen_offset in range(migration_interval):
            gen = round_idx * migration_interval + gen_offset
            best_across = max(s[gen_offset]["best"] for s in all_stats)
            mean_across = np.mean([s[gen_offset]["mean"] for s in all_stats])
            mp_history["generation"].append(gen)
            mp_history["best"].append(best_across)
            mp_history["mean"].append(float(mean_across))

        # Migration
        if round_idx < n_migration_rounds - 1:
            for q in cmd_queues:
                q.put(("get_best", n_migrants))

            all_best = []
            for q in result_queues:
                _, chroms = q.get()
                all_best.append(chroms)

            for _i, q in enumerate(cmd_queues):
                src = (_i - 1) % n_islands
                q.put(("receive_migrants", all_best[src]))

            for q in result_queues:
                q.get()  # wait for ack

    # Cleanup
    for q in cmd_queues:
        q.put(("stop",))
    for p in processes:
        p.join()

    mp_time = time.perf_counter() - t0

    # ==================================================================
    # Part C: Sequential baseline (single population, same total gens)
    # ==================================================================
    t0 = time.perf_counter()
    seq_ga = GeneticAlgorithm(
        pop_size=total_pop_size,
        elite_count=max(1, total_pop_size // 20),
        tournament_k=tournament_k,
        mutation_rate=mutation_rate,
        mutation_sigma=mutation_sigma,
        games_per_eval=games_per_eval,
        seed=seed,
    )
    seq_result = seq_ga.run(generations=n_generations)
    seq_time = time.perf_counter() - t0
    seq_history = seq_result["history"]

    return (
        monarch_history, monarch_time,
        mp_history, mp_time,
        seq_history, seq_time,
        n_islands, n_generations, migration_interval,
    )


@app.cell(hide_code=True)
def _(mo, monarch_time, mp_time, n_islands, seq_time):
    _seq_over_monarch = seq_time / monarch_time if monarch_time > 0 else float("inf")
    _seq_over_mp = seq_time / mp_time if mp_time > 0 else float("inf")
    _mp_over_monarch = mp_time / monarch_time if monarch_time > 0 else float("inf")
    mo.md(f"""
    ## ⏱ Timing

    | Method | Wall time | vs Sequential |
    |--------|-----------|---------------|
    | **Sequential** (single pop) | {seq_time:.1f}s | 1.0× |
    | **Multiprocessing** ({n_islands} islands) | {mp_time:.1f}s | **{_seq_over_mp:.2f}×** |
    | **Monarch** ({n_islands} islands) | {monarch_time:.1f}s | **{_seq_over_monarch:.2f}×** |

    Monarch vs Multiprocessing: **{_mp_over_monarch:.2f}×**
    """)
    return


@app.cell(hide_code=True)
def _(alt, monarch_time, mp_time, n_islands, pl, seq_time):
    _timing_df = pl.DataFrame({
        "method": ["Sequential", f"Multiprocessing ({n_islands} islands)", f"Monarch ({n_islands} islands)"],
        "seconds": [seq_time, mp_time, monarch_time],
        "label": [
            f"{seq_time:.1f}s",
            f"{mp_time:.1f}s",
            f"{monarch_time:.1f}s",
        ],
    })
    _order = _timing_df["method"].to_list()

    _bars = (
        alt.Chart(_timing_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("method:N", title=None, sort=_order),
            y=alt.Y("seconds:Q", title="Wall time (s)"),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=_order,
                    range=["#95a5a6", "#3498db", "#e67e22"],
                ),
                legend=None,
            ),
        )
    )
    _text = (
        alt.Chart(_timing_df)
        .mark_text(dy=-10, fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("method:N", sort=_order),
            y=alt.Y("seconds:Q"),
            text="label:N",
        )
    )
    (_bars + _text).properties(width=400, height=300, title="Island Model GA — Wall Time")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Convergence")
    return


@app.cell(hide_code=True)
def _(alt, monarch_history, mp_history, n_generations, pl, seq_history):
    def _make_df(history, method):
        # Trim to n_generations entries (sequential run() records gen+1 entries)
        n = min(len(history["generation"]), n_generations)
        return pl.DataFrame({
            "generation": history["generation"][:n],
            "best": history["best"][:n],
            "mean": history["mean"][:n],
        }).unpivot(
            index="generation",
            on=["best", "mean"],
            variable_name="metric",
            value_name="fitness",
        ).with_columns(pl.lit(method).alias("method"))

    _combined = pl.concat([
        _make_df(monarch_history, "Monarch (island)"),
        _make_df(mp_history, "Multiprocessing (island)"),
        _make_df(seq_history, "Sequential (single pop)"),
    ])

    _chart = (
        alt.Chart(_combined)
        .mark_line()
        .encode(
            x=alt.X("generation:Q", title="Generation"),
            y=alt.Y("fitness:Q", title="Win rate vs random", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("method:N", title="Method"),
            strokeDash=alt.StrokeDash("metric:N", title="Metric"),
            strokeWidth=alt.condition(
                alt.datum.metric == "best",
                alt.value(2.5),
                alt.value(1.5),
            ),
        )
        .properties(width=650, height=400, title="Fitness Convergence — Island Model vs Single Population")
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Code Complexity Story

    The Island Model requires each worker to be **stateful** — it owns a population, a
    GA instance, and fitness scores that persist across calls.  Compare how each
    framework handles this:
    """)

    _monarch_code = '''\
    class IslandActor(Actor):

        def __init__(self, island_id, ...):
            self.ga = GeneticAlgorithm(...)
            self.population = self.ga._make_population()
            self.fitness = None

        @endpoint
        def evolve(self, n_gens):
            for _ in range(n_gens):
                self.fitness = self.ga.evaluate_fitness(self.population)
                self.population = self.ga.evolve_step(...)
            return stats

        @endpoint
        def get_best(self, k): ...

        @endpoint
        def receive_migrants(self, chromosomes): ...

    # --- Usage ---
    mesh = this_host().spawn_procs(per_host={"island": 4})
    islands = [mesh.slice(island=i).spawn("island", IslandActor, ...)
               for i in range(4)]
    futures = [isl.evolve.call_one(5) for isl in islands]  # parallel!
    '''

    _mp_code = '''\
    def island_worker(island_id, ..., cmd_queue, result_queue):
        ga = GeneticAlgorithm(...)
        population = ga._make_population()
        fitness = None
        while True:
            cmd = cmd_queue.get()
            if cmd[0] == "evolve":
                # ... do work ...
                result_queue.put(("stats", stats))
            elif cmd[0] == "get_best":
                # ... more command handling ...
            elif cmd[0] == "receive_migrants":
                # ... more command handling ...
            elif cmd[0] == "stop":
                break

    # --- Usage ---
    cmd_queues, result_queues, processes = [], [], []
    for i in range(4):
        cmd_q, res_q = mp.Queue(), mp.Queue()
        p = mp.Process(target=island_worker, args=(i,...,cmd_q,res_q))
        p.start()
        ...
    for q in cmd_queues:
        q.put(("evolve", 5))
    all_stats = [q.get() for q in result_queues]  # manual collect
    for q in cmd_queues:
        q.put(("stop",))
    for p in processes:
        p.join()  # manual cleanup
    '''

    mo.hstack([
        mo.vstack([
            mo.md("### 🟠 Monarch (~30 lines)"),
            mo.md(f"```python\n{_monarch_code}```"),
        ]),
        mo.vstack([
            mo.md("### 🔵 Multiprocessing (~50 lines)"),
            mo.md(f"```python\n{_mp_code}```"),
        ]),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How It Works

    ### The Island Model GA

    Instead of one large population, we split agents across **isolated islands** that
    evolve independently. Periodically, each island sends its best agents to a
    neighbour (ring topology). This gives us:

    - **Diversity preservation** — islands explore different regions of the search space
    - **Implicit parallelism** — each island's evaluation + selection is independent
    - **Migration pressure** — good genes spread across islands, preventing stagnation

    ### Why Actors Fit Naturally

    Each island is inherently **stateful** — it holds a population that changes over
    time. The actor model maps perfectly to this:

    | Concept | Actor mapping |
    |---------|---------------|
    | Island population | Actor instance state |
    | "Evolve for N generations" | `@endpoint evolve(n_gens)` |
    | "Send me your best K agents" | `@endpoint get_best(k)` |
    | "Here are incoming migrants" | `@endpoint receive_migrants(chroms)` |
    | Ring-topology migration | Orchestrator calls endpoints in the right order |

    With **multiprocessing**, you have to build this yourself: a command loop,
    Queue-based message passing, a protocol of `("evolve", n)` tuples, and
    manual process lifecycle management.

    With **Monarch**, the actor *is* the island. Call its methods. Done.

    And the kicker: **Monarch scales to multiple machines with zero code changes**.
    Multiprocessing is limited to a single host.
    """)
    return


if __name__ == "__main__":
    app.run()
