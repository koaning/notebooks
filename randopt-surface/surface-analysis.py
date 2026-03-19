# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate==1.13.0",
#     "diskcache==5.6.3",
#     "marimo",
#     "matplotlib",
#     "numpy==2.4.3",
#     "pandas",
#     "python-dotenv",
#     "transformers==5.3.0",
#     "wandb",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import wandb
    from diskcache import Cache
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return Cache, Path, itertools, mo, np, pd, plt, wandb


@app.cell
def _():
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    return AutoConfig, AutoModelForCausalLM, init_empty_weights


@app.cell
def _():
    wandb_project = "randopt-surface"
    wandb_group = "qwen2.5-3b-s0.002-data0"
    max_runs = 500
    plot_seed_limit = 50
    camera_seed = 999
    chunk_size = 1_000_000
    projection_batch_size = 8
    return (
        camera_seed,
        chunk_size,
        max_runs,
        plot_seed_limit,
        projection_batch_size,
        wandb_group,
        wandb_project,
    )


@app.cell
def _(Path):
    runs_cache_path = Path(".context") / "randopt-surface-runs.csv"
    runs_cache_path.parent.mkdir(parents=True, exist_ok=True)
    return (runs_cache_path,)


@app.cell
def _(
    itertools,
    max_runs,
    np,
    pd,
    runs_cache_path,
    wandb,
    wandb_group,
    wandb_project,
):
    def _to_float(value):
        if value is None:
            return np.nan
        return float(value)

    api = wandb.Api()
    entity = api.default_entity
    path = f"{entity}/{wandb_project}"

    if runs_cache_path.exists():
        runs_df = pd.read_csv(runs_cache_path)
    else:
        filters = {"group": wandb_group}
        rows = []
        for run in itertools.islice(api.runs(path=path, filters=filters), max_runs):
            summary = dict(run.summary)
            _config = {k: v for k, v in run.config.items() if not k.startswith("_")}
            _seed = summary.get("seed", _config.get("seed"))
            if _seed is None:
                continue
            rows.append({
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "seed": int(_seed),
                "seed_acc": _to_float(summary.get("seed_acc")),
                "base_acc": _to_float(summary.get("base_acc")),
                "uplift_abs": _to_float(summary.get("uplift_abs")),
                "uplift_rel": _to_float(summary.get("uplift_rel")),
                "sigma": _to_float(_config.get("sigma")),
                "model_id": _config.get("model_id"),
                "data_seed": _config.get("data_seed"),
                "wandb_group": run.group,
            })

        runs_df = pd.DataFrame(rows)
        if not runs_df.empty:
            runs_df = runs_df.sort_values("seed").reset_index(drop=True)
        runs_df.to_csv(runs_cache_path, index=False)
    return path, runs_df


@app.cell
def _(runs_df):
    model_ids = runs_df["model_id"].dropna().unique().tolist()
    if len(model_ids) != 1:
        raise ValueError(f"Expected exactly one model_id, found: {model_ids}")
    model_id = model_ids[0]
    return (model_id,)


@app.cell
def _(plot_seed_limit, runs_df):
    plot_runs_df = runs_df.sort_values("seed").head(plot_seed_limit).reset_index(drop=True)
    return (plot_runs_df,)


@app.cell
def _(AutoConfig, AutoModelForCausalLM, init_empty_weights, model_id):
    _config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(_config)
    param_sizes = [param.numel() for param in model.parameters()]
    total_dim = int(sum(param_sizes))
    return param_sizes, total_dim


@app.cell
def _(mo, path, plot_runs_df):
    mo.stop(plot_runs_df.empty, mo.md(f"No runs found for `{path}`."))
    return


@app.cell
def _(np):
    def regenerate_noise_prefix(seed, length):
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        return rng.standard_normal(length, dtype=np.float32)

    def regenerate_noise_matrix(seeds, length):
        return np.stack([regenerate_noise_prefix(seed, length) for seed in seeds], axis=0)

    return (regenerate_noise_matrix,)


@app.cell
def _(plot_runs_df, regenerate_noise_matrix):
    seeds = plot_runs_df["seed"].astype(int).tolist()
    first_two_coords = regenerate_noise_matrix(seeds, 10)
    return (first_two_coords,)


@app.cell
def _(Cache, Path):
    cache_dir = Path(".context") / "randopt-surface-paper-projection-cache"
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    projection_cache = Cache(str(cache_dir))
    return (projection_cache,)


@app.cell
def _(
    camera_seed,
    chunk_size,
    np,
    param_sizes,
    pd,
    plot_runs_df,
    projection_batch_size,
    projection_cache,
    total_dim,
):
    projection_cache_version = 2

    def projection_cache_key(seed):
        return (
            projection_cache_version,
            int(camera_seed),
            int(total_dim),
            int(seed),
        )

    known_seeds = {
        int(seed)
        for seed in plot_runs_df["seed"].astype(int).tolist()
        if projection_cache.get(projection_cache_key(seed)) is not None
    }
    missing_seeds = [
        int(seed)
        for seed in plot_runs_df["seed"].astype(int).tolist()
        if int(seed) not in known_seeds
    ]

    scale = float(np.sqrt(total_dim))
    seed_batches = [
        missing_seeds[start : start + projection_batch_size]
        for start in range(0, len(missing_seeds), projection_batch_size)
    ]

    for index, seed_batch in enumerate(seed_batches, start=1):
        print(
            f"[paper projection batch {index}/{len(seed_batches)}] "
            f"seeds={seed_batch[0]}..{seed_batch[-1]}",
            flush=True,
        )
        expert_rngs = {
            seed: np.random.Generator(np.random.PCG64(seed))
            for seed in seed_batch
        }
        batch_x = {seed: 0.0 for seed in seed_batch}
        batch_y = {seed: 0.0 for seed in seed_batch}
        camera_x_rng = np.random.Generator(np.random.PCG64(camera_seed))
        camera_y_rng = np.random.Generator(np.random.PCG64(camera_seed + 1))

        for tensor_size in param_sizes:
            remaining = tensor_size
            while remaining > 0:
                current = min(chunk_size, remaining)
                u = camera_x_rng.standard_normal(current, dtype=np.float32)
                v = camera_y_rng.standard_normal(current, dtype=np.float32)
                eps_batch = np.stack(
                    [
                        expert_rngs[seed].standard_normal(current, dtype=np.float32)
                        for seed in seed_batch
                    ],
                    axis=0,
                )
                batch_x_values = eps_batch @ u
                batch_y_values = eps_batch @ v
                for batch_index, seed in enumerate(seed_batch):
                    batch_x[seed] += float(batch_x_values[batch_index])
                    batch_y[seed] += float(batch_y_values[batch_index])
                remaining -= current

        for seed in seed_batch:
            projection_cache.set(
                projection_cache_key(seed),
                (batch_x[seed] / scale, batch_y[seed] / scale),
            )

    cached_rows = []
    for seed in plot_runs_df["seed"].astype(int).tolist():
        cached_value = projection_cache.get(projection_cache_key(seed))
        if cached_value is None:
            cached_rows.append({"seed": seed, "paper_x": np.nan, "paper_y": np.nan})
        else:
            paper_x, paper_y = cached_value
            cached_rows.append({"seed": seed, "paper_x": paper_x, "paper_y": paper_y})

    cached_df = pd.DataFrame(cached_rows)
    paper_projected_df = plot_runs_df.merge(cached_df, on="seed", how="left")
    return (paper_projected_df,)


@app.cell
def _(first_two_coords, plot_runs_df):
    projected_df = plot_runs_df.copy()
    projected_df["draw_1"] = first_two_coords[:, 0]
    projected_df["draw_2"] = first_two_coords[:, 4]
    projected_df["uplift_abs_filled"] = projected_df["uplift_abs"].fillna(0.0)
    return (projected_df,)


@app.cell(hide_code=True)
def _(plt, projected_df):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _scatter = _ax.scatter(
        projected_df["draw_1"],
        projected_df["draw_2"],
        c=projected_df["uplift_abs_filled"],
        cmap="viridis",
        s=50,
        alpha=0.85,
    )
    _top_rows = projected_df.sort_values("uplift_abs", ascending=False).head(5)
    for _, _row in _top_rows.iterrows():
        _ax.annotate(str(int(_row["seed"])), (_row["draw_1"], _row["draw_2"]), fontsize=8)

    _ax.set_title("Seed First Two Dimensions")
    _ax.set_xlabel("draw_1")
    _ax.set_ylabel("draw_2")
    _fig.colorbar(_scatter, ax=_ax, label="uplift_abs")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(paper_projected_df, plt):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _scatter = _ax.scatter(
        paper_projected_df["paper_x"],
        paper_projected_df["paper_y"],
        c=paper_projected_df["uplift_abs"].fillna(0.0),
        cmap="viridis",
        s=50,
        alpha=0.85,
    )
    _top_rows = paper_projected_df.sort_values("uplift_abs", ascending=False).head(5)
    for _, _row in _top_rows.iterrows():
        _ax.annotate(str(int(_row["seed"])), (_row["paper_x"], _row["paper_y"]), fontsize=8)

    _ax.set_title("Streaming Random Projection")
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _fig.colorbar(_scatter, ax=_ax, label="uplift_abs")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
