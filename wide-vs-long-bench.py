# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "pandas",
#     "polars==1.37.1",
#     "pyarrow",
#     "numpy",
#     "altair",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    import altair as alt
    import tempfile
    import time
    import shutil
    from pathlib import Path
    return Path, alt, duckdb, mo, np, pd, pl, tempfile, time


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Wide vs Long Format Benchmark

    Comparing two ways to structure time-series sensor data from multiple Raspberry Pis:

    **Wide format:** One column per sensor
    ```
    timestamp | rpi_id | temp | humidity | pressure | ...
    ```

    **Long format:** One row per reading
    ```
    timestamp | rpi_id | sensor_id | value
    ```

    We'll benchmark analytical queries on both formats using DuckDB and Parquet.
    """)
    return


@app.cell
def _(mo):
    config_form = mo.ui.batch(
        mo.md("""
        **Configure Benchmark**

        {n_rpis} {n_sensors} {n_months}
        """),
        {
            "n_rpis": mo.ui.slider(start=2, stop=20, value=3, step=1, label="Number of RPIs"),
            "n_sensors": mo.ui.slider(start=3, stop=20, value=5, step=1, label="Sensors per RPI"),
            "n_months": mo.ui.slider(start=1, stop=24, value=1, step=1, label="Months of data"),
        }
    ).form(submit_button_label="Run Benchmark")
    config_form
    return (config_form,)


@app.cell
def _(config_form, mo):
    # Use defaults in script mode (form not submitted) or extract form values
    if config_form.value is None:
        # Script mode or form not yet submitted - use defaults
        n_rpis = 3
        n_sensors = 5
        n_months = 1
    else:
        n_rpis = config_form.value["n_rpis"]
        n_sensors = config_form.value["n_sensors"]
        n_months = config_form.value["n_months"]
    # 1 reading per minute: 60 * 24 * 30 = 43,200 per month
    n_timestamps = n_months * 43_200
    mo.stop(config_form.value is None and mo.app_meta().mode != "script")
    return n_months, n_rpis, n_sensors, n_timestamps


@app.cell
def _(mo, n_months, n_rpis, n_sensors, n_timestamps):
    total_wide_rows = n_rpis * n_timestamps
    total_long_rows = n_rpis * n_timestamps * n_sensors

    mo.md(f"""
    **Data scale:** {n_months} month(s) of data, 1 reading/minute
    - Wide format: {total_wide_rows:,} rows x {n_sensors + 2} columns
    - Long format: {total_long_rows:,} rows x 4 columns
    """)
    return


@app.cell(hide_code=True)
def _(np, pd, pl):
    def generate_data(n_rpis, n_sensors, n_timestamps, seed=42):
        """Generate simulated sensor data with varied types (vectorized)."""
        rng = np.random.default_rng(seed)
        n_rows = n_rpis * n_timestamps

        # Generate timestamps - tile for each RPI
        base_time = pd.Timestamp("2024-01-01")
        timestamps = pd.date_range(base_time, periods=n_timestamps, freq="15s")
        all_timestamps = np.tile(timestamps, n_rpis)

        # Generate rpi_ids - repeat each RPI for all timestamps
        rpi_ids = np.repeat([f"rpi_{i}" for i in range(n_rpis)], n_timestamps)

        # RPI offsets for device variation (broadcast across timestamps)
        rpi_offsets = np.repeat(np.arange(n_rpis) * 0.1, n_timestamps)

        # Build sensor columns vectorized
        sensor_data = {}
        for i in range(n_sensors):
            if i % 3 == 0:
                # Integer sensor
                base = 100 + i * 10
                noise_scale = 5
                values = (base * (1 + rpi_offsets) + rng.normal(0, noise_scale, n_rows)).astype(np.int32)
            elif i % 3 == 1:
                # Large float sensor (10k-30k range)
                base = 20000 + i * 500
                noise_scale = 2000
                values = base * (1 + rpi_offsets) + rng.normal(0, noise_scale, n_rows)
            else:
                # Small float sensor
                base = 20 + i
                noise_scale = 2
                values = base * (1 + rpi_offsets) + rng.normal(0, noise_scale, n_rows)
            sensor_data[f"sensor_{i}"] = values

        # Build wide DataFrame
        wide_df = pd.DataFrame({
            "timestamp": all_timestamps,
            "rpi_id": rpi_ids,
            **sensor_data
        })

        # Use polars for fast melt
        sensor_cols = [f"sensor_{i}" for i in range(n_sensors)]
        wide_pl = pl.from_pandas(wide_df)
        long_pl = wide_pl.unpivot(
            index=["timestamp", "rpi_id"],
            on=sensor_cols,
            variable_name="sensor_id",
            value_name="value"
        )
        long_df = long_pl.to_pandas()

        return wide_df, long_df
    return (generate_data,)


@app.cell
def _(generate_data, n_rpis, n_sensors, n_timestamps):
    wide_df, long_df = generate_data(n_rpis, n_sensors, n_timestamps)
    return long_df, wide_df


@app.cell
def _(long_df, mo, wide_df):
    mo.vstack([
        mo.md(f"""
        ## Generated Data

        **Wide format shape:** {wide_df.shape}
        **Long format shape:** {long_df.shape}
        """),
        mo.accordion({
            "Wide Format Sample": wide_df.head(10),
            "Long Format Sample": long_df.head(10),
        })
    ])
    return


@app.cell(hide_code=True)
def _(Path, duckdb, long_df, pd, tempfile, wide_df):
    # Create temp directory for parquet files
    temp_dir = Path(tempfile.mkdtemp(prefix="bench_"))

    # Add date column for date partitioning
    wide_with_date = wide_df.copy()
    wide_with_date["date"] = pd.to_datetime(wide_with_date["timestamp"]).dt.date.astype(str)

    long_with_date = long_df.copy()
    long_with_date["date"] = pd.to_datetime(long_with_date["timestamp"]).dt.date.astype(str)

    # Setup DuckDB connection
    con = duckdb.connect(":memory:")

    # Register DataFrames as DuckDB tables
    con.register("wide_table", wide_with_date)
    con.register("long_table", long_with_date)

    # Write Parquet files (no partitioning)
    wide_parquet = temp_dir / "wide.parquet"
    long_parquet = temp_dir / "long.parquet"
    wide_with_date.to_parquet(wide_parquet)
    long_with_date.to_parquet(long_parquet)

    # Partitioned by rpi_id
    wide_by_rpi = temp_dir / "wide_by_rpi"
    long_by_rpi = temp_dir / "long_by_rpi"
    wide_with_date.to_parquet(wide_by_rpi, partition_cols=["rpi_id"])
    long_with_date.to_parquet(long_by_rpi, partition_cols=["rpi_id"])

    # Partitioned by date
    wide_by_date = temp_dir / "wide_by_date"
    long_by_date = temp_dir / "long_by_date"
    wide_with_date.to_parquet(wide_by_date, partition_cols=["date"])
    long_with_date.to_parquet(long_by_date, partition_cols=["date"])

    # Long format partitioned by sensor_id
    long_by_sensor = temp_dir / "long_by_sensor"
    long_with_date.to_parquet(long_by_sensor, partition_cols=["sensor_id"])

    # Calculate file sizes
    def get_dir_size(path):
        path = Path(path)
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    storage_sizes = {
        "Wide (flat)": get_dir_size(wide_parquet),
        "Long (flat)": get_dir_size(long_parquet),
        "Wide (by rpi)": get_dir_size(wide_by_rpi),
        "Long (by rpi)": get_dir_size(long_by_rpi),
        "Wide (by date)": get_dir_size(wide_by_date),
        "Long (by date)": get_dir_size(long_by_date),
        "Long (by sensor)": get_dir_size(long_by_sensor),
    }
    return (
        con,
        long_by_date,
        long_by_rpi,
        long_by_sensor,
        long_parquet,
        storage_sizes,
        wide_by_date,
        wide_by_rpi,
        wide_parquet,
    )


@app.cell(hide_code=True)
def _(mo, pd, storage_sizes):
    def human_size(size_bytes):
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    sizes_df = pd.DataFrame([
        {"Format": k, "Size": human_size(v)}
        for k, v in storage_sizes.items()
    ])
    mo.vstack([
        mo.md("## Storage Sizes"),
        sizes_df
    ])
    return


@app.cell(hide_code=True)
def _(
    con,
    long_by_date,
    long_by_rpi,
    long_by_sensor,
    long_parquet,
    np,
    time,
    wide_by_date,
    wide_by_rpi,
    wide_parquet,
):
    def run_benchmark(n_iterations=5):
        """Run benchmark queries with multiple iterations, return median times."""
        results = []

        def time_query(sql, n_iter):
            """Run query n_iter times, return median time in ms."""
            # Warmup run
            con.execute(sql).fetchall()
            # Timed runs
            times = []
            for _ in range(n_iter):
                start = time.perf_counter()
                con.execute(sql).fetchall()
                times.append((time.perf_counter() - start) * 1000)
            return np.median(times)

        def pq(path):
            """Helper to build read_parquet with hive partitioning."""
            return f"read_parquet('{path}/**/*.parquet', hive_partitioning=true)"

        # All storage backends (wide_src, long_src) - None means skip that format
        storage_backends = [
            ("flat", f"read_parquet('{wide_parquet}')", f"read_parquet('{long_parquet}')"),
            ("by_rpi", pq(wide_by_rpi), pq(long_by_rpi)),
            ("by_date", pq(wide_by_date), pq(long_by_date)),
            ("by_sensor", None, pq(long_by_sensor)),  # Only long has sensor partitioning
        ]

        # Run all queries against all backends
        for storage, wide_src, long_src in storage_backends:
            # Query 1: One date + one sensor (filter-heavy query)
            if wide_src:
                results.append({
                    "query": "date_sensor",
                    "format": "wide",
                    "storage": storage,
                    "time_ms": time_query(f"""
                        SELECT timestamp, rpi_id, sensor_0
                        FROM {wide_src}
                        WHERE date = '2024-01-01'
                    """, n_iterations)
                })
            results.append({
                "query": "date_sensor",
                "format": "long",
                "storage": storage,
                "time_ms": time_query(f"""
                    SELECT timestamp, rpi_id, value
                    FROM {long_src}
                    WHERE date = '2024-01-01' AND sensor_id = 'sensor_0'
                """, n_iterations)
            })

            # Query 2: One RPI, one sensor, daily aggregation
            if wide_src:
                results.append({
                    "query": "rpi_daily_agg",
                    "format": "wide",
                    "storage": storage,
                    "time_ms": time_query(f"""
                        SELECT date, AVG(sensor_0) as avg_value
                        FROM {wide_src}
                        WHERE rpi_id = 'rpi_0'
                        GROUP BY date
                        ORDER BY date
                    """, n_iterations)
                })
            results.append({
                "query": "rpi_daily_agg",
                "format": "long",
                "storage": storage,
                "time_ms": time_query(f"""
                    SELECT date, AVG(value) as avg_value
                    FROM {long_src}
                    WHERE rpi_id = 'rpi_0' AND sensor_id = 'sensor_0'
                    GROUP BY date
                    ORDER BY date
                """, n_iterations)
            })

        return results

    benchmark_results = run_benchmark()
    return (benchmark_results,)


@app.cell(hide_code=True)
def _(benchmark_results, pd):
    results_df = pd.DataFrame(benchmark_results)
    results_df
    return (results_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Benchmark Results
    """)
    return


@app.cell
def _(alt, results_df):
    chart = alt.Chart(results_df).mark_bar().encode(
        x=alt.X("query:N", title="Query Type"),
        y=alt.Y("time_ms:Q", title="Time (ms)"),
        color=alt.Color("format:N", title="Format"),
        xOffset="format:N",
        column=alt.Column("storage:N", title="Storage Backend")
    ).properties(
        width=250,
        height=300,
        title="Query Performance: Wide vs Long Format"
    )
    chart
    return


@app.cell
def _(mo, results_df):
    # Create pivot for comparison
    pivot = results_df.pivot_table(
        index=["query", "storage"],
        columns="format",
        values="time_ms"
    ).reset_index()
    pivot["wide_faster"] = pivot["wide"] < pivot["long"]
    pivot["speedup"] = pivot["long"] / pivot["wide"]

    mo.md(f"""
    ## Summary

    **Wide format wins:** {pivot["wide_faster"].sum()} queries
    **Long format wins:** {(~pivot["wide_faster"]).sum()} queries

    **Queries tested:**
    - `date_sensor`: Filter by one date + one sensor
    - `rpi_daily_agg`: One RPI, one sensor, aggregate by day

    Wide format advantage: single column access, no row filtering for sensor
    Long format advantage: partition pruning on sensor_id
    """)
    return (pivot,)


@app.cell
def _(pivot):
    pivot
    return


if __name__ == "__main__":
    app.run()
