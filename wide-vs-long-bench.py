# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "polars==1.37.1",
#     "pyarrow",
#     "numpy",
#     "altair",
#     "pandas==3.0.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import numpy as np
    import altair as alt
    import tempfile
    import time
    from pathlib import Path

    return Path, alt, duckdb, mo, np, pl, tempfile, time


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
def _(np, pl):
    def generate_data(n_rpis, n_sensors, n_timestamps, seed=42):
        """Generate simulated sensor data with explicit physical dtypes."""
        rng = np.random.default_rng(seed)
        n_rows = n_rpis * n_timestamps

        # Generate timestamps - tile for each RPI
        base_time = np.datetime64("2024-01-01T00:00:00")
        timestamps = (base_time + np.arange(n_timestamps) * np.timedelta64(1, "m")).astype("datetime64[us]")
        all_timestamps = np.tile(timestamps, n_rpis)

        # Generate rpi_ids - repeat each RPI for all timestamps
        rpi_ids = np.repeat(np.array([f"rpi_{i}" for i in range(n_rpis)]), n_timestamps)

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
                values = (base * (1 + rpi_offsets) + rng.normal(0, noise_scale, n_rows)).astype(np.float32)
            else:
                # Small float sensor
                base = 20 + i
                noise_scale = 2
                values = (base * (1 + rpi_offsets) + rng.normal(0, noise_scale, n_rows)).astype(np.float32)
            sensor_data[f"sensor_{i}"] = values

        # Build wide DataFrame in Polars
        wide_df = pl.DataFrame({
            "timestamp": all_timestamps,
            "rpi_id": rpi_ids,
            **sensor_data
        }).with_columns([
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("rpi_id").cast(pl.Categorical),
        ])

        # Unpivot to long and enforce compact ID/value types
        sensor_cols = [f"sensor_{i}" for i in range(n_sensors)]
        long_df = wide_df.unpivot(
            index=["timestamp", "rpi_id"],
            on=sensor_cols,
            variable_name="sensor_id",
            value_name="value"
        ).with_columns([
            pl.col("sensor_id").cast(pl.Categorical),
            pl.col("value").cast(pl.Float32),
        ])

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
def _(Path, duckdb, long_df, pl, tempfile, wide_df):
    # Create temp directory for parquet files
    temp_dir = Path(tempfile.mkdtemp(prefix="bench_"))

    # Add date column with physical DATE type (date32 in Parquet)
    wide_with_date = wide_df.with_columns(
        pl.col("timestamp").dt.date().alias("date")
    )

    long_with_date = long_df.with_columns(
        pl.col("timestamp").dt.date().alias("date")
    )

    # Setup DuckDB connection
    con = duckdb.connect(":memory:")

    # Write Parquet files (no partitioning)
    wide_parquet = temp_dir / "wide.parquet"
    long_parquet = temp_dir / "long.parquet"
    wide_with_date.write_parquet(wide_parquet, compression="zstd", statistics=True)
    long_with_date.write_parquet(long_parquet, compression="zstd", statistics=True)

    def write_partitioned(df, path, partition_cols):
        df.write_parquet(
            path,
            use_pyarrow=True,
            compression="zstd",
            statistics=True,
            pyarrow_options={"partition_cols": partition_cols},
        )

    # Partitioned by rpi_id
    wide_by_rpi = temp_dir / "wide_by_rpi"
    long_by_rpi = temp_dir / "long_by_rpi"
    write_partitioned(wide_with_date, wide_by_rpi, ["rpi_id"])
    write_partitioned(long_with_date, long_by_rpi, ["rpi_id"])

    # Partitioned by date
    wide_by_date = temp_dir / "wide_by_date"
    long_by_date = temp_dir / "long_by_date"
    write_partitioned(wide_with_date, wide_by_date, ["date"])
    write_partitioned(long_with_date, long_by_date, ["date"])

    # Long format partitioned by sensor_id
    long_by_sensor = temp_dir / "long_by_sensor"
    write_partitioned(long_with_date, long_by_sensor, ["sensor_id"])

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
def _(mo, n_rpis, n_sensors, n_timestamps, pl, storage_sizes):
    def human_size(size_bytes):
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    total_readings = n_rpis * n_timestamps * n_sensors
    sizes_df = pl.DataFrame([
        {
            "Format": k,
            "Size": human_size(v),
            "Bytes": int(v),
            "Bytes/reading": round(v / total_readings, 4),
        }
        for k, v in storage_sizes.items()
    ]).sort("Bytes")
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
                        WHERE date = DATE '2024-01-01'
                    """, n_iterations)
                })
            results.append({
                "query": "date_sensor",
                "format": "long",
                "storage": storage,
                "time_ms": time_query(f"""
                    SELECT timestamp, rpi_id, value
                    FROM {long_src}
                    WHERE date = DATE '2024-01-01' AND sensor_id = 'sensor_0'
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
def _(benchmark_results, pl):
    results_df = pl.DataFrame(benchmark_results)
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
    chart = alt.Chart(alt.Data(values=results_df.to_dicts())).mark_bar().encode(
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
def _(mo, pl, results_df):
    # Create pivot for comparison
    pivot = results_df.pivot(
        index=["query", "storage"],
        on="format",
        values="time_ms",
        aggregate_function="first",
    ).with_columns([
        (pl.col("wide") < pl.col("long")).alias("wide_faster"),
        (pl.col("long") / pl.col("wide")).alias("speedup"),
    ])

    paired = pivot.filter(
        pl.col("wide").is_not_null() & pl.col("long").is_not_null()
    )
    wide_wins = paired.filter(pl.col("wide_faster")).height
    long_wins = paired.height - wide_wins

    mo.md(f"""
    ## Summary

    **Wide format wins:** {wide_wins} queries
    **Long format wins:** {long_wins} queries

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
