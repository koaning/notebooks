import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import diskcache
    import json
    import zlib
    import tempfile
    import os
    import shutil
    import altair as alt
    import pandas as pd
    return alt, diskcache, json, mo, os, pd, shutil, tempfile, zlib


@app.cell
def _(mo):
    mo.md("""
    # DiskCache Benchmark: Pickle vs Compressed JSON

    How much disk space can you save by using compressed JSON instead of Pickle?
    """)
    return


@app.cell
def _(diskcache, json, zlib):
    class JSONDisk(diskcache.Disk):
        """Custom Disk that stores data as compressed JSON instead of Pickle."""

        def __init__(self, directory, compress_level=1, **kwargs):
            self.compress_level = compress_level
            super().__init__(directory, **kwargs)

        def put(self, key):
            json_bytes = json.dumps(key).encode('utf-8')
            data = zlib.compress(json_bytes, self.compress_level)
            return super().put(data)

        def get(self, key, raw):
            data = super().get(key, raw)
            return json.loads(zlib.decompress(data).decode('utf-8'))

        def store(self, value, read, key=diskcache.core.UNKNOWN):
            if not read:
                json_bytes = json.dumps(value).encode('utf-8')
                value = zlib.compress(json_bytes, self.compress_level)
            return super().store(value, read, key=key)

        def fetch(self, mode, filename, value, read):
            data = super().fetch(mode, filename, value, read)
            if not read:
                data = json.loads(zlib.decompress(data).decode('utf-8'))
            return data
    return (JSONDisk,)


@app.function
def generate_normal_data(n_items):
    """Normal mixed data: simple dicts with varying values."""
    return {
        f"item_{i}": {
            "id": i,
            "name": f"user_{i}",
            "score": i * 1.23,
            "active": i % 2 == 0,
            "tags": [f"tag_{j}" for j in range(i % 5)]
        }
        for i in range(n_items)
    }


@app.function
def generate_text_heavy_data(n_items):
    """Text-heavy data: long strings, descriptions, logs."""
    lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
    return {
        f"doc_{i}": {
            "title": f"Document number {i} with a longer title",
            "body": lorem + f" This is document {i}. " + lorem,
            "author": f"Author Name {i}",
            "comments": [f"Comment {j} on document {i}" for j in range(10)]
        }
        for i in range(n_items)
    }


@app.function
def generate_highly_compressible_data(n_items):
    """Highly compressible: repeated patterns, redundant structures."""
    return {
        f"record_{i}": {
            "repeated_list": ["same_value"] * 500,
            "repeated_dict": [{"x": 1, "y": 2, "z": 3}] * 100,
            "constant_string": "AAAAAAAAAA" * 100,
            "index": i
        }
        for i in range(n_items)
    }


@app.cell
def _(JSONDisk, diskcache, os, shutil, tempfile):
    def get_cache_size(cache_dir):
        """Calculate total size of cache directory in bytes."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total

    def measure_cache_size(test_data, use_json_disk=False, compress_level=6):
        """Measure cache size for given data."""
        cache_dir = tempfile.mkdtemp(prefix="cache_bench_")
        try:
            if use_json_disk:
                with diskcache.Cache(cache_dir, disk=JSONDisk, disk_compress_level=compress_level) as cache:
                    for key, value in test_data.items():
                        cache[key] = value
            else:
                with diskcache.Cache(cache_dir) as cache:
                    for key, value in test_data.items():
                        cache[key] = value
            return get_cache_size(cache_dir)
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
    return (measure_cache_size,)


@app.cell
def _(measure_cache_size, pd):
    # Run benchmark across different data sizes and types
    item_counts = [10, 20, 50, 100, 200, 500, 1000]
    results = []

    for n in item_counts:
        # Normal data
        normal_data = generate_normal_data(n)
        results.append({
            "n_items": n,
            "data_type": "Normal",
            "pickle_size": measure_cache_size(normal_data, use_json_disk=False),
            "json_size": measure_cache_size(normal_data, use_json_disk=True)
        })

        # Text-heavy data
        text_data = generate_text_heavy_data(n)
        results.append({
            "n_items": n,
            "data_type": "Text-heavy",
            "pickle_size": measure_cache_size(text_data, use_json_disk=False),
            "json_size": measure_cache_size(text_data, use_json_disk=True)
        })

        # Highly compressible data
        compressible_data = generate_highly_compressible_data(n)
        results.append({
            "n_items": n,
            "data_type": "Highly compressible",
            "pickle_size": measure_cache_size(compressible_data, use_json_disk=False),
            "json_size": measure_cache_size(compressible_data, use_json_disk=True)
        })

    benchmark_df = pd.DataFrame(results)
    benchmark_df["savings_percent"] = ((benchmark_df["pickle_size"] - benchmark_df["json_size"]) / benchmark_df["pickle_size"] * 100).round(1)
    return (benchmark_df,)


@app.cell
def _(alt, benchmark_df):
    # Chart showing savings percentage by data type
    savings_chart = alt.Chart(benchmark_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("n_items:Q", title="Number of Items"),
        y=alt.Y("savings_percent:Q", title="Space Savings (%)"),
        color=alt.Color("data_type:N", title="Data Type",
                       scale=alt.Scale(scheme="category10"))
    ).properties(
        title="Disk Space Savings: JSONDisk vs Default Pickle",
        width=600,
        height=350
    )
    savings_chart
    return


@app.cell
def _(benchmark_df, mo):

    # Summary table for latest results
    latest = benchmark_df[benchmark_df["n_items"] == benchmark_df["n_items"].max()]

    mo.md(f"""
    ## Summary (at {latest["n_items"].iloc[0]} items)

    | Data Type | Pickle | JSONDisk | Savings |
    |-----------|--------|----------|---------|
    | Normal | {latest[latest["data_type"]=="Normal"]["pickle_size"].iloc[0]:,} bytes | {latest[latest["data_type"]=="Normal"]["json_size"].iloc[0]:,} bytes | {latest[latest["data_type"]=="Normal"]["savings_percent"].iloc[0]}% |
    | Text-heavy | {latest[latest["data_type"]=="Text-heavy"]["pickle_size"].iloc[0]:,} bytes | {latest[latest["data_type"]=="Text-heavy"]["json_size"].iloc[0]:,} bytes | {latest[latest["data_type"]=="Text-heavy"]["savings_percent"].iloc[0]}% |
    | Highly compressible | {latest[latest["data_type"]=="Highly compressible"]["pickle_size"].iloc[0]:,} bytes | {latest[latest["data_type"]=="Highly compressible"]["json_size"].iloc[0]:,} bytes | {latest[latest["data_type"]=="Highly compressible"]["savings_percent"].iloc[0]}% |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Appendix: How Does the Compression Work?

    Imagine you have a book and you want to make it smaller to fit in your pocket.

    **Pickle** is like photocopying every page exactly as-is. Fast to make, but takes up the same space.

    **JSON + zlib compression** is like writing a clever summary:
    - Instead of writing "AAAA" you write "A appears 4 times"
    - Instead of copying the same paragraph 10 times, you write "repeat paragraph 10x"
    - Common patterns get short codes, rare things get longer codes

    This is why "highly compressible" data (lots of repetition) saves the most space -
    there's more redundancy for the algorithm to find and squeeze out.

    The trade-off: compression takes a bit more CPU time to pack/unpack,
    and JSONDisk only works with JSON-friendly data (no custom Python objects).
    """)
    return


if __name__ == "__main__":
    app.run()
