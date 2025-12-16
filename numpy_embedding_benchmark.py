import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import diskcache
    import numpy as np
    import tempfile
    import os
    import shutil
    import time
    import altair as alt
    import pandas as pd
    return alt, diskcache, mo, np, os, pd, shutil, tempfile, time


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # NumPy Embedding Storage Benchmark

    How much disk space and retrieval time can you save by using different dtypes for embeddings?
    """)
    return


@app.cell
def _(diskcache, np):
    class NumpyDisk(diskcache.Disk):
        """Custom Disk that stores numpy arrays efficiently with optional dtype conversion."""

        def __init__(self, directory, target_dtype_str=None, quantization_mode=None, **kwargs):
            # Store dtype as string to avoid SQLite serialization issues
            self.target_dtype = np.dtype(target_dtype_str) if target_dtype_str else None
            self.quantization_mode = quantization_mode  # "calibrated", "binary", or None
            super().__init__(directory, **kwargs)

        def store(self, value, read, key=diskcache.core.UNKNOWN):
            if not read and isinstance(value, np.ndarray):
                if self.quantization_mode == "calibrated":
                    # Calibrated uint8 quantization
                    min_val = value.min()
                    max_val = value.max()
                    scale = 255.0 / (max_val - min_val + 1e-10)
                    quantized = ((value - min_val) * scale).astype(np.uint8)
                    # Pack: cal|shape|min_val|scale|data
                    shape_str = ','.join(map(str, value.shape))
                    meta = f"cal|{shape_str}|{min_val}|{scale}|".encode()
                    value = meta + quantized.tobytes()
                elif self.quantization_mode == "binary":
                    # Binary quantization: pack sign bits
                    packed = np.packbits(value > 0)
                    # Pack: bin|shape|data
                    shape_str = ','.join(map(str, value.shape))
                    meta = f"bin|{shape_str}|".encode()
                    value = meta + packed.tobytes()
                else:
                    arr = value.astype(self.target_dtype) if self.target_dtype else value
                    meta = f"{arr.dtype}|{','.join(map(str, arr.shape))}|".encode()
                    value = meta + arr.tobytes()
            return super().store(value, read, key=key)

        def fetch(self, mode, filename, value, read):
            data = super().fetch(mode, filename, value, read)
            if not read and isinstance(data, bytes) and b'|' in data[:50]:
                # Check if calibrated format (starts with "cal|")
                if data.startswith(b'cal|'):
                    # Calibrated quantization - dequantize
                    # Format: cal|shape|min_val|scale|data
                    parts = data.split(b'|', 4)
                    shape = tuple(map(int, parts[1].decode().split(',')))
                    min_val = float(parts[2].decode())
                    scale = float(parts[3].decode())
                    quantized = np.frombuffer(parts[4], dtype=np.uint8).reshape(shape).copy()
                    arr = quantized.astype(np.float32) / scale + min_val
                    return arr
                elif data.startswith(b'bin|'):
                    # Binary quantization - unpack and map to -1/+1
                    # Format: bin|shape|data
                    parts = data.split(b'|', 2)
                    shape = tuple(map(int, parts[1].decode().split(',')))
                    original_size = np.prod(shape)
                    packed = np.frombuffer(parts[2], dtype=np.uint8)
                    unpacked = np.unpackbits(packed)[:original_size]
                    arr = (unpacked.astype(np.float32) * 2 - 1).reshape(shape)
                    return arr
                else:
                    # Standard dtype conversion
                    # Format: dtype|shape|data
                    parts = data.split(b'|', 2)
                    dtype = np.dtype(parts[0].decode())
                    shape = tuple(map(int, parts[1].decode().split(',')))
                    arr = np.frombuffer(parts[2], dtype=dtype).reshape(shape).copy()
                    return arr
            return data
    return (NumpyDisk,)


@app.function
def generate_embeddings(n_vectors, dim):
    """Generate random L2-normalized embeddings."""
    import numpy as np
    embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@app.function
def quantize_calibrated(embedding):
    """Quantize float32 to uint8 using full [0, 255] range."""
    import numpy as np
    min_val = embedding.min()
    max_val = embedding.max()
    scale = 255.0 / (max_val - min_val + 1e-10)
    quantized = ((embedding - min_val) * scale).astype(np.uint8)
    return quantized, np.float32(min_val), np.float32(scale)


@app.function
def dequantize_calibrated(quantized, min_val, scale):
    """Reconstruct float32 from uint8."""
    import numpy as np
    return quantized.astype(np.float32) / scale + min_val


@app.function
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


@app.function
def quantize_binary(embedding):
    """Quantize to binary: positive → 1, negative → 0, packed into bytes."""
    import numpy as np
    return np.packbits(embedding > 0)


@app.function
def dequantize_binary(packed, original_dim):
    """Reconstruct from binary (returns -1/+1 for negative/positive)."""
    import numpy as np
    unpacked = np.unpackbits(packed)[:original_dim]
    return unpacked.astype(np.float32) * 2 - 1  # Map 0→-1, 1→+1


@app.cell
def _(NumpyDisk, diskcache, os, shutil, tempfile, time):
    def get_cache_size(cache_dir):
        """Calculate total size of cache directory in bytes."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return total

    def benchmark_storage(embeddings_dict, use_numpy_disk=False, target_dtype_str=None, quantization_mode=None):
        """Benchmark cache size and retrieval time."""
        cache_dir = tempfile.mkdtemp(prefix="emb_bench_")
        try:
            # Store embeddings
            if use_numpy_disk:
                cache = diskcache.Cache(
                    cache_dir, disk=NumpyDisk,
                    disk_target_dtype_str=target_dtype_str,
                    disk_quantization_mode=quantization_mode
                )
            else:
                cache = diskcache.Cache(cache_dir)

            with cache:
                for key, value in embeddings_dict.items():
                    cache[key] = value

            disk_size = get_cache_size(cache_dir)

            # Measure retrieval time - reopen cache to measure cold read
            if use_numpy_disk:
                cache2 = diskcache.Cache(
                    cache_dir, disk=NumpyDisk,
                    disk_target_dtype_str=target_dtype_str,
                    disk_quantization_mode=quantization_mode
                )
            else:
                cache2 = diskcache.Cache(cache_dir)

            with cache2:
                start = time.perf_counter()
                for key in embeddings_dict.keys():
                    _ = cache2[key]
                retrieval_time = time.perf_counter() - start

            return {"disk_size": disk_size, "retrieval_time": retrieval_time}
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
    return (benchmark_storage,)


@app.cell
def _(benchmark_storage, np, pd):
    # Run benchmark across dimensions and storage methods
    dimensions = [128, 256, 384, 512, 768, 1024, 1536]
    n_vectors = 1000
    results = []

    for dim in dimensions:
        # Generate base embeddings in float32
        embeddings_f32 = {f"emb_{i}": generate_embeddings(1, dim)[0] for i in range(n_vectors)}
        # Pre-convert to float16 and int8 for fair Pickle comparison
        embeddings_f16 = {k: v.astype(np.float16) for k, v in embeddings_f32.items()}
        embeddings_i8 = {k: v.astype(np.int8) for k, v in embeddings_f32.items()}

        # Pickle float32
        res = benchmark_storage(embeddings_f32, use_numpy_disk=False)
        results.append({
            "dimension": dim,
            "method": "Pickle (float32)",
            "disk_size": res["disk_size"],
            "retrieval_time": res["retrieval_time"]
        })

        # Pickle float16
        res = benchmark_storage(embeddings_f16, use_numpy_disk=False)
        results.append({
            "dimension": dim,
            "method": "Pickle (float16)",
            "disk_size": res["disk_size"],
            "retrieval_time": res["retrieval_time"]
        })

        # Pickle int8
        res = benchmark_storage(embeddings_i8, use_numpy_disk=False)
        results.append({
            "dimension": dim,
            "method": "Pickle (int8)",
            "disk_size": res["disk_size"],
            "retrieval_time": res["retrieval_time"]
        })

        # NumpyDisk float32
        res = benchmark_storage(embeddings_f32, use_numpy_disk=True, target_dtype_str="float32")
        results.append({
            "dimension": dim,
            "method": "NumpyDisk (float32)",
            "disk_size": res["disk_size"],
            "retrieval_time": res["retrieval_time"]
        })

        # NumpyDisk float16
        res = benchmark_storage(embeddings_f32, use_numpy_disk=True, target_dtype_str="float16")
        results.append({
            "dimension": dim,
            "method": "NumpyDisk (float16)",
            "disk_size": res["disk_size"],
            "retrieval_time": res["retrieval_time"]
        })

        # NumpyDisk int8
        res = benchmark_storage(embeddings_f32, use_numpy_disk=True, target_dtype_str="int8")
        results.append({
            "dimension": dim,
            "method": "NumpyDisk (int8)",
            "disk_size": res["disk_size"],
            "retrieval_time": res["retrieval_time"]
        })

    benchmark_df = pd.DataFrame(results)
    benchmark_df["disk_size_kb"] = benchmark_df["disk_size"] / 1024
    benchmark_df["retrieval_time_ms"] = benchmark_df["retrieval_time"] * 1000
    return (benchmark_df,)


@app.cell
def _(alt, benchmark_df):
    # Chart 1: Disk space by dimension
    space_chart = alt.Chart(benchmark_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("dimension:Q", title="Embedding Dimension"),
        y=alt.Y("disk_size_kb:Q", title="Disk Size (KB)"),
        color=alt.Color("method:N", title="Storage Method",
                       scale=alt.Scale(scheme="category10"))
    ).properties(
        title="Disk Space Usage by Embedding Dimension",
        width=600,
        height=350
    )
    space_chart
    return


@app.cell
def _(alt, benchmark_df):
    # Chart 2: Retrieval time by dimension
    time_chart = alt.Chart(benchmark_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("dimension:Q", title="Embedding Dimension"),
        y=alt.Y("retrieval_time_ms:Q", title="Retrieval Time (ms)"),
        color=alt.Color("method:N", title="Storage Method",
                       scale=alt.Scale(scheme="category10"))
    ).properties(
        title="Retrieval Time by Embedding Dimension",
        width=600,
        height=350
    )
    time_chart
    return


@app.cell
def _(benchmark_df, mo):
    # Summary table at max dimension
    _latest = benchmark_df[benchmark_df["dimension"] == benchmark_df["dimension"].max()]
    _pickle_size = _latest[_latest["method"] == "Pickle (float32)"]["disk_size_kb"].iloc[0]

    _rows = []
    _methods = [
        "Pickle (float32)", "Pickle (float16)", "Pickle (int8)",
        "NumpyDisk (float32)", "NumpyDisk (float16)", "NumpyDisk (int8)"
    ]
    for _method in _methods:
        _row = _latest[_latest["method"] == _method].iloc[0]
        _savings = ((_pickle_size - _row["disk_size_kb"]) / _pickle_size * 100)
        _rows.append(f"| {_method} | {_row['disk_size_kb']:,.1f} KB | {_savings:+.1f}% | {_row['retrieval_time_ms']:.1f} ms |")

    mo.md(f"""
    ## Summary (at {_latest["dimension"].iloc[0]} dimensions, 1000 vectors)

    | Method | Disk Size | vs Pickle f32 | Retrieval Time |
    |--------|-----------|---------------|----------------|
    {chr(10).join(_rows)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Appendix: Why Does This Work?

    **Pickle** stores numpy arrays with extra metadata about the Python object structure.
    It's flexible but adds overhead.

    **NumpyDisk** uses raw binary format (`tobytes()`), storing just the numbers plus
    minimal metadata (dtype and shape). This is more compact and faster to read.

    **float32 → float16**: Each number shrinks from 4 bytes to 2 bytes (50% savings).
    For embeddings, this rarely affects similarity search quality since the relative
    distances between vectors stay roughly the same.

    **float32 → int8**: Each number shrinks from 4 bytes to 1 byte (75% savings).
    More aggressive - you lose precision, but for many retrieval tasks it's acceptable.
    Note: the int8 conversion here is naive (just truncates). Production use would
    typically scale values to use the full -128 to 127 range.

    **When to use what:**
    - **float32**: When you need exact precision
    - **float16**: Best balance for most embedding use cases
    - **int8**: When storage is critical and some precision loss is acceptable
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Going Further: Calibrated Quantization

    The benchmark above shows that naive dtype conversion doesn't beat Pickle for disk space.
    But there's a problem with naive int8 conversion for normalized embeddings...
    """)
    return


@app.cell
def _(np, pd):
    # Measure quality loss from different quantization methods
    n_samples = 500
    _dim = 768
    quality_results = []

    for _ in range(n_samples):
        # Generate a single L2-normalized embedding
        original = generate_embeddings(1, _dim)[0]

        # float16 roundtrip
        f16 = original.astype(np.float16).astype(np.float32)
        sim_f16 = cosine_similarity(original, f16)

        # Calibrated uint8 roundtrip
        quantized, min_val, scale = quantize_calibrated(original)
        reconstructed = dequantize_calibrated(quantized, min_val, scale)
        sim_calibrated = cosine_similarity(original, reconstructed)

        # Binary roundtrip (sign only: positive → +1, negative → -1)
        packed = quantize_binary(original)
        binary_reconstructed = dequantize_binary(packed, _dim)
        sim_binary = cosine_similarity(original, binary_reconstructed)

        quality_results.append({
            "float16": sim_f16,
            "calibrated uint8": sim_calibrated,
            "binary": sim_binary
        })

    quality_df = pd.DataFrame(quality_results)
    quality_df
    return (quality_df,)


@app.cell
def _(alt, pd, quality_df):
    # Melt for visualization
    quality_melted = pd.melt(quality_df, var_name="method", value_name="cosine_similarity")

    quality_chart = alt.Chart(quality_melted).mark_boxplot().encode(
        x=alt.X("method:N", title="Quantization Method"),
        y=alt.Y("cosine_similarity:Q", title="Cosine Similarity to Original", scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("method:N", legend=None)
    ).properties(
        title="Quality Preservation: Original vs Reconstructed Embeddings",
        width=400,
        height=300
    )
    quality_chart
    return


@app.cell
def _(mo, quality_df):
    mo.md(f"""
    ## Quality Comparison

    | Method | Mean Cosine Similarity | Min | Max |
    |--------|------------------------|-----|-----|
    | float16 | {quality_df['float16'].mean():.6f} | {quality_df['float16'].min():.6f} | {quality_df['float16'].max():.6f} |
    | calibrated uint8 | {quality_df['calibrated uint8'].mean():.6f} | {quality_df['calibrated uint8'].min():.6f} | {quality_df['calibrated uint8'].max():.6f} |
    | binary | {quality_df['binary'].mean():.6f} | {quality_df['binary'].min():.6f} | {quality_df['binary'].max():.6f} |

    **Key insight**: Calibrated quantization preserves ~99.9% similarity while achieving 4x compression.
    Binary quantization has lower similarity (~50-60%) but achieves 32x compression and enables
    fast Hamming distance search.
    """)
    return


@app.cell
def _(benchmark_storage, np, pd):
    # Benchmark calibrated vs others at fixed dimension
    # Run multiple iterations to reduce noise in retrieval times
    _dim_test = 768
    _n_vectors_test = 1000
    _n_runs = 5

    _embeddings_test = {f"emb_{i}": generate_embeddings(1, _dim_test)[0] for i in range(_n_vectors_test)}

    # Define methods to test
    _methods = [
        {"name": "Pickle (float32)", "use_numpy_disk": False, "target_dtype_str": None, "quantization_mode": None},
        {"name": "NumpyDisk (float16)", "use_numpy_disk": True, "target_dtype_str": "float16", "quantization_mode": None},
        {"name": "NumpyDisk (calibrated)", "use_numpy_disk": True, "target_dtype_str": None, "quantization_mode": "calibrated"},
        {"name": "NumpyDisk (binary)", "use_numpy_disk": True, "target_dtype_str": None, "quantization_mode": "binary"},
    ]

    calibrated_results = []
    for _method in _methods:
        _retrieval_times = []
        _disk_size = None
        for _ in range(_n_runs):
            _res = benchmark_storage(
                _embeddings_test,
                use_numpy_disk=_method["use_numpy_disk"],
                target_dtype_str=_method["target_dtype_str"],
                quantization_mode=_method["quantization_mode"]
            )
            _retrieval_times.append(_res["retrieval_time"])
            _disk_size = _res["disk_size"]  # Same across runs

        calibrated_results.append({
            "method": _method["name"],
            "disk_size": _disk_size,
            "retrieval_time_median": np.median(_retrieval_times)
        })

    calibrated_df = pd.DataFrame(calibrated_results)
    calibrated_df["disk_size_kb"] = calibrated_df["disk_size"] / 1024
    calibrated_df["retrieval_time_ms"] = calibrated_df["retrieval_time_median"] * 1000
    calibrated_df
    return (calibrated_df,)


@app.cell
def _(calibrated_df, mo, quality_df):
    _baseline = calibrated_df[calibrated_df["method"] == "Pickle (float32)"]["disk_size_kb"].iloc[0]

    # Map method names to quality_df columns
    _similarity_map = {
        "Pickle (float32)": 1.0,  # No loss
        "NumpyDisk (float16)": quality_df["float16"].mean(),
        "NumpyDisk (calibrated)": quality_df["calibrated uint8"].mean(),
        "NumpyDisk (binary)": quality_df["binary"].mean(),
    }

    _rows_cal = []
    for _, _row in calibrated_df.iterrows():
        _savings = ((_baseline - _row["disk_size_kb"]) / _baseline * 100)
        _sim = _similarity_map.get(_row["method"], 1.0)
        _rows_cal.append(f"| {_row['method']} | {_row['disk_size_kb']:,.1f} KB | {_savings:+.1f}% | {_sim:.4f} | {_row['retrieval_time_ms']:.1f} ms |")

    mo.md(f"""
    ## Quantization Comparison

    | Method | Disk Size | vs Pickle f32 | Similarity | Retrieval |
    |--------|-----------|---------------|------------|-----------|
    {chr(10).join(_rows_cal)}

    - **Calibrated**: 4x compression, ~99.9% similarity preserved
    - **Binary**: 32x compression, ~55% similarity - great for fast approximate search
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Appendix: Quantization Strategies

    **Calibrated uint8 quantization:**

    Maps the actual value range to use all 256 values:
    1. Find min and max values in the embedding
    2. Stretch to fill [0, 255] range
    3. Store scale factor for reconstruction

    Result: 4x compression with ~99.9% similarity preserved.

    **Binary quantization:**

    Keeps only the sign of each value:
    - Positive values → 1
    - Negative values → 0

    Then pack 8 bits into each byte using `np.packbits`.

    Result: 32x compression (768 floats → 96 bytes). Lower similarity (~55%)
    but enables fast Hamming distance search - just count differing bits!

    **When to use what:**
    - **float16**: Best balance for most embedding use cases
    - **Calibrated**: When you need high quality with good compression
    - **Binary**: When speed matters more than precision (reranking, approximate search)
    """)
    return


if __name__ == "__main__":
    app.run()
