# Prisoner Simulation Benchmarks

Performance comparison of different implementations for the [100 prisoners problem](https://en.wikipedia.org/wiki/100_prisoners_problem) simulation.

## Results

| Implementation | Speedup vs Python |
|---------------|-------------------|
| Pure Python   | 1.0x (baseline)   |
| NumPy         | 0.6x (slower)     |
| Numba         | ~4x               |
| Mojo          | ~5x               |
| Rust          | ~7x               |

## Setup

```bash
cd prisoners-benchmark
uv sync
```

### Build Rust extension (optional)

```bash
uv run maturin develop --manifest-path bench_rust/Cargo.toml --release
```

## Run benchmarks

```bash
uv run marimo edit run_benchmarks.py
```

## Files

- `run_benchmarks.py` - Interactive Marimo notebook with all benchmarks
- `bench_python.py` - Pure Python and NumPy implementations
- `bench_numba.py` - Numba JIT implementation
- `bench_mojo.mojo` - Standalone Mojo benchmark
- `bench_mojo_runner.mojo` - Mojo with CLI args for Python integration
- `bench_rust/` - Rust PyO3 extension
