"""Baseline Python implementation for benchmarking."""

import random
import numpy as np
import time


def max_cycle_length_python(perm):
    """Find the maximum cycle length in a permutation (pure Python)."""
    n = len(perm)
    visited = [False] * n
    max_len = 0
    for start in range(n):
        if visited[start]:
            continue
        length = 0
        current = start
        while not visited[current]:
            visited[current] = True
            length += 1
            current = perm[current]
        if length > max_len:
            max_len = length
    return max_len


def max_cycle_length_numpy(perm):
    """Find the maximum cycle length using NumPy arrays."""
    n = len(perm)
    visited = np.zeros(n, dtype=bool)
    max_len = 0
    for start in range(n):
        if visited[start]:
            continue
        length = 0
        current = start
        while not visited[current]:
            visited[current] = True
            length += 1
            current = perm[current]
        if length > max_len:
            max_len = length
    return max_len


def simulate_python(n_prisoners: int, n_sims: int) -> list[int]:
    """Run simulations using pure Python."""
    results = []
    for _ in range(n_sims):
        perm = np.random.permutation(n_prisoners)
        results.append(max_cycle_length_python(perm))
    return results


def simulate_numpy(n_prisoners: int, n_sims: int) -> list[int]:
    """Run simulations using NumPy arrays."""
    results = []
    for _ in range(n_sims):
        perm = np.random.permutation(n_prisoners)
        results.append(max_cycle_length_numpy(perm))
    return results


def benchmark(func, n_prisoners: int, n_sims: int, name: str) -> float:
    """Benchmark a simulation function and return elapsed time."""
    start = time.perf_counter()
    results = func(n_prisoners, n_sims)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s for {n_sims} sims ({n_sims/elapsed:.0f} sims/sec)")
    return elapsed


if __name__ == "__main__":
    N_PRISONERS = 100
    N_SIMS = 10_000  # Use 10k for quick benchmarks, 100k for full test

    print(f"Benchmarking with {N_PRISONERS} prisoners, {N_SIMS} simulations\n")

    benchmark(simulate_python, N_PRISONERS, N_SIMS, "Pure Python")
    benchmark(simulate_numpy, N_PRISONERS, N_SIMS, "NumPy arrays")
