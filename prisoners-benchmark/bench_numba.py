"""Numba JIT-compiled implementation for benchmarking."""

import numpy as np
from numba import njit


@njit
def max_cycle_length_numba(perm):
    """Find the maximum cycle length in a permutation (Numba JIT)."""
    n = len(perm)
    visited = np.zeros(n, dtype=np.bool_)
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


def simulate_numba(n_prisoners: int, n_sims: int) -> list[int]:
    """Run simulations using Numba JIT-compiled function."""
    results = []
    for _ in range(n_sims):
        perm = np.random.permutation(n_prisoners).astype(np.int64)
        results.append(max_cycle_length_numba(perm))
    return results


def warmup():
    """Warm up Numba JIT compilation."""
    perm = np.random.permutation(100).astype(np.int64)
    max_cycle_length_numba(perm)


if __name__ == "__main__":
    import time

    N_PRISONERS = 100
    N_SIMS = 10_000

    print("Warming up Numba JIT...")
    warmup()

    print(f"\nBenchmarking with {N_PRISONERS} prisoners, {N_SIMS} simulations\n")

    start = time.perf_counter()
    results = simulate_numba(N_PRISONERS, N_SIMS)
    elapsed = time.perf_counter() - start
    print(f"Numba JIT: {elapsed:.3f}s for {N_SIMS} sims ({N_SIMS/elapsed:.0f} sims/sec)")
