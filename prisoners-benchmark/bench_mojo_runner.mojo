from random import random_ui64, seed
from time import perf_counter_ns
from sys import argv


fn max_cycle_length(perm: List[Int]) -> Int:
    """Find the maximum cycle length in a permutation."""
    var n = len(perm)
    var visited = List[Bool](capacity=n)
    for _ in range(n):
        visited.append(False)

    var max_len = 0
    for start in range(n):
        if visited[start]:
            continue
        var length = 0
        var current = start
        while not visited[current]:
            visited[current] = True
            length += 1
            current = perm[current]
        if length > max_len:
            max_len = length
    return max_len


fn fisher_yates_shuffle(mut perm: List[Int]):
    """Shuffle a list in place using Fisher-Yates algorithm."""
    var n = len(perm)
    for i in range(n - 1, 0, -1):
        var j = Int(random_ui64(0, i))
        var tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp


fn simulate_mojo(n_prisoners: Int, n_sims: Int) -> List[Int]:
    """Run n_sims simulations and return max cycle lengths."""
    seed()
    var results = List[Int](capacity=n_sims)

    for _ in range(n_sims):
        var perm = List[Int](capacity=n_prisoners)
        for i in range(n_prisoners):
            perm.append(i)
        fisher_yates_shuffle(perm)
        results.append(max_cycle_length(perm))

    return results^


fn main() raises:
    # Parse command line arguments
    var args = argv()
    var n_prisoners = 100
    var n_sims = 10000

    if len(args) > 1:
        n_prisoners = atol(args[1])
    if len(args) > 2:
        n_sims = atol(args[2])

    var start = perf_counter_ns()
    var results = simulate_mojo(n_prisoners, n_sims)
    var elapsed_ns = perf_counter_ns() - start
    var elapsed_s = Float64(elapsed_ns) / 1_000_000_000.0

    # Output in parseable format: elapsed_seconds,result1,result2,...
    print(elapsed_s, end="")
    for i in range(len(results)):
        print(",", results[i], sep="", end="")
    print()
