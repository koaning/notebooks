from random import seed, shuffle
from python import PythonObject, Python
from python.bindings import PythonModuleBuilder


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


fn simulate_mojo_internal(n_prisoners: Int, n_sims: Int) -> List[Int]:
    """Run n_sims simulations and return max cycle lengths."""
    var results = List[Int](capacity=n_sims)

    for _ in range(n_sims):
        var perm = List[Int](capacity=n_prisoners)
        for i in range(n_prisoners):
            perm.append(i)
        shuffle(perm)  # Use stdlib shuffle
        results.append(max_cycle_length(perm))

    return results^


fn simulate_mojo_wrapper(
    n_prisoners_obj: PythonObject,
    n_sims_obj: PythonObject
) raises -> PythonObject:
    """Python-callable wrapper that converts types and returns Python list."""
    var n_prisoners = Int(n_prisoners_obj)
    var n_sims = Int(n_sims_obj)

    var mojo_results = simulate_mojo_internal(n_prisoners, n_sims)

    var py_list = Python.list()
    for i in range(len(mojo_results)):
        py_list.append(mojo_results[i])

    return py_list


@export
fn PyInit_bench_mojo() -> PythonObject:
    """Module initialization function required by Python's import system."""
    seed()  # Initialize RNG once at module load time
    try:
        var m = PythonModuleBuilder("bench_mojo")
        m.def_function[simulate_mojo_wrapper](
            "simulate_mojo",
            docstring="Run prisoner simulation. Args: n_prisoners (int), n_sims (int). Returns: list[int]"
        )
        return m.finalize()
    except:
        return PythonObject()
