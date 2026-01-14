use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Find the maximum cycle length in a single permutation.
fn max_cycle_length(perm: &[usize]) -> usize {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut max_len = 0;

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut length = 0;
        let mut current = start;
        while !visited[current] {
            visited[current] = true;
            length += 1;
            current = perm[current];
        }
        if length > max_len {
            max_len = length;
        }
    }
    max_len
}

/// Run n_sims simulations and return the max cycle length for each.
#[pyfunction]
fn simulate_rust(n_prisoners: usize, n_sims: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut results = Vec::with_capacity(n_sims);

    for _ in 0..n_sims {
        // Create and shuffle permutation
        let mut perm: Vec<usize> = (0..n_prisoners).collect();
        perm.shuffle(&mut rng);

        results.push(max_cycle_length(&perm));
    }
    results
}

/// Python module
#[pymodule]
fn bench_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_rust, m)?)?;
    Ok(())
}
