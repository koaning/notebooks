"""Multiprocessing worker for Island Model GA."""
import numpy as np


def island_worker(island_id, pop_size, games_per_eval, elite_count, tournament_k,
                  mutation_rate, mutation_sigma, seed, src_dir,
                  cmd_queue, result_queue):
    """Worker process for one island. Listens for commands on cmd_queue."""
    import sys
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from ga import GeneticAlgorithm
    from agent import NeuralAgent

    ga = GeneticAlgorithm(
        pop_size=pop_size, elite_count=elite_count, tournament_k=tournament_k,
        mutation_rate=mutation_rate, mutation_sigma=mutation_sigma,
        games_per_eval=games_per_eval, seed=seed + island_id,
    )
    population = ga._make_population()
    fitness = None

    while True:
        cmd = cmd_queue.get()

        if cmd[0] == "evolve":
            n_gens = cmd[1]
            stats = []
            for _ in range(n_gens):
                fitness = ga.evaluate_fitness(population)
                stats.append({
                    "best": float(np.max(fitness)),
                    "mean": float(np.mean(fitness)),
                    "worst": float(np.min(fitness)),
                })
                population = ga.evolve_step(population, fitness)
            result_queue.put(("stats", stats))

        elif cmd[0] == "get_best":
            k = cmd[1]
            if fitness is None:
                fitness = ga.evaluate_fitness(population)
            top_idx = np.argsort(fitness)[-k:]
            chroms = [population[i].chromosome.tolist() for i in top_idx]
            result_queue.put(("best", chroms))

        elif cmd[0] == "receive_migrants":
            chromosomes = cmd[1]
            if fitness is None:
                fitness = ga.evaluate_fitness(population)
            worst_idx = np.argsort(fitness)[:len(chromosomes)]
            for idx, chrom in zip(worst_idx, chromosomes):
                population[idx] = NeuralAgent(weights=np.array(chrom), rng=ga.rng)
            fitness = None
            result_queue.put(("ack", None))

        elif cmd[0] == "get_global_best":
            if fitness is None:
                fitness = ga.evaluate_fitness(population)
            best_idx = int(np.argmax(fitness))
            result_queue.put(("global_best", {
                "chromosome": population[best_idx].chromosome.tolist(),
                "fitness": float(fitness[best_idx]),
            }))

        elif cmd[0] == "stop":
            break
