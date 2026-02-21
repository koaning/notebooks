"""Genetic algorithm for evolving Sushi Go agents."""

import numpy as np
from game import SushiGoGame
from agent import NeuralAgent, RandomAgent


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int = 100,
        elite_count: int = 5,
        tournament_k: int = 3,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.3,
        games_per_eval: int = 20,
        seed: int | None = None,
    ):
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.tournament_k = tournament_k
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.games_per_eval = games_per_eval
        self.rng = np.random.default_rng(seed)

    def _make_population(self) -> list[NeuralAgent]:
        return [NeuralAgent(rng=self.rng) for _ in range(self.pop_size)]

    def evaluate_fitness(self, population: list[NeuralAgent]) -> np.ndarray:
        """Evaluate each agent by win rate against a random agent.
        Fitness = (wins + 0.5 * draws) / games_per_eval, i.e. a value in [0, 1]."""
        fitness = np.zeros(len(population))
        game = SushiGoGame(rng=self.rng)
        random_opponent = RandomAgent(rng=self.rng)

        for i, agent in enumerate(population):
            wins = 0.0
            for _ in range(self.games_per_eval):
                s1, s2 = game.play(agent, random_opponent)
                if s1 > s2:
                    wins += 1.0
                elif s1 == s2:
                    wins += 0.5
            fitness[i] = wins / self.games_per_eval
        return fitness

    def _tournament_select(self, population: list[NeuralAgent], fitness: np.ndarray) -> NeuralAgent:
        """Select one agent via tournament selection."""
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best = indices[np.argmax(fitness[indices])]
        return population[best]

    def evolve_step(self, population: list[NeuralAgent], fitness: np.ndarray) -> list[NeuralAgent]:
        """Produce next generation via elitism + tournament selection + crossover + mutation."""
        # Elitism: keep top agents
        elite_indices = np.argsort(fitness)[-self.elite_count:]
        new_pop = [NeuralAgent(weights=population[i].chromosome, rng=self.rng) for i in elite_indices]

        # Fill remaining slots
        while len(new_pop) < self.pop_size:
            parent1 = self._tournament_select(population, fitness)
            parent2 = self._tournament_select(population, fitness)
            child = NeuralAgent.crossover(parent1, parent2, rng=self.rng)
            child = child.mutate(rate=self.mutation_rate, sigma=self.mutation_sigma, rng=self.rng)
            new_pop.append(child)

        return new_pop

    def run(self, generations: int = 50) -> dict:
        """Run the full evolution loop. Returns history dict."""
        population = self._make_population()
        history = {"best": [], "mean": [], "worst": [], "generation": []}

        for gen in range(generations):
            fitness = self.evaluate_fitness(population)

            history["generation"].append(gen)
            history["best"].append(float(np.max(fitness)))
            history["mean"].append(float(np.mean(fitness)))
            history["worst"].append(float(np.min(fitness)))

            population = self.evolve_step(population, fitness)

        # Final evaluation
        fitness = self.evaluate_fitness(population)
        history["generation"].append(generations)
        history["best"].append(float(np.max(fitness)))
        history["mean"].append(float(np.mean(fitness)))
        history["worst"].append(float(np.min(fitness)))

        best_idx = int(np.argmax(fitness))
        return {
            "history": history,
            "best_agent": population[best_idx],
            "population": population,
            "final_fitness": fitness,
        }
