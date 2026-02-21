"""Neural network agent for Sushi Go."""

import numpy as np
from game import NUM_CARD_TYPES

INPUT_SIZE = 37  # 11 + 1 + 11 + 11 + 1 + 1 + 1
HIDDEN_SIZE = 16
OUTPUT_SIZE = NUM_CARD_TYPES  # 11

# Weight layout in flat chromosome
_W1_SIZE = INPUT_SIZE * HIDDEN_SIZE   # 592
_B1_SIZE = HIDDEN_SIZE                # 16
_W2_SIZE = HIDDEN_SIZE * OUTPUT_SIZE  # 176
_B2_SIZE = OUTPUT_SIZE                # 11
CHROMOSOME_SIZE = _W1_SIZE + _B1_SIZE + _W2_SIZE + _B2_SIZE  # 795


def _build_features(state: dict) -> np.ndarray:
    """Build a (37,) feature vector from game state dict."""
    features = np.zeros(INPUT_SIZE, dtype=np.float64)
    i = 0
    # My collected cards (11)
    features[i:i + NUM_CARD_TYPES] = state["my_collected"]
    i += NUM_CARD_TYPES
    # My unused wasabi (1)
    features[i] = state["my_unused_wasabi"]
    i += 1
    # Hand cards (11)
    features[i:i + NUM_CARD_TYPES] = state["hand"]
    i += NUM_CARD_TYPES
    # Opponent collected (11)
    features[i:i + NUM_CARD_TYPES] = state["opp_collected"]
    i += NUM_CARD_TYPES
    # Opponent unused wasabi (1)
    features[i] = state["opp_unused_wasabi"]
    i += 1
    # Round number normalized (1)
    features[i] = state["round"] / 2.0
    i += 1
    # Cards remaining normalized (1)
    features[i] = state["cards_remaining"] / 10.0
    return features


class NeuralAgent:
    """Agent whose strategy is a small feedforward neural network."""

    def __init__(self, weights: np.ndarray | None = None, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        if weights is None:
            # Xavier initialization
            self._weights = self.rng.standard_normal(CHROMOSOME_SIZE).astype(np.float64) * 0.1
        else:
            self._weights = weights.copy().astype(np.float64)
        self._unpack()

    def _unpack(self):
        """Unpack flat weight vector into matrices."""
        idx = 0
        self.W1 = self._weights[idx:idx + _W1_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
        idx += _W1_SIZE
        self.b1 = self._weights[idx:idx + _B1_SIZE]
        idx += _B1_SIZE
        self.W2 = self._weights[idx:idx + _W2_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
        idx += _W2_SIZE
        self.b2 = self._weights[idx:idx + _B2_SIZE]

    @property
    def chromosome(self) -> np.ndarray:
        return self._weights.copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: input (37,) -> output (11,)."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def pick_card(self, state: dict) -> int:
        """Pick a card type given the game state. Returns a card type int."""
        x = _build_features(state)
        scores = self.forward(x)
        hand = state["hand"]
        # Mask unavailable card types
        mask = hand > 0
        scores[~mask] = -np.inf
        return int(np.argmax(scores))

    @staticmethod
    def crossover(parent1: "NeuralAgent", parent2: "NeuralAgent",
                  rng: np.random.Generator | None = None) -> "NeuralAgent":
        """Single-point crossover on flat weight vectors."""
        rng = rng or np.random.default_rng()
        point = rng.integers(1, CHROMOSOME_SIZE)
        child_weights = np.concatenate([
            parent1.chromosome[:point],
            parent2.chromosome[point:],
        ])
        return NeuralAgent(weights=child_weights, rng=rng)

    def mutate(self, rate: float = 0.1, sigma: float = 0.3,
               rng: np.random.Generator | None = None) -> "NeuralAgent":
        """Return a mutated copy. Each weight is perturbed with probability `rate`."""
        rng = rng or self.rng
        new_weights = self.chromosome
        mask = rng.random(CHROMOSOME_SIZE) < rate
        new_weights[mask] += rng.normal(0, sigma, size=mask.sum())
        return NeuralAgent(weights=new_weights, rng=rng)


class RandomAgent:
    """Baseline agent that picks a random card from the hand."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def pick_card(self, state: dict) -> int:
        hand = state["hand"]
        available = np.where(hand > 0)[0]
        return int(self.rng.choice(available))
