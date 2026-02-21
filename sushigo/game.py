"""Sushi Go game engine for 2 players, 3 rounds. No chopsticks."""

from dataclasses import dataclass, field
import numpy as np

# Card type indices
EGG_NIGIRI = 0
SALMON_NIGIRI = 1
SQUID_NIGIRI = 2
MAKI_1 = 3
MAKI_2 = 4
MAKI_3 = 5
TEMPURA = 6
SASHIMI = 7
DUMPLING = 8
WASABI = 9
PUDDING = 10

NUM_CARD_TYPES = 11

CARD_NAMES = [
    "Egg Nigiri", "Salmon Nigiri", "Squid Nigiri",
    "Maki 1", "Maki 2", "Maki 3",
    "Tempura", "Sashimi", "Dumpling",
    "Wasabi", "Pudding",
]

# Standard deck composition (no chopsticks)
DECK_COUNTS = np.array([5, 10, 5, 6, 12, 8, 14, 14, 14, 6, 10], dtype=np.int32)

NIGIRI_TYPES = {EGG_NIGIRI, SALMON_NIGIRI, SQUID_NIGIRI}
NIGIRI_POINTS = {EGG_NIGIRI: 1, SALMON_NIGIRI: 2, SQUID_NIGIRI: 3}
MAKI_SYMBOLS = {MAKI_1: 1, MAKI_2: 2, MAKI_3: 3}
DUMPLING_SCORES = [0, 1, 3, 6, 10, 15]

CARDS_PER_HAND = 10
NUM_ROUNDS = 3


@dataclass
class PlayerState:
    collected: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CARD_TYPES, dtype=np.int32))
    pudding_total: int = 0
    unused_wasabi: int = 0
    round_score: int = 0

    def reset_round(self):
        """Keep pudding_total and round_score, clear the rest for a new round."""
        self.pudding_total += int(self.collected[PUDDING])
        self.collected = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        self.unused_wasabi = 0


def _build_deck() -> list[int]:
    """Return a list of card type ints representing the full deck."""
    deck = []
    for card_type, count in enumerate(DECK_COUNTS):
        deck.extend([card_type] * count)
    return deck


def _hand_to_counts(hand: list[int]) -> np.ndarray:
    """Convert a list of card ints to a count array of shape (NUM_CARD_TYPES,)."""
    counts = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    for card in hand:
        counts[card] += 1
    return counts


def score_round(p1: PlayerState, p2: PlayerState) -> tuple[int, int]:
    """Score a single round for both players. Modifies round_score in place."""
    s1, s2 = 0, 0

    # --- Maki rolls ---
    maki1 = sum(int(p1.collected[m]) * sym for m, sym in MAKI_SYMBOLS.items())
    maki2 = sum(int(p2.collected[m]) * sym for m, sym in MAKI_SYMBOLS.items())
    if maki1 > maki2:
        s1 += 6
        s2 += 3
    elif maki2 > maki1:
        s2 += 6
        s1 += 3
    elif maki1 > 0:  # tied and non-zero
        s1 += 3
        s2 += 3

    # --- Tempura (5 per pair) ---
    s1 += (int(p1.collected[TEMPURA]) // 2) * 5
    s2 += (int(p2.collected[TEMPURA]) // 2) * 5

    # --- Sashimi (10 per set of 3) ---
    s1 += (int(p1.collected[SASHIMI]) // 3) * 10
    s2 += (int(p2.collected[SASHIMI]) // 3) * 10

    # --- Dumplings ---
    for p, acc in [(p1, lambda v: v), (p2, lambda v: v)]:
        n = min(int(p.collected[DUMPLING]), 5)
        if p is p1:
            s1 += DUMPLING_SCORES[n]
        else:
            s2 += DUMPLING_SCORES[n]

    # --- Nigiri (with wasabi already accounted during card placement) ---
    # Nigiri points are scored during play via _place_card, stored in a running tally.
    # But we count them here from collected cards. The wasabi-boosted nigiri were
    # already handled: unused_wasabi tracks wasabi NOT yet paired.
    # We need a different approach: track nigiri_points accumulated during play.
    # Actually, let's just score nigiri simply here and handle wasabi during placement.
    # During placement, when a nigiri is placed on wasabi, we add bonus points.
    # Here we just score base nigiri points.
    for nigiri, pts in NIGIRI_POINTS.items():
        s1 += int(p1.collected[nigiri]) * pts
        s2 += int(p2.collected[nigiri]) * pts

    p1.round_score += s1
    p2.round_score += s2
    return s1, s2


def score_pudding(p1: PlayerState, p2: PlayerState) -> tuple[int, int]:
    """End-of-game pudding scoring."""
    s1, s2 = 0, 0
    pud1, pud2 = p1.pudding_total, p2.pudding_total
    if pud1 > pud2:
        s1 += 6
        s2 -= 6
    elif pud2 > pud1:
        s2 += 6
        s1 -= 6
    # tie: no change
    p1.round_score += s1
    p2.round_score += s2
    return s1, s2


class SushiGoGame:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def play(self, agent1, agent2) -> tuple[int, int]:
        """Play a full 3-round game. Returns (score1, score2)."""
        deck = _build_deck()
        self.rng.shuffle(deck)
        deck_idx = 0

        p1 = PlayerState()
        p2 = PlayerState()
        # Track wasabi bonus points separately
        wasabi_bonus = [0, 0]

        for round_num in range(NUM_ROUNDS):
            # Deal hands
            hand1 = list(deck[deck_idx:deck_idx + CARDS_PER_HAND])
            deck_idx += CARDS_PER_HAND
            hand2 = list(deck[deck_idx:deck_idx + CARDS_PER_HAND])
            deck_idx += CARDS_PER_HAND

            for turn in range(CARDS_PER_HAND):
                # Build state for each agent
                hand1_counts = _hand_to_counts(hand1)
                hand2_counts = _hand_to_counts(hand2)

                state1 = {
                    "my_collected": p1.collected.copy(),
                    "my_unused_wasabi": p1.unused_wasabi,
                    "hand": hand1_counts,
                    "opp_collected": p2.collected.copy(),
                    "opp_unused_wasabi": p2.unused_wasabi,
                    "round": round_num,
                    "cards_remaining": len(hand1),
                }
                state2 = {
                    "my_collected": p2.collected.copy(),
                    "my_unused_wasabi": p2.unused_wasabi,
                    "hand": hand2_counts,
                    "opp_collected": p1.collected.copy(),
                    "opp_unused_wasabi": p1.unused_wasabi,
                    "round": round_num,
                    "cards_remaining": len(hand2),
                }

                pick1 = agent1.pick_card(state1)
                pick2 = agent2.pick_card(state2)

                # Validate picks are in hand
                if pick1 not in hand1:
                    pick1 = hand1[0]
                if pick2 not in hand2:
                    pick2 = hand2[0]

                # Remove picked cards from hands
                hand1.remove(pick1)
                hand2.remove(pick2)

                # Place cards
                for pick, ps, bonus_idx in [(pick1, p1, 0), (pick2, p2, 1)]:
                    ps.collected[pick] += 1
                    if pick == WASABI:
                        ps.unused_wasabi += 1
                    elif pick in NIGIRI_TYPES and ps.unused_wasabi > 0:
                        # Wasabi triples the nigiri: add 2x bonus (base counted in scoring)
                        wasabi_bonus[bonus_idx] += NIGIRI_POINTS[pick] * 2
                        ps.unused_wasabi -= 1

                # Swap hands (pick-and-pass)
                hand1, hand2 = hand2, hand1

            # Score the round
            score_round(p1, p2)
            p1.round_score += wasabi_bonus[0]
            p2.round_score += wasabi_bonus[1]
            wasabi_bonus = [0, 0]

            # Reset for next round (preserves pudding_total and round_score)
            p1.reset_round()
            p2.reset_round()

        # Score pudding
        score_pudding(p1, p2)

        return p1.round_score, p2.round_score
