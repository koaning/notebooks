# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "altair",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import polars as pl

    return alt, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sushi Go: Evolving Neural Network Players

    This notebook evolves [Sushi Go](https://boardgamegeek.com/boardgame/133473/sushi-go) players
    using a **genetic algorithm**. Each agent's strategy is a small feedforward neural network
    (37 inputs → 16 hidden → 11 outputs = **795 weights**).

    The input features capture: cards collected so far, what's in the current hand, opponent's
    visible cards, wasabi state, round number, and cards remaining. The output is a preference
    score for each card type — the agent picks the highest-scoring available card.

    The GA evolves these weight vectors through tournament selection, crossover, and mutation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pop_size = mo.ui.slider(20, 200, value=80, step=10, label="Population size")
    generations = mo.ui.slider(10, 200, value=50, step=10, label="Generations")
    mutation_rate = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="Mutation rate")
    mutation_sigma = mo.ui.slider(0.05, 1.0, value=0.3, step=0.05, label="Mutation sigma")
    games_per_eval = mo.ui.slider(5, 50, value=20, step=5, label="Games per evaluation")
    seed_input = mo.ui.number(value=42, start=0, stop=9999, label="Random seed")

    controls = mo.vstack([
        mo.md("## Settings"),
        mo.hstack([pop_size, generations, seed_input], justify="start"),
        mo.hstack([mutation_rate, mutation_sigma, games_per_eval], justify="start"),
    ])
    controls
    return (
        games_per_eval,
        generations,
        mutation_rate,
        mutation_sigma,
        pop_size,
        seed_input,
    )


@app.cell
def _(generations, mo, pop_size, seed_input):
    run_btn = mo.ui.run_button(label="Evolve!")
    mo.hstack([
        run_btn,
        mo.md(f"Will run **{generations.value}** generations with **{pop_size.value}** agents "
              f"(seed={seed_input.value})"),
    ], justify="start")
    return (run_btn,)


@app.cell
def _(
    games_per_eval,
    generations,
    mo,
    mutation_rate,
    mutation_sigma,
    pop_size,
    run_btn,
    seed_input,
):
    mo.stop(not run_btn.value, mo.md("*Click 'Evolve!' to start the genetic algorithm.*"))

    from ga import GeneticAlgorithm

    ga = GeneticAlgorithm(
        pop_size=pop_size.value,
        elite_count=max(1, pop_size.value // 20),
        tournament_k=3,
        mutation_rate=mutation_rate.value,
        mutation_sigma=mutation_sigma.value,
        games_per_eval=games_per_eval.value,
        seed=seed_input.value,
    )
    result = ga.run(generations=generations.value)
    best_agent = result["best_agent"]
    history = result["history"]
    result
    return best_agent, history


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fitness Over Generations
    """)
    return


@app.cell
def _(alt, history, pl):
    df = pl.DataFrame({
        "generation": history["generation"],
        "best": history["best"],
        "mean": history["mean"],
        "worst": history["worst"],
    }).unpivot(
        index="generation",
        on=["best", "mean", "worst"],
        variable_name="metric",
        value_name="fitness",
    )

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("generation:Q", title="Generation"),
            y=alt.Y("fitness:Q", title="Win rate vs random agent"),
            color=alt.Color("metric:N", title="Metric"),
        )
        .properties(width=600, height=350)
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Best Agent vs Random Agent

    Let's play some games between the evolved best agent and a random baseline to see how much the GA learned.
    """)
    return


@app.cell
def _(best_agent, mo, np, pl):
    from game import SushiGoGame
    from agent import RandomAgent

    rng = np.random.default_rng(0)
    game = SushiGoGame(rng=rng)
    random_agent = RandomAgent(rng=rng)

    n_games = 100
    wins, losses, draws = 0, 0, 0
    score_diffs = []
    for _ in range(n_games):
        s1, s2 = game.play(best_agent, random_agent)
        score_diffs.append(s1 - s2)
        if s1 > s2:
            wins += 1
        elif s1 < s2:
            losses += 1
        else:
            draws += 1

    summary_df = pl.DataFrame({
        "Result": ["Wins", "Losses", "Draws"],
        "Count": [wins, losses, draws],
        "Percentage": [f"{100*wins/n_games:.0f}%", f"{100*losses/n_games:.0f}%", f"{100*draws/n_games:.0f}%"],
    })

    mo.vstack([
        mo.md(f"Over **{n_games}** games against a random agent:"),
        summary_df,
        mo.md(f"Average score difference: **{np.mean(score_diffs):.1f}** points"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agent Card Preferences

    What does the evolved agent prefer to pick in different situations?
    Below we show the neural network's raw output scores for each card type given a few
    representative board states (empty board at start, and with some cards already collected).
    """)
    return


@app.cell
def _(alt, best_agent, np, pl):
    from game import CARD_NAMES, NUM_CARD_TYPES

    def _get_preferences(agent, collected=None, hand=None, round_num=0, cards_remaining=10):
        """Get raw preference scores from the agent's NN for a given state."""
        if collected is None:
            collected = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
        if hand is None:
            hand = np.ones(NUM_CARD_TYPES, dtype=np.int32)
        state = {
            "my_collected": collected,
            "my_unused_wasabi": 0,
            "hand": hand,
            "opp_collected": np.zeros(NUM_CARD_TYPES, dtype=np.int32),
            "opp_unused_wasabi": 0,
            "round": round_num,
            "cards_remaining": cards_remaining,
        }
        from agent import _build_features
        x = _build_features(state)
        return agent.forward(x)

    # Scenario 1: Empty board, all cards available
    scores_empty = _get_preferences(best_agent)
    # Scenario 2: Have 1 tempura already
    collected_1t = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    collected_1t[6] = 1  # TEMPURA
    scores_1tempura = _get_preferences(best_agent, collected=collected_1t)
    # Scenario 3: Have 2 sashimi already
    collected_2s = np.zeros(NUM_CARD_TYPES, dtype=np.int32)
    collected_2s[7] = 2  # SASHIMI
    scores_2sashimi = _get_preferences(best_agent, collected=collected_2s)

    pref_df = pl.DataFrame({
        "card": CARD_NAMES * 3,
        "score": list(scores_empty) + list(scores_1tempura) + list(scores_2sashimi),
        "scenario": (["Empty board"] * NUM_CARD_TYPES
                     + ["Has 1 Tempura"] * NUM_CARD_TYPES
                     + ["Has 2 Sashimi"] * NUM_CARD_TYPES),
    })

    pref_chart = (
        alt.Chart(pref_df)
        .mark_bar()
        .encode(
            x=alt.X("card:N", title="Card Type", sort=CARD_NAMES),
            y=alt.Y("score:Q", title="Preference Score"),
            color=alt.Color("scenario:N", title="Scenario"),
            xOffset="scenario:N",
        )
        .properties(width=650, height=350)
    )
    pref_chart
    return


if __name__ == "__main__":
    app.run()
