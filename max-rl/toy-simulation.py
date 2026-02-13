# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.8",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MaxRL vs GRPO: Toy Simulation

    A pure-simulation comparison of two advantage estimation methods from the
    [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710) paper (Tajwar et al., 2026).

    No LLM needed — we simulate binary reward rollouts (like coin flips) at various success rates
    to show how each method scales the learning signal.

    | Method | Advantage | Scaling factor |
    |--------|-----------|----------------|
    | **MaxRL** | $(r - \mu) / \mu$ | $1/\mu$ — large when success is rare |
    | **GRPO** | $(r - \mu) / \sigma$ | $1/\sigma$ — collapses when $\sigma \approx 0$ |
    """)
    return


@app.cell
def _(np):
    def maxrl_advantages(rewards):
        """(r - mu) / mu"""
        mu = rewards.mean(axis=1, keepdims=True)
        return (rewards - mu) / (mu + 1e-8)

    def grpo_advantages(rewards):
        """(r - mu) / sigma"""
        mu = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True)
        return (rewards - mu) / (std + 1e-8)

    def run_simulation(G, n_repeats, probabilities):
        """Simulate binary rollouts and compute advantage magnitudes for both methods."""
        results = {"p": probabilities, "maxrl_signal": [], "grpo_signal": []}

        for p in probabilities:
            # Shape: (n_repeats, G) — each row is one batch of G rollouts
            rewards = (np.random.rand(n_repeats, G) < p).astype(np.float32)

            maxrl_adv = maxrl_advantages(rewards)
            grpo_adv = grpo_advantages(rewards)

            # Mean absolute advantage across rollouts, then average over repeats
            results["maxrl_signal"].append(np.abs(maxrl_adv).mean())
            results["grpo_signal"].append(np.abs(grpo_adv).mean())

        results["maxrl_signal"] = np.array(results["maxrl_signal"])
        results["grpo_signal"] = np.array(results["grpo_signal"])
        return results

    return grpo_advantages, maxrl_advantages, run_simulation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Theoretical scaling factors

    For binary rewards with success probability $p$, the population statistics are $\mu = p$ and $\sigma = \sqrt{p(1-p)}$.
    The scaling factor each method applies to the advantage is:

    - **MaxRL**: $1/\mu = 1/p$
    - **GRPO**: $1/\sigma = 1/\sqrt{p(1-p)}$
    """)
    return


@app.cell
def _(np, plt):
    p_theory = np.linspace(0.01, 0.99, 200)
    inv_mu = 1 / p_theory
    inv_sigma = 1 / np.sqrt(p_theory * (1 - p_theory))

    fig_theory, ax_theory = plt.subplots(figsize=(8, 4))
    ax_theory.plot(p_theory, inv_mu, label=r"MaxRL: $1/\mu$", linewidth=2)
    ax_theory.plot(p_theory, inv_sigma, label=r"GRPO: $1/\sigma$", linewidth=2, linestyle="--")
    ax_theory.set_xlabel("Success probability (p)")
    ax_theory.set_ylabel("Scaling factor")
    ax_theory.set_title("Advantage scaling factor vs success rate")
    ax_theory.legend()
    ax_theory.set_ylim(0, 40)
    ax_theory.axvspan(0, 0.15, alpha=0.1, color="red", label="Sparse reward regime")
    ax_theory.legend()
    fig_theory.tight_layout()
    fig_theory
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Simulated gradient signal

    We simulate `G` binary rollouts at each success rate, compute advantages under both methods,
    and measure the mean absolute advantage (a proxy for gradient signal strength).
    Averaged over many Monte Carlo repeats.
    """)
    return


@app.cell
def _(group_size_slider, n_repeats_slider, np, run_simulation):
    G = group_size_slider.value
    n_repeats = n_repeats_slider.value
    probabilities = np.linspace(0.02, 0.98, 100)

    sim = run_simulation(G, n_repeats, probabilities)
    return G, n_repeats, sim


@app.cell
def _(mo):
    group_size_slider = mo.ui.slider(
        start=1, stop=64, step=1, value=16, label="Group size (G)"
    )
    n_repeats_slider = mo.ui.slider(
        start=100, stop=2000, step=100, value=500, label="Monte Carlo repeats"
    )
    mo.hstack([group_size_slider, n_repeats_slider])
    return group_size_slider, n_repeats_slider


@app.cell
def _(G, n_repeats, plt, sim):
    fig_sim, ax_sim = plt.subplots(figsize=(8, 4))
    ax_sim.plot(sim["p"], sim["maxrl_signal"], label="MaxRL", linewidth=2)
    ax_sim.plot(sim["p"], sim["grpo_signal"], label="GRPO", linewidth=2, linestyle="--")
    ax_sim.set_xlabel("Success probability (p)")
    ax_sim.set_ylabel("Mean |advantage|")
    ax_sim.set_title(f"Simulated gradient signal (G={G}, {n_repeats} repeats)")
    ax_sim.legend()
    ax_sim.axvspan(0, 0.15, alpha=0.1, color="red")
    fig_sim.tight_layout()
    fig_sim
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Gradient descent convergence race

    A toy "training loop" where a single parameter $\theta$ controls success probability via
    $p = \text{sigmoid}(\theta)$. At each step we sample $G$ binary rollouts, compute advantages,
    and update $\theta$ with REINFORCE. Starting from low $p$ (sparse rewards), does MaxRL converge faster?

    The policy gradient for a Bernoulli policy is:

    $$\nabla_\theta \approx \frac{1}{G} \sum_i A_i \cdot (r_i - p)$$

    where $A_i$ is the advantage for rollout $i$.
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.01, value=0.2, label="Learning rate"
    )
    init_theta_slider = mo.ui.slider(
        start=-5.0, stop=-1.0, step=0.01, value=-3.0, label="Initial theta (lower = harder start)"
    )
    difficulty_slider = mo.ui.slider(
        start=0.5, stop=3, step=0.01, value=1.0, label="Difficulty (higher = harder)"
    )
    n_steps_slider = mo.ui.slider(
        start=50, stop=1000, step=50, value=1000, label="Training steps"
    )
    [lr_slider, init_theta_slider, difficulty_slider, n_steps_slider]
    return difficulty_slider, init_theta_slider, lr_slider, n_steps_slider


@app.cell(hide_code=True)
def _(
    G,
    difficulty_slider,
    grpo_advantages,
    init_theta_slider,
    lr_slider,
    maxrl_advantages,
    n_steps_slider,
    np,
):
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def run_training(method_fn, lr, init_theta, n_steps, group_size, seed=42, difficulty=1):
        rng = np.random.default_rng(seed)
        theta = init_theta
        history = {"p": [], "theta": []}

        for _ in range(n_steps):
            p = sigmoid(theta)
            history["p"].append(p)
            history["theta"].append(theta)

            # Sample G binary rollouts
            rewards = rng.random((1, group_size)) < p/difficulty
            rewards = rewards.astype(np.float32)

            # Skip update if no successes (no signal)
            if rewards.sum() == 0:
                continue

            advantages = method_fn(rewards).flatten()
            # REINFORCE gradient for Bernoulli: advantage * (r - p)
            score = rewards.flatten() - p
            grad = (advantages * score).mean()
            theta += lr * grad

        history["p"].append(sigmoid(theta))
        history["theta"].append(theta)
        return history

    lr_val = lr_slider.value
    init_theta_val = init_theta_slider.value
    n_steps_val = n_steps_slider.value

    hist_maxrl = run_training(maxrl_advantages, lr_val, init_theta_val, n_steps_val, G, difficulty=difficulty_slider.value)
    hist_grpo = run_training(grpo_advantages, lr_val, init_theta_val, n_steps_val, G, difficulty=difficulty_slider.value)
    return hist_grpo, hist_maxrl, init_theta_val, lr_val


@app.cell(hide_code=True)
def _(G, hist_grpo, hist_maxrl, init_theta_val, lr_val, np, plt):
    fig_train, (ax_p, ax_theta) = plt.subplots(1, 2, figsize=(12, 4))

    ax_p.plot(hist_maxrl["p"], label="MaxRL", linewidth=2)
    ax_p.plot(hist_grpo["p"], label="GRPO", linewidth=2, linestyle="--")
    ax_p.set_xlabel("Step")
    ax_p.set_ylabel("Success probability (p)")
    ax_p.set_title(f"Convergence (G={G}, lr={lr_val}, init p={1/(1+np.exp(-init_theta_val)):.3f})")
    ax_p.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.legend()

    ax_theta.plot(hist_maxrl["theta"], label="MaxRL", linewidth=2)
    ax_theta.plot(hist_grpo["theta"], label="GRPO", linewidth=2, linestyle="--")
    ax_theta.set_xlabel("Step")
    ax_theta.set_ylabel(r"$\theta$")
    ax_theta.set_title(r"Parameter $\theta$ over training")
    ax_theta.legend()

    fig_train.tight_layout()
    fig_train
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaway

    Starting from **sparse rewards** (low initial $p$), MaxRL's $1/\mu$ scaling produces
    a stronger gradient signal and converges faster than GRPO's $1/\sigma$.

    This is exactly the regime that matters most in RL for LLMs — early in training
    when the model rarely produces correct answers. As $p$ increases, both methods
    converge in behavior since the scaling factors become similar.
    """)
    return


if __name__ == "__main__":
    app.run()
