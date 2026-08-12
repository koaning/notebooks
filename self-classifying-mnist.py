# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo",
#     "jax==0.10.2",
#     "jaxlib==0.10.2",
#     # For a GPU box (e.g. molab): comment out the two lines above and use instead:
#     # "jax[cuda12]==0.10.2",
#     "flax==0.12.7",
#     "optax==0.2.8",
#     "numpy==2.5.1",
#     "matplotlib==3.11.0",
#     "pillow==12.3.0",
#     "scikit-learn==1.9.0",
#     "wigglystuff==0.5.14",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    import numpy as np
    import matplotlib.pyplot as plt
    from functools import partial
    from sklearn.datasets import fetch_openml

    return fetch_openml, jax, jnp, nnx, np, optax, partial, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Self-Classifying MNIST Digits

    A [Neural Cellular Automaton](https://distill.pub/2020/selforg/mnist/): every
    pixel is an agent that only sees its 3×3 neighbourhood. Starting from a drawn
    digit, cells repeatedly update a small state vector and must *collectively* agree
    on which digit they form — with no global view of the shape.

    Each cell carries a **20-channel** state:

    - **1** immutable channel holding the pixel intensity (the drawn digit),
    - **9** hidden channels for local communication,
    - **10** output channels — a per-cell vote over digit classes 0–9.

    The update rule is a tiny (<25k-param) network applied identically at every cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data
    """)
    return


@app.cell
def _(fetch_openml, np):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    images = mnist.data.astype("float32").reshape(-1, 28, 28) / 255.0
    labels = mnist.target.astype(int)

    # A small, class-balanced training subset keeps in-notebook training snappy.
    rng = np.random.default_rng(0)
    per_class = 400
    idx = np.concatenate(
        [rng.choice(np.where(labels == d)[0], per_class, replace=False) for d in range(10)]
    )
    rng.shuffle(idx)
    train_images, train_labels = images[idx], labels[idx]
    return train_images, train_labels


@app.cell(hide_code=True)
def _(np, plt, train_images, train_labels):
    fig, axes = plt.subplots(2, 5, figsize=(6, 2.6))
    for d, ax in enumerate(axes.flat):
        sample = train_images[np.where(train_labels == d)[0][0]]
        ax.imshow(sample, cmap="gray_r")  # black ink on white
        ax.set_title(str(d), fontsize=9)
        ax.axis("off")
    fig.suptitle("One sample per class", fontsize=10)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model

    Every cell runs the **same** tiny network on its 3×3 neighbourhood:

    1. a trainable **3×3 convolution** — how a cell *perceives* its neighbours,
    2. two **1×1 convolutions** (a per-cell MLP) producing a state update `dx`.

    The final layer is **zero-initialised**, so before training every cell is a
    no-op — the digit just sits there. Training teaches cells to nudge their
    9 hidden + 10 vote channels until neighbours agree. The pixel channel is
    **immutable** (clamped back each step) and dead cells (no ink) stay dead.
    """)
    return


@app.cell(hide_code=True)
def _(jax, mo, nnx):
    class UpdateRule(nnx.Module):
        """Per-cell update: 3x3 perception conv + a 1x1 MLP producing dx."""

        def __init__(self, rngs, perception_dim=48, hidden_dim=80):
            self.perceive = nnx.Conv(
                CHANNELS, perception_dim, (3, 3), padding="SAME", rngs=rngs
            )
            self.hidden = nnx.Conv(perception_dim, hidden_dim, (1, 1), rngs=rngs)
            self.update = nnx.Conv(
                hidden_dim,
                CHANNELS,
                (1, 1),
                kernel_init=nnx.initializers.zeros,  # zero-init -> starts as a no-op
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )

        def __call__(self, state):
            return self.update(nnx.relu(self.hidden(self.perceive(state))))

    CHANNELS, N_CLASS = 20, 10  # 1 pixel + 9 hidden + 10 class votes
    model = UpdateRule(nnx.Rngs(0))
    graphdef, init_params = nnx.split(model, nnx.Param)  # functional split for jit/scan
    n_params = sum(x.size for x in jax.tree.leaves(init_params))
    mo.md(f"**{n_params:,} parameters** — comfortably under the paper's 25k budget.")
    model
    return CHANNELS, N_CLASS, graphdef, init_params


@app.cell(hide_code=True)
def _(CHANNELS, graphdef, jax, jnp, nnx, partial):
    def apply_model(params, state):
        """Reconstruct the module from its params and run one forward pass."""
        return nnx.merge(graphdef, params)(state)

    def to_state(imgs):
        """(N,28,28) ink image -> (N,28,28,20) CA state with pixel in channel 0."""
        return jnp.zeros(imgs.shape + (CHANNELS,)).at[..., 0].set(imgs)

    def living_mask(state):
        return state[..., 0:1] > 0.1  # a cell is alive where the digit has ink

    def ca_step(params, state, key, fire_rate=0.5):
        pixel = state[..., 0:1]
        dx = apply_model(params, state)
        fire = jax.random.uniform(key, dx.shape[:-1] + (1,)) < fire_rate  # async update
        state = jnp.concatenate(
            [pixel, (state + dx * fire)[..., 1:]], axis=-1
        )  # pixel immutable
        return state * living_mask(state)  # dead cells stay dead

    @partial(jax.jit, static_argnums=(3,))
    def rollout(params, state, key, steps):
        def body(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            return (ca_step(params, state, sub), key), None

        (state, _), _ = jax.lax.scan(body, (state, key), None, length=steps)
        return state

    return ca_step, living_mask, to_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    A **sample pool**: seed states from random digits, run the CA for a few steps,
    score the per-cell votes, backprop, write states back. Loss is **L2** between
    each living cell's 10-vote and the one-hot label; small Gaussian **noise** each
    step keeps votes bounded (the paper's fix for flicker).
    """)
    return


@app.cell(hide_code=True)
def _(N_CLASS, ca_step, jax, living_mask, optax):
    STEPS, BATCH, POOL, NOISE, LR = 16, 32, 512, 0.02, 2e-3
    optimizer = optax.adam(LR)

    def loss_fn(params, state0, target, key):
        def body(carry, _):
            st, k = carry
            k, s1, s2 = jax.random.split(k, 3)
            st = ca_step(params, st, s1)
            noise = (NOISE * jax.random.normal(s2, st.shape)).at[..., 0].set(0.0)
            st = (st + noise) * living_mask(st)
            return (st, k), None

        (st, _), _ = jax.lax.scan(body, (state0, key), None, length=STEPS)
        logits = st[..., -N_CLASS:]
        live = living_mask(st)
        loss = ((logits - target[:, None, None, :]) ** 2 * live).sum() / (
            live.sum() * N_CLASS + 1e-8
        )
        return loss, st

    @jax.jit
    def train_step(params, opt_state, state0, target, key):
        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state0, target, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, new_state

    return BATCH, POOL, optimizer, train_step


@app.cell(hide_code=True)
def _(mo):
    n_iters = mo.ui.slider(50, 3000, value=200, step=50, label="iterations", show_value=True)
    train_button = mo.ui.run_button(label="Train the CA")
    mo.hstack([n_iters, train_button], justify="start", gap=1)
    return n_iters, train_button


@app.cell(hide_code=True)
def _(
    BATCH,
    N_CLASS,
    POOL,
    init_params,
    jax,
    jnp,
    mo,
    n_iters,
    np,
    optimizer,
    to_state,
    train_button,
    train_images,
    train_labels,
    train_step,
):
    mo.stop(
        not train_button.value,
        mo.md(
            "▶️ Press **Train the CA** to start. (~1.7 s/iter on this CPU; use a GPU for long runs.)"
        ),
    )

    pool_rng = np.random.default_rng(0)
    seed_ix = pool_rng.integers(0, len(train_images), POOL)
    pool_states = to_state(jnp.asarray(train_images[seed_ix]))
    pool_imgs = jnp.asarray(train_images[seed_ix])
    pool_lbls = jnp.asarray(train_labels[seed_ix])

    params = init_params
    opt_state = optimizer.init(params)
    tkey = jax.random.PRNGKey(0)
    loss_hist = []
    n = int(n_iters.value)

    with mo.status.progress_bar(total=n, title="Training NCA", remove_on_exit=False) as bar:
        for it in range(n):
            tkey, kb, ks = jax.random.split(tkey, 3)
            bix = np.asarray(jax.random.choice(kb, POOL, (BATCH,), replace=False))
            state0, batch_imgs, batch_lbls = pool_states[bix], pool_imgs[bix], pool_lbls[bix]
            # reseed the first few slots with fresh digits so the pool stays varied
            fi = pool_rng.integers(0, len(train_images), 4)
            batch_imgs = batch_imgs.at[:4].set(jnp.asarray(train_images[fi]))
            batch_lbls = batch_lbls.at[:4].set(jnp.asarray(train_labels[fi]))
            state0 = state0.at[:4].set(to_state(batch_imgs[:4]))
            target = jax.nn.one_hot(batch_lbls, N_CLASS)
            params, opt_state, loss, new_state = train_step(
                params, opt_state, state0, target, ks
            )
            pool_states = pool_states.at[bix].set(new_state)
            pool_imgs = pool_imgs.at[bix].set(batch_imgs)
            pool_lbls = pool_lbls.at[bix].set(batch_lbls)
            loss_hist.append(float(loss))
            bar.update(subtitle=f"loss {loss_hist[-1]:.4f}")

    trained_params = params
    mo.md(f"Trained **{n}** iters — L2 loss {loss_hist[0]:.4f} → {min(loss_hist[-10:]):.4f}")
    return (loss_hist,)


@app.cell(hide_code=True)
def _(loss_hist, mo, plt, train_button):
    mo.stop(not train_button.value, mo.md("*Loss curve appears after training.*"))

    figL, axL = plt.subplots(figsize=(6, 2.6))
    axL.plot(loss_hist, color="#4c78a8", lw=1)
    axL.set_xlabel("iteration")
    axL.set_ylabel("L2 loss")
    axL.set_title("Training loss")
    figL.tight_layout()
    figL
    return


if __name__ == "__main__":
    app.run()
