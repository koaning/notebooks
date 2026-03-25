# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    return PCA, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Johnson-Lindenstrauss Random Projections

    The **Johnson-Lindenstrauss lemma** states that a set of $n$ points in high-dimensional
    space can be projected into $k$ dimensions while preserving all
    pairwise **Euclidean distances** up to a factor of $(1 \pm \varepsilon)$, provided

    $$k \geq \frac{4 \ln(n)}{\varepsilon^2 / 2 - \varepsilon^3 / 3}$$

    A simple random Gaussian matrix is enough to achieve this.

    Note that $k$ only depends on the number of points $n$ and the distortion tolerance
    $\varepsilon$, **not** on the original dimensionality $d$. This is only useful when
    $d$ is already large — if your data lives in, say, 3 dimensions, the bound will
    suggest a $k$ that's *higher* than $d$, which defeats the purpose. The lemma shines
    when $d \gg k$, for example when projecting thousands of features down to a few hundred.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    eps_slider = mo.ui.slider(
        start=0.05,
        stop=0.95,
        value=0.5,
        step=0.05,
        label="ε",
    )
    eps_slider
    return (eps_slider,)


@app.cell(hide_code=True)
def _(eps_slider, np, plt):
    eps = eps_slider.value
    n_values = np.arange(2, 10_001)
    k_bound = 4 * np.log(n_values) / (eps**2 / 2 - eps**3 / 3)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(n_values, k_bound)
    _ax.set_xlabel("Number of points (n)")
    _ax.set_ylabel("Minimum dimensions (k)")
    _ax.set_title(f"JL bound: k ≥ 4 ln(n) / (ε²/2 − ε³/3),  ε = {eps:.2f}")
    _ax.set_xscale("log")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation

    By rotational invariance of the Gaussian distribution, we can check distance preservation
    by sampling random vectors and comparing their distance to the first unit vector $e_1$
    before and after projection.
    """)
    return


@app.cell(hide_code=True)
def _(PCA, d_slider, k_slider, n_slider, np):
    D = d_slider.value
    K = k_slider.value
    n_points = n_slider.value

    rng = np.random.default_rng(42)

    # Random points in D dimensions
    points = rng.standard_normal((n_points, D))

    # First unit vector
    e1 = np.zeros(D)
    e1[0] = 1.0

    # Original distances to e1
    dist_orig = np.linalg.norm(points - e1, axis=1)

    # Cosine similarity to e1 in original space (e1 is unit, so just x_0 / ||x||)
    cos_orig = points[:, 0] / np.linalg.norm(points, axis=1)

    # --- Random Gaussian projection ---
    _proj_rp = rng.standard_normal((D, K)) / np.sqrt(K)
    points_rp = points @ _proj_rp
    e1_rp = e1 @ _proj_rp
    dist_rp = np.linalg.norm(points_rp - e1_rp, axis=1)

    # --- PCA projection (rescaled by explained variance) ---
    pca = PCA(n_components=K)
    points_pca = pca.fit_transform(points)
    e1_pca = pca.transform(e1.reshape(1, -1)).ravel()
    _scale = 1.0 / np.sqrt(pca.explained_variance_ratio_.sum())
    dist_pca = np.linalg.norm(points_pca - e1_pca, axis=1) * _scale

    # --- Orthogonalized Gaussian ---
    _rng2 = np.random.default_rng(123)
    _raw = _rng2.standard_normal((D, K))
    _Q, _ = np.linalg.qr(_raw, mode="reduced")
    _proj_orth = _Q * np.sqrt(D / K)
    points_orth = points @ _proj_orth
    e1_orth = e1 @ _proj_orth
    dist_orth = np.linalg.norm(points_orth - e1_orth, axis=1)

    # --- Sparse +/- 1 (Achlioptas) ---
    _vals = _rng2.choice([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0], size=(D, K))
    _proj_sparse = _vals * np.sqrt(3.0 / K)
    points_sparse = points @ _proj_sparse
    e1_sparse = e1 @ _proj_sparse
    dist_sparse = np.linalg.norm(points_sparse - e1_sparse, axis=1)
    return (
        D,
        K,
        cos_orig,
        dist_orig,
        dist_orth,
        dist_pca,
        dist_rp,
        dist_sparse,
        e1_orth,
        e1_pca,
        e1_rp,
        e1_sparse,
        points_orth,
        points_pca,
        points_rp,
        points_sparse,
    )


@app.cell(hide_code=True)
def _(D, K, dist_orig, dist_pca, dist_rp, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 5))

    _lo = min(dist_orig.min(), dist_rp.min(), dist_pca.min())
    _hi = max(dist_orig.max(), dist_rp.max(), dist_pca.max())

    _ax1.scatter(dist_orig, dist_rp, s=1, alpha=0.3)
    _ax1.plot([_lo, _hi], [_lo, _hi], "r--", linewidth=1, label="y = x")
    _ax1.set_xlabel("Original distance")
    _ax1.set_ylabel("Projected distance")
    _ax1.set_title(f"Random Projection (D={D} → k={K})")
    _ax1.legend()
    _ax1.set_aspect("equal")
    _ax1.grid(True, alpha=0.3)

    _ax2.scatter(dist_orig, dist_pca, s=1, alpha=0.3)
    _ax2.plot([_lo, _hi], [_lo, _hi], "r--", linewidth=1, label="y = x")
    _ax2.set_xlabel("Original distance")
    _ax2.set_ylabel("Projected distance (rescaled)")
    _ax2.set_title(f"PCA rescaled (D={D} → k={K})")
    _ax2.legend()
    _ax2.set_aspect("equal")
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Projection Variants

    Can we do better cheaply? Two simple modifications:

    - **Orthogonalized Gaussian** — take the same random Gaussian matrix and orthogonalize
      via QR decomposition. Removes small correlations between projection directions.
    - **Sparse ±1** — replace Gaussian entries with $\{-1, 0, +1\}$ drawn with probabilities
      $\{1/6,\; 2/3,\; 1/6\}$ (Achlioptas). The JL bound still holds, but the projection
      is cheaper to compute since it only requires additions.
    """)
    return


@app.cell(hide_code=True)
def _(D, K, dist_orig, dist_orth, dist_sparse, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 5))

    _lo = min(dist_orig.min(), dist_orth.min(), dist_sparse.min())
    _hi = max(dist_orig.max(), dist_orth.max(), dist_sparse.max())

    _ax1.scatter(dist_orig, dist_orth, s=1, alpha=0.3)
    _ax1.plot([_lo, _hi], [_lo, _hi], "r--", linewidth=1, label="y = x")
    _ax1.set_xlabel("Original distance")
    _ax1.set_ylabel("Projected distance")
    _ax1.set_title(f"Orthogonalized Gaussian (D={D} → k={K})")
    _ax1.legend()
    _ax1.set_aspect("equal")
    _ax1.grid(True, alpha=0.3)

    _ax2.scatter(dist_orig, dist_sparse, s=1, alpha=0.3)
    _ax2.plot([_lo, _hi], [_lo, _hi], "r--", linewidth=1, label="y = x")
    _ax2.set_xlabel("Original distance")
    _ax2.set_ylabel("Projected distance")
    _ax2.set_title(f"Sparse ±1 (D={D} → k={K})")
    _ax2.legend()
    _ax2.set_aspect("equal")
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(dist_orig, dist_orth, dist_pca, dist_rp, dist_sparse, mo, np):
    _methods = {
        "Random Gaussian": dist_rp,
        "PCA (rescaled)": dist_pca,
        "Orthogonalized Gaussian": dist_orth,
        "Sparse ±1": dist_sparse,
    }

    _rows = []
    for _name, _dist in _methods.items():
        _corr = np.corrcoef(dist_orig, _dist)[0, 1]
        _rows.append(
            {
                "Method": _name,
                "Correlation": f"{_corr:.4f}",
                "Mean abs error": f"{np.mean(np.abs(_dist - dist_orig)):.3f}",
                "Max abs error": f"{np.max(np.abs(_dist - dist_orig)):.3f}",
            }
        )

    mo.ui.table(_rows, selection=None, label="Distance preservation summary")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cosine Similarity

    The JL lemma is stated for Euclidean distances, but cosine similarity is determined
    by norms and distances:

    $$\cos(\mathbf{a}, \mathbf{b}) = \frac{\|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - \|\mathbf{a} - \mathbf{b}\|^2}{2\|\mathbf{a}\|\|\mathbf{b}\|}$$

    So if a projection preserves distances and norms well, cosine similarity comes along for free.
    """)
    return


@app.cell(hide_code=True)
def _(
    cos_orig,
    e1_orth,
    e1_pca,
    e1_rp,
    e1_sparse,
    np,
    plt,
    points_orth,
    points_pca,
    points_rp,
    points_sparse,
):
    def _cosine_sim(vecs, ref):
        """Cosine similarity of each row in vecs to ref."""
        _norms = np.linalg.norm(vecs, axis=1)
        _ref_norm = np.linalg.norm(ref)
        return (vecs @ ref) / (_norms * _ref_norm)


    cos_rp = _cosine_sim(points_rp, e1_rp)
    cos_pca = _cosine_sim(points_pca, e1_pca)
    cos_orth = _cosine_sim(points_orth, e1_orth)
    cos_sparse = _cosine_sim(points_sparse, e1_sparse)

    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))

    for _ax, _cos, _title in zip(
        _axes,
        [cos_rp, cos_pca, cos_orth, cos_sparse],
        ["Random Gaussian", "PCA", "Orthogonalized", "Sparse ±1"],
    ):
        _ax.scatter(cos_orig, _cos, s=1, alpha=0.3)
        _ax.plot([-1, 1], [-1, 1], "r--", linewidth=1)
        _ax.set_xlabel("Original cosine sim")
        _ax.set_ylabel("Projected cosine sim")
        _ax.set_title(_title)
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.3)
        _ax.set_xlim(-0.4, 0.4)
        _ax.set_ylim(-0.4, 0.4)

    plt.tight_layout()
    _fig
    return cos_orth, cos_pca, cos_rp, cos_sparse


@app.cell(hide_code=True)
def _(cos_orig, cos_orth, cos_pca, cos_rp, cos_sparse, mo, np):
    _cos_methods = {
        "Random Gaussian": cos_rp,
        "PCA": cos_pca,
        "Orthogonalized Gaussian": cos_orth,
        "Sparse ±1": cos_sparse,
    }

    _rows = []
    for _name, _cos in _cos_methods.items():
        _corr = np.corrcoef(cos_orig, _cos)[0, 1]
        _mae = np.mean(np.abs(_cos - cos_orig))
        _rows.append(
            {
                "Method": _name,
                "Correlation": f"{_corr:.4f}",
                "Mean abs error": f"{_mae:.4f}",
            }
        )

    mo.ui.table(_rows, selection=None, label="Cosine similarity preservation")
    return


@app.cell
def _(mo):
    d_slider = mo.ui.slider(
        start=10,
        stop=2000,
        value=500,
        step=10,
        label="D (original dimensions)",
    )
    k_slider = mo.ui.slider(
        start=2,
        stop=1000,
        value=250,
        step=1,
        label="k (projected dimensions)",
    )
    n_slider = mo.ui.slider(
        start=100,
        stop=10_000,
        value=5_000,
        step=100,
        label="n (number of points)",
    )
    mo.hstack([d_slider, k_slider, n_slider], justify="start")
    return d_slider, k_slider, n_slider


if __name__ == "__main__":
    app.run()
