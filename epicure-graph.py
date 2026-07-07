# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=0.28.0",
#     "marimo>=0.23.8",
#     "matplotlib==3.10.9",
#     "networkx>=3.4",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "safetensors>=0.5.0",
#     "scikit-learn==1.8.0",
#     "wigglystuff==0.5.9",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from wigglystuff import GraphWidget

    return (
        GraphWidget,
        hf_hub_download,
        json,
        load_file,
        math,
        mo,
        np,
        nx,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Epicure: Ingredient Graphs from Embeddings

    [Epicure](https://arxiv.org/abs/2605.22391) trains three sibling ingredient embeddings
    (Cooc, Core, Chem) on a graph built from **4M recipes** and **FlavorDB chemistry**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Epicure embeddings

    Three models, same 1,790-ingredient vocabulary, different **random walks**:

    | Model | Walks | Intuition |
    |-------|-------|-----------|
    | **Cooc** | ingredient–ingredient only | "What do I cook with this?" |
    | **Chem** | compound metapaths only | "What shares its flavor chemistry?" |
    | **Core** | both, with recipe walks **injected 10×** | mostly recipe context + some chemistry |

    ### What does "10× I–I injection" mean?

    During Core training, each round samples walk templates. Core adds **10 extra pure
    ingredient–ingredient walks** per round on top of the chemistry metapaths.

    So if one chemistry walk fires once, recipe-context walks fire ten times — recipe
    co-occurrence dominates the signal Core sees, but chemistry still nudges the geometry.
    That's why Core sits between Cooc and Chem, and why its embedding space is more
    **concentrated** (tighter clusters) than the other two.
    """)
    return


@app.cell
def _(hf_hub_download, json, load_file, np):
    def _unit(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)


    class EpicureModel:
        """Minimal loader for the published Epicure embedding repos."""

        def __init__(
            self,
            embeddings: np.ndarray,
            vocab: dict[str, int],
            supervised_poles: dict[str, np.ndarray],
        ):
            self.E_raw = embeddings.astype(np.float32)
            self.E = _unit(self.E_raw)
            self.vocab = vocab
            self.itos = {i: name for name, i in vocab.items()}
            self.supervised_poles = supervised_poles

        @classmethod
        def from_pretrained(cls, repo_id: str) -> "EpicureModel":
            emb_path = hf_hub_download(repo_id, "embeddings.safetensors")
            vocab_path = hf_hub_download(repo_id, "vocab.json")
            poles_path = hf_hub_download(repo_id, "supervised_poles.json")

            embeddings = load_file(emb_path)["embeddings"]
            with open(vocab_path) as f:
                vocab = json.load(f)
            with open(poles_path) as f:
                supervised_poles = {
                    key: np.array(value, dtype=np.float32) for key, value in json.load(f).items()
                }
            return cls(embeddings, vocab, supervised_poles)

        def neighbors(self, name: str, k: int = 5) -> list[tuple[str, float]]:
            vec = self.E[self.vocab[name]]
            sims = self.E @ vec
            order = np.argsort(-sims)[1 : k + 1]
            return [(self.itos[int(i)], float(sims[i])) for i in order]

        def list_supervised_poles(self, prefix: str) -> list[str]:
            return [key for key in self.supervised_poles if key.startswith(prefix)]

        def slerp(
            self,
            seed: str,
            direction: str,
            theta_deg: float,
            k: int = 5,
        ) -> list[tuple[str, float]]:
            seed_idx = self.vocab[seed]
            v = self.E[seed_idx]
            d = _unit(self.supervised_poles[direction].reshape(1, -1))[0]
            d_perp = d - (d @ v) * v
            n_perp = np.linalg.norm(d_perp)
            if n_perp < 1e-9:
                return self.neighbors(seed, k=k)
            d_perp = d_perp / n_perp
            theta = np.deg2rad(theta_deg)
            q = _unit((np.cos(theta) * v + np.sin(theta) * d_perp).reshape(1, -1))[0]
            sims = self.E @ q
            sims[seed_idx] = -np.inf
            order = np.argsort(-sims)[:k]
            return [(self.itos[int(i)], float(sims[i])) for i in order]


    models = {
        "Cooc": EpicureModel.from_pretrained("Kaikaku/epicure-cooc"),
        "Core": EpicureModel.from_pretrained("Kaikaku/epicure-core"),
        "Chem": EpicureModel.from_pretrained("Kaikaku/epicure-chem"),
    }
    ingredient_names = sorted(models["Cooc"].vocab.keys())
    len(ingredient_names)
    return ingredient_names, models


@app.cell(hide_code=True)
def _(ingredient_names, mo):
    seed_dropdown = mo.ui.dropdown(
        options=ingredient_names,
        value="chicken" if "chicken" in ingredient_names else ingredient_names[0],
        label="Seed ingredient",
    )
    model_dropdown = mo.ui.dropdown(
        options=["Cooc", "Core", "Chem"],
        value="Cooc",
        label="Model",
    )
    k_slider = mo.ui.slider(3, 12, value=6, step=1, label="Neighbors (k) — table")
    graph_k_slider = mo.ui.slider(2, 12, value=6, step=1, label="Max neighbors (k) — graph")
    cosine_threshold_slider = mo.ui.slider(
        0.20, 0.95, value=0.45, step=0.05, label="Min cosine — graph edges"
    )
    hop_slider = mo.ui.slider(1, 4, value=2, step=1, label="Hops from seed — display")
    return (
        cosine_threshold_slider,
        graph_k_slider,
        hop_slider,
        k_slider,
        model_dropdown,
        seed_dropdown,
    )


@app.cell(hide_code=True)
def _(k_slider, model_dropdown, models, pd, seed_dropdown):
    seed = seed_dropdown.value
    model_name = model_dropdown.value
    k = k_slider.value

    top_neighbors = models[model_name].neighbors(seed, k=k)
    neighbors_df = pd.DataFrame(top_neighbors, columns=["ingredient", "cosine"])
    neighbors_df.index = range(1, len(neighbors_df) + 1)
    neighbors_df.index.name = "rank"
    neighbors_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What about flavor compounds?

    **Correct — this notebook will not show compound nodes.**

    Epicure's training graph has two node types:
    - **1,790 ingredients** (what you query here)
    - **2,247 typed compound nodes** from FlavorDB (walk intermediates only)

    Skip-gram still runs on compound IDs during Core/Chem walks, but the
    [published HuggingFace release](https://huggingface.co/datasets/Kaikaku/epicure-corpus-resources)
    ships **ingredient embeddings only** — not compound vectors, and not the
    80k ingredient–compound edge list (held back for license review).

    So **Chem neighbors are still ingredients** — e.g. for `chicken` you get
    `pork`, `beef` (shared flavor chemistry), not `limonene` or `myristicin`.
    The chemistry is *implicit* in those ingredient vectors, not explicit in the output.

    Below: a **toy** $I \rightarrow C \rightarrow I$ metapath (not Epicure's real graph).
    """)
    return


@app.cell
def _(nx, plt):
    # Illustrative chemistry layer — not from published Epicure edge data.
    chem_edges = [
        ("chicken", "myristicin_meaty", "meaty"),
        ("pork", "myristicin_meaty", "meaty"),
        ("beef", "oleic_acid_fatty", "fatty"),
        ("chicken", "oleic_acid_fatty", "fatty"),
        ("lemon", "citral_citrus", "citrus"),
        ("orange", "citral_citrus", "citrus"),
        ("garlic", "chicken", "via_recipe"),
    ]

    G_chem = nx.Graph()
    for ingredient, compound, flavor_type in chem_edges:
        G_chem.add_node(ingredient, kind="ingredient")
        G_chem.add_node(compound, kind="compound", flavor=flavor_type)
        G_chem.add_edge(ingredient, compound, flavor=flavor_type)

    chem_pos = {
        "chicken": (0, 1),
        "pork": (2, 1),
        "beef": (3, 0),
        "garlic": (-1.5, 0),
        "lemon": (-1, -1.5),
        "orange": (1, -1.5),
        "myristicin_meaty": (1, 2),
        "oleic_acid_fatty": (2.5, 0.5),
        "citral_citrus": (0, -2.5),
    }

    ingredient_nodes = [n for n, d in G_chem.nodes(data=True) if d["kind"] == "ingredient"]
    compound_nodes = [n for n, d in G_chem.nodes(data=True) if d["kind"] == "compound"]

    fig_chem, ax_chem = plt.subplots(figsize=(9, 5))
    nx.draw_networkx_edges(G_chem, chem_pos, alpha=0.5, ax=ax_chem)
    nx.draw_networkx_nodes(
        G_chem,
        chem_pos,
        nodelist=ingredient_nodes,
        node_color="#3498db",
        node_size=900,
        ax=ax_chem,
    )
    nx.draw_networkx_nodes(
        G_chem,
        chem_pos,
        nodelist=compound_nodes,
        node_color="#e67e22",
        node_shape="s",
        node_size=700,
        ax=ax_chem,
    )
    nx.draw_networkx_labels(G_chem, chem_pos, font_size=8, ax=ax_chem)
    ax_chem.set_title("Toy I–C–I chemistry layer (orange squares = compounds)")
    ax_chem.axis("off")
    fig_chem.tight_layout()
    fig_chem
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Approximate cosine graph (interactive)

    Epicure's real training edges use **NPMI > 0** (not published). Here we proxy with:

    1. **Edge rule:** for each ingredient, take its top-$k$ cosine neighbors, then keep only edges with $\cos \geq$ threshold.
    2. **Display rule:** show nodes within **N hops** of the seed along those edges (capped so the browser stays responsive).

    `wigglystuff.GraphWidget` below — drag nodes to explore.
    """)
    return


@app.cell
def _(
    GraphWidget,
    cosine_threshold_slider,
    graph_k_slider,
    hop_slider,
    mo,
    model_dropdown,
    models,
    np,
    nx,
    seed_dropdown,
):
    _MAX_WIDGET_NODES = 50
    _MAX_WIDGET_EDGES = 100

    active_model = models[model_dropdown.value]
    names = [active_model.itos[i] for i in range(len(active_model.itos))]
    embeddings = active_model.E

    threshold = cosine_threshold_slider.value
    k_graph = graph_k_slider.value
    hop_breadth = hop_slider.value

    G = nx.Graph()
    for row_idx in range(len(names)):
        sims = embeddings[row_idx] @ embeddings.T
        sims[row_idx] = -np.inf
        neighbor_count = min(k_graph, len(names) - 1)
        top_idx = np.argpartition(-sims, neighbor_count - 1)[:neighbor_count]
        for col_idx in top_idx:
            cos_ij = float(sims[col_idx])
            if cos_ij >= threshold:
                G.add_edge(names[row_idx], names[col_idx], weight=cos_ij)

    focus_seed = seed_dropdown.value
    if focus_seed in G:
        focus_nodes = set(
            nx.single_source_shortest_path_length(G, focus_seed, cutoff=hop_breadth)
        )
    else:
        focus_nodes = {focus_seed}

    subgraph = G.subgraph(focus_nodes).copy()
    graph_truncated = False

    if subgraph.number_of_nodes() > _MAX_WIDGET_NODES:
        graph_truncated = True
        seed_idx = active_model.vocab[focus_seed]
        seed_sims = embeddings[seed_idx] @ embeddings.T
        ranked_idx = np.argsort(-seed_sims)
        trimmed = {focus_seed}
        for rank_idx in ranked_idx:
            if names[rank_idx] in focus_nodes:
                trimmed.add(names[rank_idx])
            if len(trimmed) >= _MAX_WIDGET_NODES:
                break
        subgraph = G.subgraph(trimmed).copy()

    display_edges = list(subgraph.edges(data=True))
    if len(display_edges) > _MAX_WIDGET_EDGES:
        graph_truncated = True
        display_edges = sorted(display_edges, key=lambda e: -e[2].get("weight", 0))[
            :_MAX_WIDGET_EDGES
        ]
        kept_nodes = {focus_seed}
        for src, dst, _ in display_edges:
            kept_nodes.add(src)
            kept_nodes.add(dst)
        subgraph = G.subgraph(kept_nodes).copy()

    full_graph_stats = (G.number_of_nodes(), G.number_of_edges())

    graph_nodes = [
        {
            "id": node,
            "name": node,
            "color": "#e74c3c" if node == focus_seed else "#3498db",
            "size": 16 if node == focus_seed else 10,
        }
        for node in subgraph.nodes()
    ]
    graph_edges = [
        {
            "source": src,
            "target": dst,
            "width": 1.5 + 3.0 * edge_data.get("weight", 0.5),
        }
        for src, dst, edge_data in subgraph.edges(data=True)
    ]

    graph_widget = mo.ui.anywidget(
        GraphWidget(
            nodes=graph_nodes,
            edges=graph_edges,
            directed=False,
            height=520,
        )
    )

    graph_widget
    return full_graph_stats, graph_truncated, subgraph


@app.cell(hide_code=True)
def _(
    cosine_threshold_slider,
    full_graph_stats,
    graph_k_slider,
    graph_truncated,
    hop_slider,
    mo,
    model_dropdown,
    seed_dropdown,
    subgraph,
):
    hop_word = "hop" if hop_slider.value == 1 else "hops"
    truncated_note = " (trimmed for display)" if graph_truncated else ""
    mo.vstack(
        [
            mo.hstack([seed_dropdown, model_dropdown], gap=2),
            mo.hstack(
                [graph_k_slider, cosine_threshold_slider, hop_slider], gap=2
            ),
            mo.md(
                f"Full graph (top-{graph_k_slider.value} per node, cosine $\\geq$ "
                f"{cosine_threshold_slider.value}): **{full_graph_stats[1]}** edges across "
                f"**{full_graph_stats[0]}** nodes.\n\n"
                f"Displayed subgraph ({hop_slider.value} {hop_word} from "
                f"**{seed_dropdown.value}**){truncated_note}: "
                f"**{subgraph.number_of_nodes()}** nodes, "
                f"**{subgraph.number_of_edges()}** edges."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## SLERP: rotate an ingredient toward a cuisine

    Epicure's direction arithmetic rotates a seed on the unit sphere toward a supervised pole.
    """)
    return


@app.cell
def _(mo, models):
    core_model = models["Core"]
    cuisine_poles = core_model.list_supervised_poles("cuisine:")
    cuisine_options = {
        key.split("/", 1)[-1].replace("_", " "): key for key in cuisine_poles
    }
    default_cuisine = next(
        (label for label, key in cuisine_options.items() if "South Asian" in label),
        next(iter(cuisine_options)),
    )

    slerp_seed = mo.ui.dropdown(
        options=sorted(core_model.vocab.keys()),
        value="rice" if "rice" in core_model.vocab else sorted(core_model.vocab.keys())[0],
        label="SLERP seed",
    )
    cuisine_dropdown = mo.ui.dropdown(
        options=cuisine_options,
        value=default_cuisine,
        label="Target cuisine",
    )
    theta_slider = mo.ui.slider(0, 60, value=30, step=5, label="Rotation angle θ (degrees)")
    mo.hstack([slerp_seed, cuisine_dropdown, theta_slider], gap=2)
    return cuisine_dropdown, slerp_seed, theta_slider


@app.cell
def _(cuisine_dropdown, models, pd, slerp_seed, theta_slider):
    direction = cuisine_dropdown.value
    slerp_results = models["Core"].slerp(
        slerp_seed.value,
        direction,
        theta_deg=theta_slider.value,
        k=8,
    )
    slerp_df = pd.DataFrame(slerp_results, columns=["ingredient", "cosine"])
    slerp_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    **Rebuild the real training graph?** You'd need the 11 source recipe corpora,
    canonicalize to the [published vocabulary](https://huggingface.co/datasets/Kaikaku/epicure-corpus-resources),
    compute NPMI edges, and add FlavorDB compound links (523 hub ingredients, 80k I–C edges).
    Neither edge list is on HuggingFace yet — the kNN graph above is a quick proxy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix: PMI and NPMI

    When building the recipe graph, Epicure asks: do ingredients **A** and **B** show up together more than you'd expect by chance?

    ### PMI — Pointwise Mutual Information

    **PMI** measures how much more (or less) two ingredients co-occur compared to independence:

    $$
    \mathrm{PMI}(A,B) = \log \frac{P(A,B)}{P(A)\,P(B)}
    $$

    - $\mathrm{PMI} > 0$: they co-occur **more** than chance (a candidate edge)
    - $\mathrm{PMI} = 0$: independent
    - $\mathrm{PMI} < 0$: they co-occur **less** than chance

    ### NPMI — Normalized Pointwise Mutual Information

    **NPMI** rescales PMI to a bounded range, roughly $[-1, 1]$:

    $$
    \mathrm{NPMI}(A,B) = \frac{\mathrm{PMI}(A,B)}{-\log P(A,B)}
    $$

    - $\mathrm{NPMI} \approx 1$: almost always appear together
    - $\mathrm{NPMI} \approx 0$: no meaningful association
    - $\mathrm{NPMI} < 0$: negative association

    Epicure keeps only pairs with **positive NPMI** as ingredient–ingredient edges.
    """)
    return


@app.cell
def _(math, np, pd):
    toy_recipes = [
        ["tomato", "basil", "garlic"],
        ["tomato", "basil", "mozzarella"],
        ["chicken", "garlic", "onion"],
        ["chicken", "onion", "rice"],
        ["tomato", "onion", "olive_oil"],
    ]
    ingredients = sorted({item for recipe in toy_recipes for item in recipe})
    n_recipes = len(toy_recipes)

    pair_counts = {}
    ingredient_counts = {name: 0 for name in ingredients}
    for recipe in toy_recipes:
        for item in recipe:
            ingredient_counts[item] += 1
        for idx, a in enumerate(recipe):
            for b in recipe[idx + 1 :]:
                key = tuple(sorted((a, b)))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    rows = []
    for a in ingredients:
        for b in ingredients:
            if a >= b:
                continue
            count_ab = pair_counts.get(tuple(sorted((a, b))), 0)
            p_ab = count_ab / n_recipes
            p_a = ingredient_counts[a] / n_recipes
            p_b = ingredient_counts[b] / n_recipes
            if count_ab == 0 or p_ab == 0:
                npmi = float("nan")
                pmi = float("nan")
            else:
                pmi = math.log(p_ab / (p_a * p_b))
                npmi = pmi / -math.log(p_ab)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "co_recipes": count_ab,
                    "pmi": pmi,
                    "npmi": npmi,
                    "edge": npmi > 0 if not np.isnan(npmi) else False,
                }
            )

    npmi_df = pd.DataFrame(rows).sort_values("npmi", ascending=False, na_position="last")
    npmi_df
    return


if __name__ == "__main__":
    app.run()
