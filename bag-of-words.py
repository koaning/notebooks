# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "diskcache==5.6.3",
#     "ipython==9.13.0",
#     "marimo>=0.23.8",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "openai==2.38.0",
#     "pandas==3.0.3",
#     "scikit-learn==1.8.0",
#     "sentence-transformers==5.5.1",
#     "wigglystuff==0.5.3",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import diskcache as dc
    import os
    from openai import OpenAI
    from wigglystuff import EnvConfig

    # Initialize cache
    cache = dc.Cache("notebooks/.cache")
    return (
        EnvConfig,
        OpenAI,
        SentenceTransformer,
        cache,
        cosine_similarity,
        mo,
        np,
        os,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Exploring Sentence Embeddings and Bag of Words

    Let's experiment to see if embedding models pick up on the difference in meaning when sentences have the same bag of words. We have 75 sentence pairs total, divided into three themes:
    1. **Word Swaps**: Meaning changes, but the sentence is still coherent (e.g., "A man eats a lion" vs "A lion eats a man").
    2. **Shuffles**: Random word order of the same words (grammatically incorrect).
    3. **Negations/Reversals**: Moving negation or adjectives while keeping exact words.
    """)
    return


@app.cell
def _(mo):
    model_options = {
        "Local: all-MiniLM-L6-v2": "local:all-MiniLM-L6-v2",
        "Local: all-mpnet-base-v2": "local:all-mpnet-base-v2",
        "Local: BAAI/bge-small-en-v1.5": "local:BAAI/bge-small-en-v1.5",
        "OpenRouter: openai/text-embedding-3-small": "openrouter:openai/text-embedding-3-small",
        "OpenRouter: openai/text-embedding-3-large": "openrouter:openai/text-embedding-3-large",
        "OpenRouter: google/gemini-embedding-2-preview": "openrouter:google/gemini-embedding-2-preview",
        "OpenRouter: baai/bge-m3": "openrouter:baai/bge-m3",
        "OpenRouter: qwen/qwen3-embedding-4b": "openrouter:qwen/qwen3-embedding-4b",
    }

    dropdown_1 = mo.ui.dropdown(
        options=model_options, value="Local: all-MiniLM-L6-v2", label="Model 1"
    )

    dropdown_2 = mo.ui.dropdown(
        options=model_options, value="Local: all-mpnet-base-v2", label="Model 2"
    )

    ui_layout = mo.hstack([dropdown_1, dropdown_2], justify="start", gap=4)
    return dropdown_1, dropdown_2, ui_layout


@app.cell
def _(EnvConfig):
    env_config = EnvConfig(["OPENROUTER_API_KEY"])
    env_config
    return (env_config,)


@app.cell
def _(OpenAI, SentenceTransformer, cache, env_config, os):
    def get_embeddings(texts, provider_key, batch_size=64):
        """Batch-embed texts; cache hits are skipped, misses raise on API failure."""
        keys = [f"{provider_key}:{t}" for t in texts]
        missing = [(i, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in cache]

        if missing:
            provider_type, model_name = provider_key.split(":", 1)
            idxs, missing_texts = zip(*missing)
            missing_texts = list(missing_texts)

            if provider_type == "local":
                if model_name not in get_embeddings.models:
                    get_embeddings.models[model_name] = SentenceTransformer(model_name)
                new_embs = (
                    get_embeddings.models[model_name]
                    .encode(missing_texts, show_progress_bar=True)
                    .tolist()
                )
            else:
                if provider_type == "openrouter":
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=env_config.get("OPENROUTER_API_KEY")
                        or os.environ.get("OPENROUTER_API_KEY", "dummy_key"),
                    )
                else:
                    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy_key"))

                new_embs = []
                for start in range(0, len(missing_texts), batch_size):
                    chunk = missing_texts[start : start + batch_size]
                    response = client.embeddings.create(
                        input=chunk, model=model_name, encoding_format="float"
                    )
                    if not response.data or len(response.data) != len(chunk):
                        raise RuntimeError(
                            f"{provider_key}: API returned {len(response.data or [])} embeddings for {len(chunk)} inputs"
                        )
                    new_embs.extend(d.embedding for d in response.data)

            for i, emb in zip(idxs, new_embs):
                cache.set(keys[i], emb)

        return [cache.get(k) for k in keys]


    get_embeddings.models = {}
    return (get_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examples
    """)
    return


@app.cell
def _():
    # Define pairs of sentences to compare
    swaps = [
        ("A man eats a lion", "A lion eats a man"),
        (
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog jumps over the quick brown fox",
        ),
        ("She drove the car to the house", "She drove the house to the car"),
        ("The teacher asked the student a question", "The student asked the teacher a question"),
        ("The cat chased the mouse", "The mouse chased the cat"),
        ("John gave Mary a book", "Mary gave John a book"),
        ("The parent scolded the child", "The child scolded the parent"),
        ("The doctor examined the patient", "The patient examined the doctor"),
        ("A dog bites a man", "A man bites a dog"),
        ("The police arrested the thief", "The thief arrested the police"),
        ("The bird ate the worm", "The worm ate the bird"),
        ("The chef cooked the fish", "The fish cooked the chef"),
        ("The car hit the tree", "The tree hit the car"),
        ("The sun warms the earth", "The earth warms the sun"),
        ("The boy hit the ball", "The ball hit the boy"),
        ("The customer paid the waiter", "The waiter paid the customer"),
        ("The wind blew the leaves", "The leaves blew the wind"),
        ("The manager fired the employee", "The employee fired the manager"),
        ("The snake bit the horse", "The horse bit the snake"),
        ("The river carved the canyon", "The canyon carved the river"),
        ("The artist painted the model", "The model painted the artist"),
        ("The software controls the hardware", "The hardware controls the software"),
        ("The king ruled the subjects", "The subjects ruled the king"),
        ("The hunter tracked the bear", "The bear tracked the hunter"),
        ("The predator caught the prey", "The prey caught the predator"),
    ]

    shuffles = [
        ("I love coding in Python", "in love coding Python I"),
        ("The bright sun shines in the blue sky", "sky the sun in The blue bright shines"),
        (
            "Artificial intelligence is transforming the modern world",
            "transforming is the intelligence modern world Artificial",
        ),
        (
            "Learning a new language takes time and practice",
            "new a Learning practice takes and time language",
        ),
        (
            "The quick brown fox jumps over the lazy dog",
            "dog lazy over brown fox quick the The jumps",
        ),
        ("We are going to the park today", "park to today the We going are"),
        ("She likes to read books in the evening", "read likes She evening books to the in"),
        (
            "They played football in the rain yesterday",
            "yesterday played football rain in They the",
        ),
        ("Music can express what words cannot say", "can Music what express words cannot say"),
        ("The old man walked slowly down the street", "slowly walked the man street down The old"),
        ("A delicious smell came from the kitchen", "the from delicious smell came A kitchen"),
        ("He found a rare coin on the beach", "coin He rare on found the beach a"),
        (
            "The tall tree provided shade in the summer",
            "The summer provided the tall in shade tree",
        ),
        (
            "Computers have become an essential part of life",
            "Computers an become have life of part essential",
        ),
        ("The loud noise woke the sleeping baby", "woke noise the sleeping loud The baby"),
        (
            "She painted a beautiful landscape on the canvas",
            "beautiful on the a canvas painted She landscape",
        ),
        ("The train arrived exactly on time", "arrived The on train time exactly"),
        (
            "The mysterious stranger stood in the shadows",
            "The shadows stood stranger in the mysterious",
        ),
        ("An apple a day keeps the doctor away", "keeps away An day the apple a doctor"),
        ("The heavy storm caused widespread damage", "damage caused storm widespread heavy The"),
        ("The little girl wore a bright red dress", "The dress girl bright a red wore little"),
        ("He drank a hot cup of coffee", "coffee drank hot a of He cup"),
        ("The brave soldier fought for his country", "his country for brave fought soldier The"),
        ("A sudden thought crossed her mind", "crossed A her sudden thought mind"),
        ("The ancient ruins hold many secrets", "many hold The ancient ruins secrets"),
    ]

    negations = [
        ("The movie was not good it was bad", "The movie was not bad it was good"),
        ("I do not like coffee I like tea", "I do not like tea I like coffee"),
        ("He is not happy he is sad", "He is not sad he is happy"),
        ("It is not hot it is cold", "It is not cold it is hot"),
        ("The answer is not yes it is no", "The answer is not no it is yes"),
        ("I am not full I am hungry", "I am not hungry I am full"),
        ("The glass is not empty it is full", "The glass is not full it is empty"),
        ("The light is not on it is off", "The light is not off it is on"),
        ("She is not young she is old", "She is not old she is young"),
        ("The box is not light it is heavy", "The box is not heavy it is light"),
        ("I am not wrong I am right", "I am not right I am wrong"),
        ("The water is not shallow it is deep", "The water is not deep it is shallow"),
        ("He is not slow he is fast", "He is not fast he is slow"),
        ("The task is not easy it is hard", "The task is not hard it is easy"),
        ("The room is not clean it is dirty", "The room is not dirty it is clean"),
        ("She is not poor she is rich", "She is not rich she is poor"),
        ("The sky is not clear it is cloudy", "The sky is not cloudy it is clear"),
        ("He is not brave he is scared", "He is not scared he is brave"),
        ("The store is not open it is closed", "The store is not closed it is open"),
        ("The knife is not dull it is sharp", "The knife is not sharp it is dull"),
        ("I am not asleep I am awake", "I am not awake I am asleep"),
        ("The road is not smooth it is rough", "The road is not rough it is smooth"),
        ("She is not tall she is short", "She is not short she is tall"),
        ("She did not win she lost", "She lost she did not win"),
        ("He did not stay he left", "He left he did not stay"),
    ]

    categories = {"Word Swaps": swaps, "Shuffles": shuffles, "Negations": negations}
    return (categories,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding calculation
    """)
    return


@app.cell
def _(categories, cosine_similarity, dropdown_1, dropdown_2, get_embeddings):
    # Collect all unique sentences across pairs, then batch-embed per model
    all_pairs = [(cat, s1, s2) for cat, pairs in categories.items() for s1, s2 in pairs]
    unique_texts = sorted({t for _, s1, s2 in all_pairs for t in (s1, s2)})

    embs_1 = dict(zip(unique_texts, get_embeddings(unique_texts, dropdown_1.value)))
    embs_2 = dict(zip(unique_texts, get_embeddings(unique_texts, dropdown_2.value)))

    results = []
    for cat, s1, s2 in all_pairs:
        sim_1 = cosine_similarity([embs_1[s1]], [embs_1[s2]])[0][0]
        sim_2 = cosine_similarity([embs_2[s1]], [embs_2[s2]])[0][0]
        results.append(
            {
                "Category": cat,
                "Sentence 1": s1,
                "Sentence 2": s2,
                "Model 1 Sim": sim_1,
                "Model 2 Sim": sim_2,
                "Sim Diff": abs(sim_1 - sim_2),
            }
        )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing models
    """)
    return


@app.cell
def _(mo, ui_layout):
    mo.md(f"""
    **Settings: Compare Models**\n\n{ui_layout}
    """)
    return


@app.cell(hide_code=True)
def _(
    categories,
    cosine_similarity,
    dropdown_1,
    dropdown_2,
    get_embeddings,
    mo,
    np,
    plt,
):
    # Similarity matrices for a sample of pairs
    matrix_sentences = []
    matrix_labels = []
    for mat_cat, mat_pairs in categories.items():
        for mat_idx in range(5):
            mat_s1, mat_s2 = mat_pairs[mat_idx]
            matrix_sentences.extend([mat_s1, mat_s2])
            matrix_labels.extend(
                [f"{mat_cat[:4]}: {mat_s1[:15]}...", f"{mat_cat[:4]}: {mat_s2[:15]}..."]
            )

    matrix_embs_1 = get_embeddings(matrix_sentences, dropdown_1.value)
    matrix_embs_2 = get_embeddings(matrix_sentences, dropdown_2.value)

    sim_matrix_1 = cosine_similarity(matrix_embs_1)
    sim_matrix_2 = cosine_similarity(matrix_embs_2)

    fig_mat, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(18, 8))

    cax_m1 = ax_m1.matshow(sim_matrix_1, cmap="viridis", vmin=0, vmax=1)
    fig_mat.colorbar(cax_m1, ax=ax_m1, fraction=0.046, pad=0.04)
    ax_m1.set_xticks(np.arange(len(matrix_labels)))
    ax_m1.set_yticks(np.arange(len(matrix_labels)))
    ax_m1.set_xticklabels(matrix_labels, rotation=90, fontsize=8)
    ax_m1.set_yticklabels(matrix_labels, fontsize=8)
    ax_m1.set_title(f"Model 1: {dropdown_1.value.split(':')[1]}", pad=20)

    cax_m2 = ax_m2.matshow(sim_matrix_2, cmap="viridis", vmin=0, vmax=1)
    fig_mat.colorbar(cax_m2, ax=ax_m2, fraction=0.046, pad=0.04)
    ax_m2.set_xticks(np.arange(len(matrix_labels)))
    ax_m2.set_yticks(np.arange(len(matrix_labels)))
    ax_m2.set_xticklabels(matrix_labels, rotation=90, fontsize=8)
    ax_m2.set_yticklabels(matrix_labels, fontsize=8)
    ax_m2.set_title(f"Model 2: {dropdown_2.value.split(':')[1]}", pad=20)

    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                "### Similarity Matrices\nThese matrices visualize the similarity between a sample of 30 sentences (10 from each category). Bright yellow means high similarity (1.0)."
            ),
            mo.as_html(fig_mat),
        ]
    )
    return


@app.cell
def _(cache, categories, cosine_similarity, mo, np, plt, results):
    # Comparison across all models seen in the cache (not just the two selected).
    # Reference `results` so this re-runs whenever the dropdowns change and add
    # new entries to the cache.
    _ = results

    from collections import defaultdict

    cached_by_model = defaultdict(dict)
    for cache_key in cache.iterkeys():
        parts = cache_key.split(":", 2)
        if len(parts) < 3:
            continue
        cmp_provider, cmp_model, cmp_sentence = parts
        cached_by_model[f"{cmp_provider}:{cmp_model}"][cmp_sentence] = cache.get(cache_key)

    category_order = ["Word Swaps", "Shuffles", "Negations"]
    required_sentences = {t for cmp_pairs in categories.values() for p in cmp_pairs for t in p}

    model_avgs = {}
    for cmp_key, emb_map in cached_by_model.items():
        if not required_sentences.issubset(emb_map.keys()):
            continue
        per_cat = {}
        for cmp_cat, cmp_pair_list in categories.items():
            sims = [cosine_similarity([emb_map[a]], [emb_map[b]])[0][0] for a, b in cmp_pair_list]
            per_cat[cmp_cat] = float(np.mean(sims))
        model_avgs[cmp_key] = per_cat

    fig_cmp, ax_cmp = plt.subplots(figsize=(11, 5))
    cmp_model_keys = sorted(model_avgs.keys())
    n_cmp_models = len(cmp_model_keys)
    cmp_x = np.arange(len(category_order))
    cmp_width = 0.8 / max(n_cmp_models, 1)

    for cmp_i, cmp_mk in enumerate(cmp_model_keys):
        cmp_vals = [model_avgs[cmp_mk][c] for c in category_order]
        cmp_offset = (cmp_i - (n_cmp_models - 1) / 2) * cmp_width
        cmp_label = cmp_mk.split(":", 1)[1]
        cmp_bars = ax_cmp.bar(cmp_x + cmp_offset, cmp_vals, cmp_width, label=cmp_label)
        for cmp_rect in cmp_bars:
            cmp_h = cmp_rect.get_height()
            ax_cmp.annotate(
                f"{cmp_h:.2f}",
                xy=(cmp_rect.get_x() + cmp_rect.get_width() / 2, cmp_h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax_cmp.set_xticks(cmp_x)
    ax_cmp.set_xticklabels(category_order)
    ax_cmp.set_ylabel("Average cosine similarity")
    ax_cmp.set_ylim(0, 1.1)
    ax_cmp.set_title(f"Average similarity per category — all {n_cmp_models} cached model(s)")
    ax_cmp.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"### Cross-model comparison\n"
                f"Averages across all {n_cmp_models} model(s) currently in the cache. Try other models in the dropdowns to populate more bars."
            ),
            mo.as_html(fig_cmp),
        ]
    )
    return


@app.cell
def _(mo, pd, results):
    # Full per-pair results table for the two selected models
    df = pd.DataFrame(results)
    df_formatted = df.copy()
    df_formatted["Model 1 Sim"] = df_formatted["Model 1 Sim"].apply(lambda v: f"{v:.4f}")
    df_formatted["Model 2 Sim"] = df_formatted["Model 2 Sim"].apply(lambda v: f"{v:.4f}")
    df_formatted["Sim Diff"] = df_formatted["Sim Diff"].apply(lambda v: f"{v:.4f}")

    mo.vstack(
        [
            mo.md("### Per-pair results (selected two models)"),
            mo.ui.table(df_formatted, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
