# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy==2.4.2",
#     "polars",
#     "scikit-learn==1.8.0",
#     "datasets",
#     "sentence-transformers",
#     "embetter[sbert]==0.9.0",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Text Classification via Zstd Compression

    Benchmarking the zstd compression-based classifier from
    [Max Halford's blog post](https://maxhalford.github.io/blog/text-classification-zstd/)
    across multiple text classification datasets using cross-validation.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from compression.zstd import ZstdCompressor, ZstdDict
    from datasets import load_dataset
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import make_pipeline
    from embetter.text import SentenceEncoder, TextEncoder
    from embetter.utils import cached

    return (
        BaseEstimator,
        ClassifierMixin,
        CountVectorizer,
        LogisticRegression,
        TextEncoder,
        TfidfVectorizer,
        ZstdCompressor,
        ZstdDict,
        cross_validate,
        fetch_20newsgroups,
        load_dataset,
        make_pipeline,
        np,
        pl,
    )


@app.cell
def _(BaseEstimator, ClassifierMixin, ZstdCompressor, ZstdDict, np):
    class ZstdClassifier(BaseEstimator, ClassifierMixin):
        """Zstd compression-based text classifier.

        From https://maxhalford.github.io/blog/text-classification-zstd/
        """

        def __init__(self, window=1 << 20, level=3):
            self.window = window
            self.level = level

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            buffers = {}
            for text, label in zip(X, y):
                buf = buffers.get(label, b"") + text.encode("utf-8", errors="replace")
                if len(buf) > self.window:
                    buf = buf[-self.window:]
                buffers[label] = buf
            self.compressors_ = {
                label: ZstdCompressor(
                    level=self.level,
                    zstd_dict=ZstdDict(buf, is_raw=True),
                )
                for label, buf in buffers.items()
            }
            return self

        def _compress_sizes(self, text):
            encoded = text.encode("utf-8", errors="replace")
            mode = ZstdCompressor.FLUSH_FRAME
            return {
                label: len(comp.compress(encoded, mode))
                for label, comp in self.compressors_.items()
            }

        def predict(self, X):
            predictions = []
            for text in X:
                sizes = self._compress_sizes(text)
                predictions.append(min(sizes, key=sizes.get))
            return np.array(predictions)

        def predict_proba(self, X):
            probas = []
            for text in X:
                sizes = self._compress_sizes(text)
                inv = {label: 1.0 / size for label, size in sizes.items()}
                total = sum(inv.values())
                probas.append([inv[c] / total for c in self.classes_])
            return np.array(probas)

    return (ZstdClassifier,)


@app.cell
def _(fetch_20newsgroups, load_dataset, np):
    def subsample(X, y, max_samples=None, rng=42):
        X = list(X)
        y = np.asarray(y)
        if max_samples is None or len(X) <= max_samples:
            return X, y
        if not hasattr(rng, "choice"):
            rng = np.random.default_rng(rng)
        idx = rng.choice(len(X), max_samples, replace=False)
        return [X[i] for i in idx], y[idx]

    def load_all_datasets(max_samples=20_000, seed=42):
        rng = np.random.default_rng(seed)
        result = {}

        news = fetch_20newsgroups(subset="all")
        X, y = subsample(news.data, news.target, max_samples, rng)
        result["20 Newsgroups"] = (X, y)

        ag = load_dataset("fancyzhx/ag_news", split="train")
        X, y = subsample(ag["text"], ag["label"], max_samples, rng)
        result["AG News"] = (X, y)

        imdb = load_dataset("imdb", split="train")
        X, y = subsample(imdb["text"], imdb["label"], max_samples, rng)
        result["IMDb"] = (X, y)

        dbpedia = load_dataset("fancyzhx/dbpedia_14", split="train")
        X, y = subsample(dbpedia["content"], dbpedia["label"], max_samples, rng)
        result["DBPedia 14"] = (X, y)

        return result

    return (load_all_datasets,)


@app.cell
def _(load_all_datasets):
    all_datasets = load_all_datasets()
    return (all_datasets,)


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    TextEncoder,
    TfidfVectorizer,
    ZstdClassifier,
    make_pipeline,
):
    pipelines = {
        "zstd": ZstdClassifier(),
        "count+lr": make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000)),
        "tfidf+lr": make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
        "text-enc+lr": make_pipeline(TextEncoder(), LogisticRegression(max_iter=1000)),
    }
    return (pipelines,)


@app.cell
def _(all_datasets, cross_validate, mo, np, pipelines, pl):
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    rows = []
    for ds_name, (X, y) in all_datasets.items():
        X_arr = np.array(X)
        n_samples = len(X_arr)
        for pipe_name, pipe in pipelines.items():
            scores = cross_validate(pipe, X_arr, y, cv=5, scoring=scoring)
            rows.append({
                "dataset": ds_name,
                "n_samples": n_samples,
                "pipeline": pipe_name,
                "accuracy": round(scores["test_accuracy"].mean(), 3),
                "f1": round(scores["test_f1_macro"].mean(), 3),
                "precision": round(scores["test_precision_macro"].mean(), 3),
                "recall": round(scores["test_recall_macro"].mean(), 3),
                "fit_time": round(scores["fit_time"].mean(), 2),
                "score_time": round(scores["score_time"].mean(), 2),
            })
            mo.output.replace(pl.DataFrame(rows))
    results = pl.DataFrame(rows)
    results
    return


if __name__ == "__main__":
    app.run()
