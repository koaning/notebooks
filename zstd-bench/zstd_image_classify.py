# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "numpy==2.4.2",
#     "polars",
#     "Pillow",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Image Classification via Zstd Compression

    Benchmarking the zstd compression-based classifier from
    [Max Halford's blog post](https://maxhalford.github.io/blog/text-classification-zstd/)
    on image datasets. The classifier treats each image as raw bytes and builds
    per-class zstd dictionaries — no feature engineering required.
    """)
    return


@app.cell
def _():
    import io

    import numpy as np
    import polars as pl
    from compression.zstd import ZstdCompressor, ZstdDict
    from PIL import Image
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.datasets import fetch_olivetti_faces, fetch_openml, load_digits
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        BaseEstimator,
        ClassifierMixin,
        Image,
        LogisticRegression,
        PCA,
        StandardScaler,
        ZstdCompressor,
        ZstdDict,
        cross_validate,
        fetch_olivetti_faces,
        fetch_openml,
        io,
        load_digits,
        make_pipeline,
        np,
        pl,
    )


@app.cell
def _(BaseEstimator, ClassifierMixin, Image, ZstdCompressor, ZstdDict, io, np):
    class ZstdBytesClassifier(BaseEstimator, ClassifierMixin):
        """Zstd compression-based classifier that compresses images as PNG bytes.

        From https://maxhalford.github.io/blog/text-classification-zstd/
        """

        def __init__(self, window=1 << 20, level=3, image_shape=None):
            self.window = window
            self.level = level
            self.image_shape = image_shape

        def _to_bytes(self, sample):
            if self.image_shape is not None:
                arr = np.asarray(sample).reshape(self.image_shape)
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                img = Image.fromarray(arr, mode="L")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
            if hasattr(sample, "tobytes"):
                return sample.tobytes()
            return sample.encode("utf-8", errors="replace")

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            buffers = {}
            for sample, label in zip(X, y):
                buf = buffers.get(label, b"") + self._to_bytes(sample)
                if len(buf) > self.window:
                    buf = buf[-self.window :]
                buffers[label] = buf
            self.compressors_ = {
                label: ZstdCompressor(
                    level=self.level,
                    zstd_dict=ZstdDict(buf, is_raw=True),
                )
                for label, buf in buffers.items()
            }
            return self

        def _compress_sizes(self, sample):
            encoded = self._to_bytes(sample)
            mode = ZstdCompressor.FLUSH_FRAME
            return {
                label: len(comp.compress(encoded, mode))
                for label, comp in self.compressors_.items()
            }

        def predict(self, X):
            predictions = []
            for sample in X:
                sizes = self._compress_sizes(sample)
                predictions.append(min(sizes, key=sizes.get))
            return np.array(predictions)

        def predict_proba(self, X):
            probas = []
            for sample in X:
                sizes = self._compress_sizes(sample)
                inv = {label: 1.0 / size for label, size in sizes.items()}
                total = sum(inv.values())
                probas.append([inv[c] / total for c in self.classes_])
            return np.array(probas)

    return (ZstdBytesClassifier,)


@app.cell
def _(fetch_olivetti_faces, fetch_openml, load_digits, np):
    def subsample(X, y, max_samples=None, rng=42):
        if max_samples is None or len(X) <= max_samples:
            return X, y
        if not hasattr(rng, "choice"):
            rng = np.random.default_rng(rng)
        idx = rng.choice(len(X), max_samples, replace=False)
        return X[idx], y[idx]

    def load_all_datasets(max_samples=10_000, seed=42):
        rng = np.random.default_rng(seed)
        result = {}

        digits = load_digits()
        result["Digits (8x8)"] = (digits.data, digits.target, (8, 8))

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X, y = subsample(mnist.data, mnist.target.astype(int), max_samples, rng)
        result["MNIST (28x28)"] = (X, y, (28, 28))

        faces = fetch_olivetti_faces()
        result["Olivetti Faces (64x64)"] = (faces.data, faces.target, (64, 64))

        return result

    return (load_all_datasets,)


@app.cell
def _(load_all_datasets):
    all_datasets = load_all_datasets()
    return (all_datasets,)


@app.cell
def _(all_datasets):
    all_datasets
    return


@app.cell
def _(
    LogisticRegression,
    PCA,
    StandardScaler,
    ZstdBytesClassifier,
    make_pipeline,
):
    def get_pipelines(n_features, image_shape=None):
        pipes = {
            "zstd": ZstdBytesClassifier(image_shape=image_shape),
            "raw+lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        }
        for k in [30, 100]:
            if k < n_features:
                pipes[f"pca({k})+lr"] = make_pipeline(
                    StandardScaler(), PCA(n_components=k), LogisticRegression(max_iter=1000)
                )
        return pipes

    return (get_pipelines,)


@app.cell
def _(all_datasets, cross_validate, get_pipelines, mo, np, pl):
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    rows = []
    for ds_name, (X, y, image_shape) in all_datasets.items():
        n_samples = len(X)
        n_classes = len(np.unique(y))
        pipelines = get_pipelines(X.shape[1], image_shape)
        for pipe_name, pipe in pipelines.items():
            scores = cross_validate(pipe, X, y, cv=5, scoring=scoring)
            rows.append({
                "dataset": ds_name,
                "n_samples": n_samples,
                "n_classes": n_classes,
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
