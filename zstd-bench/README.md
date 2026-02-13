# Zstd Compression Classifiers

Benchmarking classification via zstd compression dictionaries across text and image domains. Based on [Max Halford's blog post](https://maxhalford.github.io/blog/text-classification-zstd/).

## Idea

Build a zstd dictionary from the training samples of each class. At prediction time, compress the test sample with each dictionary and pick the class whose dictionary yields the smallest compressed size.

## Notebooks

- `zstd_text_classify.py` — Text classification (20 Newsgroups, AG News, IMDb, DBPedia)
- `zstd_image_classify.py` — Image classification (Digits 8x8, MNIST 28x28, Olivetti Faces 64x64)

## Running

```bash
uv run marimo edit zstd_text_classify.py
uv run marimo edit zstd_image_classify.py
```

Requires Python 3.14+ (for `compression.zstd` in stdlib).
