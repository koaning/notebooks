# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "datasets==4.5.0",
#     "marimo",
#     "torch==2.10.0",
#     "transformers==5.2.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Softmax vs Sigmoid Contrastive Loss

    $$
    L_{\text{softmax}} = -\frac{1}{B} \sum_{i=1}^{B} \log \left( \frac{e^{t \cdot x_i \cdot y_i}}{\sum_{j=1}^{B} e^{t \cdot x_i \cdot y_j}} \right)
    $$

    $$
    L_{\text{sigmoid}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{B} \log \left( \frac{1}{1 + e^{-z_{ij}(t \cdot x_i \cdot y_j + b)}} \right)
    $$

    Where $x_i$ and $y_j$ are embeddings, $t$ is temperature, $b$ is a learned bias, and
    $z_{ij}=1$ for positives ($i=j$) and $z_{ij}=-1$ for negatives ($i \neq j$).

    For the sigmoid objective, each $(i, j)$ term is independent: there is no shared
    denominator coupling scores across pairs. That allows computing it piece by piece,
    e.g. in row chunks, and summing the contributions.

    Note on labels: sigmoid needs a target label for each pair, $z_{ij}\in\{+1,-1\}$.
    In this notebook, we do not pass labels explicitly; we infer them from batch order:
    diagonal pairs $(i=i)$ are treated as positives and off-diagonal pairs $(i\neq j)$
    are treated as negatives.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.functional as F

    return F, mo, torch


@app.cell
def _(mo):
    t = mo.ui.slider(start=0.05, stop=5.0, step=0.05, value=2.0, label="temperature (t)")
    t
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss implementations
    """)
    return


@app.cell
def _(torch):
    def softmax_loss(x: torch.Tensor, y: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """Compute the batch softmax contrastive loss."""
        if x.shape[0] != y.shape[0]:
            raise ValueError("softmax_loss expects x and y to have the same batch size")

        logits = t * (x @ y.T)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -torch.diagonal(log_probs).mean()

    return (softmax_loss,)


@app.cell
def _(F, torch):
    def sigmoid_loss(x: torch.Tensor, y: torch.Tensor, t: float = 1.0, b: float = 0.0) -> torch.Tensor:
        """Compute batch sigmoid contrastive loss."""
        if x.shape[0] != y.shape[0]:
            raise ValueError("sigmoid_loss expects x and y to have the same batch size")

        logits = t * (x @ y.T) + b
        z = 2 * torch.eye(x.shape[0], device=x.device) - 1
        return F.softplus(-z * logits).sum() / x.shape[0]


    def sigmoid_loss_chunked(
        x: torch.Tensor,
        y: torch.Tensor,
        t: float = 1.0,
        b: float = 0.0,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Compute batch sigmoid contrastive loss in row chunks.

        Mathematically identical to sigmoid_loss, but avoids materializing the
        full B×B similarity matrix — useful when B is large.
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError("sigmoid_loss_chunked expects x and y to have the same batch size")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        batch_size = x.shape[0]
        total = torch.zeros((), dtype=x.dtype, device=x.device)

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            logits_block = t * (x[start:end] @ y.T) + b

            z_block = -torch.ones_like(logits_block)
            local_rows = torch.arange(end - start, device=x.device)
            global_cols = torch.arange(start, end, device=x.device)
            z_block[local_rows, global_cols] = 1.0

            total = total + F.softplus(-(z_block * logits_block)).sum()

        return total / batch_size

    return sigmoid_loss, sigmoid_loss_chunked


@app.cell
def _(sigmoid_loss, sigmoid_loss_chunked, softmax_loss, t, torch):
    torch.manual_seed(0)
    _x = torch.randn(8, 4)
    _y = torch.randn(8, 4)
    {
        "temperature_t": t.value,
        "softmax_loss": float(softmax_loss(_x, _y, t=t.value)),
        "sigmoid_loss": float(sigmoid_loss(_x, _y, t=t.value, b=-0.3)),
        "chunked_loss": float(sigmoid_loss_chunked(_x, _y, t=t.value, b=-0.3, chunk_size=3)),
    }
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset Task: Headline -> Lead

    We prepare a contrastive dataset from `heegyu/news-category-dataset`:
    - anchor: `headline`
    - positive: `short_description` (lead sentence style summary)

    We keep stratified `train`/`val`/`test` splits in memory as Hugging Face `Dataset` objects.
    """)
    return


@app.cell
def _(mo):
    max_samples = mo.ui.slider(
        start=200, stop=5000, step=100, value=1000, label="max prepared samples"
    )
    max_samples
    return (max_samples,)


@app.cell
def _():
    import random
    from collections import Counter, defaultdict

    from datasets import Dataset, load_dataset

    return Counter, Dataset, defaultdict, load_dataset, random


@app.function
def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


@app.cell
def _(Counter, Dataset, defaultdict, load_dataset, max_samples, random):
    raw = load_dataset("heegyu/news-category-dataset", split="train")

    rows = []
    for idx, item in enumerate(raw):
        headline = clean_text(item["headline"])
        lead = clean_text(item["short_description"])
        if len(headline.split()) < 3 or len(lead.split()) < 6:
            continue

        rows.append({
            "id": f"news-{idx}",
            "topic": clean_text(item["category"]),
            "date": str(item["date"]),
            "headline": headline,
            "lead": lead,
        })

    rng = random.Random(42)
    rng.shuffle(rows)
    rows = rows[: min(max_samples.value, len(rows))]

    by_topic = defaultdict(list)
    for row in rows:
        by_topic[row["topic"]].append(row)

    train_rows, val_rows, test_rows = [], [], []

    for topic_rows in by_topic.values():
        rng.shuffle(topic_rows)
        n = len(topic_rows)
        n_val = max(1, int(0.05 * n))
        n_test = max(1, int(0.05 * n))

        val_rows.extend(topic_rows[:n_val])
        test_rows.extend(topic_rows[n_val : n_val + n_test])
        train_rows.extend(topic_rows[n_val + n_test :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"
    for row in test_rows:
        row["split"] = "test"

    summary = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "topic_count": len(by_topic),
        "train_topic_counts": dict(Counter(row["topic"] for row in train_rows)),
        "val_topic_counts": dict(Counter(row["topic"] for row in val_rows)),
        "test_topic_counts": dict(Counter(row["topic"] for row in test_rows)),
    }

    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)
    test_dataset = Dataset.from_list(test_rows)
    return test_dataset, train_dataset, val_dataset


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple PyTorch Bi-Encoder

    Lightweight encoder backbone: `sentence-transformers/all-MiniLM-L6-v2`.
    We tokenize headlines and leads separately, encode both with shared weights,
    and optimize either softmax loss or sigmoid loss.
    """)
    return


@app.cell
def _(torch):
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    class BiEncoder(nn.Module):
        def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            self.bias = nn.Parameter(torch.zeros(()))

        def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in kwargs:
                model_inputs["token_type_ids"] = kwargs["token_type_ids"]

            outputs = self.backbone(**model_inputs)
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            return nn.functional.normalize(pooled, p=2, dim=1)

    return AutoTokenizer, BiEncoder, DataLoader


@app.cell
def _(
    AutoTokenizer,
    DataLoader,
    batch_size,
    max_length,
    test_dataset,
    train_dataset,
    val_dataset,
):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def collate_fn(batch):
        headlines = [row["headline"] for row in batch]
        leads = [row["lead"] for row in batch]
        x_tokens = tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=max_length.value,
            return_tensors="pt",
        )
        y_tokens = tokenizer(
            leads,
            padding=True,
            truncation=True,
            max_length=max_length.value,
            return_tensors="pt",
        )
        return x_tokens, y_tokens

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size.value, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size.value, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size.value, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


@app.cell(column=2)
def _(mo):
    loss_name = mo.ui.dropdown(["softmax", "sigmoid"], value="softmax", label="loss")
    epochs = mo.ui.slider(start=1, stop=5, step=1, value=1, label="epochs")
    batch_size = mo.ui.slider(start=8, stop=64, step=8, value=32, label="batch size")
    learning_rate = mo.ui.slider(
        start=1e-5, stop=5e-4, step=1e-5, value=1e-4, label="learning rate"
    )
    max_length = mo.ui.slider(start=32, stop=192, step=16, value=96, label="max tokens")
    train_button = mo.ui.run_button(label="Train bi-encoder")
    mo.vstack([loss_name, epochs, batch_size, learning_rate, max_length, train_button])
    return (
        batch_size,
        epochs,
        learning_rate,
        loss_name,
        max_length,
        train_button,
    )


@app.cell
def _(
    BiEncoder,
    epochs,
    learning_rate,
    loss_name,
    sigmoid_loss,
    softmax_loss,
    t,
    torch,
    train_button,
    train_loader,
    val_loader,
):
    training_log = []

    if train_button.value:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiEncoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate.value)

        for epoch in range(epochs.value):
            model.train()
            train_total = 0.0
            train_batches = 0
            for x_tokens, y_tokens in train_loader:
                x_tokens = {k: v.to(device) for k, v in x_tokens.items()}
                y_tokens = {k: v.to(device) for k, v in y_tokens.items()}
                x_emb = model.encode(**x_tokens)
                y_emb = model.encode(**y_tokens)

                if loss_name.value == "softmax":
                    loss = softmax_loss(x_emb, y_emb, t=t.value)
                else:
                    loss = sigmoid_loss(x_emb, y_emb, t=t.value, b=model.bias)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_total += float(loss.item())
                train_batches += 1

            model.eval()
            val_total = 0.0
            val_batches = 0
            val_x_all = []
            val_y_all = []
            with torch.no_grad():
                for x_tokens, y_tokens in val_loader:
                    x_tokens = {k: v.to(device) for k, v in x_tokens.items()}
                    y_tokens = {k: v.to(device) for k, v in y_tokens.items()}
                    x_emb = model.encode(**x_tokens)
                    y_emb = model.encode(**y_tokens)
                    val_x_all.append(x_emb)
                    val_y_all.append(y_emb)
                    if loss_name.value == "softmax":
                        val_loss = softmax_loss(x_emb, y_emb, t=t.value)
                    else:
                        val_loss = sigmoid_loss(x_emb, y_emb, t=t.value, b=model.bias)
                    val_total += float(val_loss.item())
                    val_batches += 1

            val_x = torch.cat(val_x_all, dim=0)
            val_y = torch.cat(val_y_all, dim=0)
            sim = val_x @ val_y.T
            top1 = sim.argmax(dim=1)
            targets = torch.arange(sim.shape[0], device=sim.device)
            val_recall_at_1 = (top1 == targets).float().mean().item()

            training_log.append({
                "epoch": epoch + 1,
                "train_loss": train_total / max(train_batches, 1),
                "val_loss_in_batch": val_total / max(val_batches, 1),
                "val_recall_at_1": val_recall_at_1,
            })
    return (training_log,)


@app.cell
def _(training_log):
    training_log
    return


@app.cell
def _():
    return


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
