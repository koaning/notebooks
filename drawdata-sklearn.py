# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "drawdata",
#     "scikit-learn",
#     "numpy",
#     "matplotlib",
#     "polars==1.37.1",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from drawdata import ScatterWidget
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, SplineTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.datasets import make_blobs
    return (
        LabelEncoder,
        LogisticRegression,
        ScatterWidget,
        SplineTransformer,
        make_blobs,
        make_pipeline,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(ScatterWidget, mo):
    scatter_widget = mo.ui.anywidget(ScatterWidget())
    scatter_widget
    return (scatter_widget,)


@app.cell
def _(mo):
    test_size_slider = mo.ui.slider(start=0.1, stop=0.5, step=0.05, value=0.2, label="Test Size")
    test_size_slider
    return (test_size_slider,)


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(LabelEncoder, is_script_mode, make_blobs, mo, np, scatter_widget):
    if is_script_mode:
        X, y = make_blobs(n_samples=200, centers=2, random_state=42)
        X = X.astype(np.float32)
        # Default colors for script mode
        colors = np.array(["#1f77b4" if yi == 0 else "#ff7f0e" for yi in y])
        label_to_color = {0: "#1f77b4", 1: "#ff7f0e"}
    else:
        df = scatter_widget.widget.data_as_polars
        mo.stop(len(df) == 0, mo.md("Draw some data points to get started."))
        X = df.select(["x", "y"]).to_numpy().astype(np.float32)
        colors = df["color"].to_numpy()
        # Encode labels and build label->color mapping
        le = LabelEncoder()
        y = le.fit_transform(colors)
        label_to_color = {i: c for i, c in enumerate(le.classes_)}
    return X, colors, label_to_color, y


@app.cell
def _(LogisticRegression, X, colors, test_size_slider, train_test_split, y):
    X_train, X_test, y_train, y_test, colors_train, colors_test = train_test_split(
        X, y, colors, test_size=test_size_slider.value, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    return (
        X_test,
        X_train,
        colors_test,
        colors_train,
        model,
        test_accuracy,
        train_accuracy,
        y_test,
        y_train,
    )


@app.cell
def _(
    X_test,
    X_train,
    colors_test,
    colors_train,
    label_to_color,
    model,
    np,
    plt,
    test_accuracy,
    train_accuracy,
):
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create colormap from label_to_color mapping
    n_classes = len(label_to_color)
    cmap_colors = [label_to_color[i] for i in range(n_classes)]
    cmap = ListedColormap(cmap_colors)

    # Create mesh grid for decision boundary
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
    y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    # Left panel: Training data
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, n_classes, 1))
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=colors_train, edgecolors="black", s=50)
    axes[0].set_title(f"Training Data (Accuracy: {train_accuracy:.2%})")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    # Right panel: Test data
    axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, n_classes, 1))
    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=colors_test, edgecolors="black", s=50)
    axes[1].set_title(f"Test Data (Accuracy: {test_accuracy:.2%})")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")

    plt.tight_layout()
    return (fig,)


@app.cell
def _(mo):
    n_knots_slider = mo.ui.slider(start=2, stop=10, step=1, value=4, label="Number of Knots")
    knots_dropdown = mo.ui.dropdown(options=["uniform", "quantile"], value="uniform", label="Knots")
    mo.hstack([n_knots_slider, knots_dropdown])
    return knots_dropdown, n_knots_slider


@app.cell
def _(fig_spline):
    fig_spline
    return


@app.cell
def _(
    LogisticRegression,
    SplineTransformer,
    X_test,
    X_train,
    knots_dropdown,
    make_pipeline,
    n_knots_slider,
    y_test,
    y_train,
):
    spline_model = make_pipeline(
        SplineTransformer(n_knots=n_knots_slider.value, degree=3, knots=knots_dropdown.value),
        LogisticRegression(),
    )
    spline_model.fit(X_train, y_train)
    spline_train_accuracy = spline_model.score(X_train, y_train)
    spline_test_accuracy = spline_model.score(X_test, y_test)
    return spline_model, spline_test_accuracy, spline_train_accuracy


@app.cell
def _(
    X_test,
    X_train,
    colors_test,
    colors_train,
    label_to_color,
    np,
    plt,
    spline_model,
    spline_test_accuracy,
    spline_train_accuracy,
):
    from matplotlib.colors import ListedColormap as ListedColormap2

    fig_spline, axes_spline = plt.subplots(1, 2, figsize=(12, 5))

    # Create colormap from label_to_color mapping
    n_classes_spline = len(label_to_color)
    cmap_colors_spline = [label_to_color[i] for i in range(n_classes_spline)]
    cmap_spline = ListedColormap2(cmap_colors_spline)

    # Create mesh grid for decision boundary
    X_all_spline = np.vstack([X_train, X_test])
    x_min_s, x_max_s = X_all_spline[:, 0].min() - 1, X_all_spline[:, 0].max() + 1
    y_min_s, y_max_s = X_all_spline[:, 1].min() - 1, X_all_spline[:, 1].max() + 1
    xx_s, yy_s = np.meshgrid(np.linspace(x_min_s, x_max_s, 200), np.linspace(y_min_s, y_max_s, 200))
    grid_spline = np.c_[xx_s.ravel(), yy_s.ravel()]
    Z_spline = spline_model.predict(grid_spline).reshape(xx_s.shape)

    # Left panel: Training data
    axes_spline[0].contourf(
        xx_s,
        yy_s,
        Z_spline,
        alpha=0.3,
        cmap=cmap_spline,
        levels=np.arange(-0.5, n_classes_spline, 1),
    )
    axes_spline[0].scatter(X_train[:, 0], X_train[:, 1], c=colors_train, edgecolors="black", s=50)
    axes_spline[0].set_title(f"Spline Train (Accuracy: {spline_train_accuracy:.2%})")
    axes_spline[0].set_xlabel("Feature 1")
    axes_spline[0].set_ylabel("Feature 2")

    # Right panel: Test data
    axes_spline[1].contourf(
        xx_s,
        yy_s,
        Z_spline,
        alpha=0.3,
        cmap=cmap_spline,
        levels=np.arange(-0.5, n_classes_spline, 1),
    )
    axes_spline[1].scatter(X_test[:, 0], X_test[:, 1], c=colors_test, edgecolors="black", s=50)
    axes_spline[1].set_title(f"Spline Test (Accuracy: {spline_test_accuracy:.2%})")
    axes_spline[1].set_xlabel("Feature 1")
    axes_spline[1].set_ylabel("Feature 2")

    plt.tight_layout()
    return (fig_spline,)


@app.cell
def _(mo):
    mo.md("""
    ## Spline Basis Functions
    """)
    return


@app.cell
def _(mo):
    viz_n_knots_slider = mo.ui.slider(start=2, stop=10, step=1, value=4, label="Number of Knots")
    viz_knots_dropdown = mo.ui.dropdown(
        options=["uniform", "quantile"], value="uniform", label="Knots"
    )
    mo.hstack([viz_n_knots_slider, viz_knots_dropdown])
    return viz_knots_dropdown, viz_n_knots_slider


@app.cell
def _(SplineTransformer, X, np, plt, viz_knots_dropdown, viz_n_knots_slider):
    fig_basis, axes_basis = plt.subplots(1, 2, figsize=(12, 4))

    # Get x and y ranges from the data
    x_vals = X[:, 0]
    y_vals = X[:, 1]

    # Create spline transformer for visualization
    spline_viz = SplineTransformer(
        n_knots=viz_n_knots_slider.value, degree=3, knots=viz_knots_dropdown.value
    )

    # Plot spline basis for x-axis
    x_range = np.linspace(x_vals.min(), x_vals.max(), 200).reshape(-1, 1)
    spline_viz.fit(x_vals.reshape(-1, 1))
    x_basis = spline_viz.transform(x_range)
    for i in range(x_basis.shape[1]):
        axes_basis[0].plot(x_range, x_basis[:, i], alpha=0.7)
    axes_basis[0].vlines(x_vals, ymin=-0.05, ymax=0.05, color="black", alpha=0.3, zorder=5)
    axes_basis[0].set_title(f"Spline Basis (Feature 1 / x-axis)")
    axes_basis[0].set_xlabel("Feature 1")
    axes_basis[0].set_ylabel("Basis value")

    # Plot spline basis for y-axis
    y_range = np.linspace(y_vals.min(), y_vals.max(), 200).reshape(-1, 1)
    spline_viz.fit(y_vals.reshape(-1, 1))
    y_basis = spline_viz.transform(y_range)
    for i in range(y_basis.shape[1]):
        axes_basis[1].plot(y_range, y_basis[:, i], alpha=0.7)
    axes_basis[1].vlines(y_vals, ymin=-0.05, ymax=0.05, color="black", alpha=0.3, zorder=5)
    axes_basis[1].set_title(f"Spline Basis (Feature 2 / y-axis)")
    axes_basis[1].set_xlabel("Feature 2")
    axes_basis[1].set_ylabel("Basis value")

    plt.tight_layout()
    fig_basis
    return


if __name__ == "__main__":
    app.run()
