# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo", "anywidget", "traitlets"]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import anywidget
    import traitlets
    import marimo as mo
    return Path, anywidget, mo, traitlets


@app.cell
def _(Path, anywidget, traitlets):
    _ASSET_DIR = Path(__file__).parent
    _JS_SOURCE = (_ASSET_DIR / "fireworks_widget.js").read_text()
    _CSS_SOURCE = (_ASSET_DIR / "fireworks_widget.css").read_text()

    class FireworksWidget(anywidget.AnyWidget):
        _esm = _JS_SOURCE
        _css = _CSS_SOURCE

        trigger = traitlets.Int(default_value=0).tag(sync=True)

        def launch(self):
            """Trigger fireworks animation from Python"""
            self.trigger += 1
    return (FireworksWidget,)


@app.cell
def _(FireworksWidget):
    fireworks = FireworksWidget()
    return (fireworks,)


@app.cell
def _(fireworks, mo):
    fireworks_view = mo.ui.anywidget(fireworks)
    return (fireworks_view,)


@app.cell
def _(fireworks_view):
    fireworks_view
    return


@app.cell
def _(fireworks):
    # You can also trigger fireworks from Python code!
    fireworks.launch() 
    return


if __name__ == "__main__":
    app.run()
