from pathlib import Path
import anywidget
import traitlets

_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "rpsls_widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "rpsls_widget.css").read_text()


class RpslsWidget(anywidget.AnyWidget):
    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    n = traitlets.Int(default_value=3).tag(sync=True)
