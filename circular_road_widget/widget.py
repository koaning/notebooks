from pathlib import Path

import anywidget
import traitlets

_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "widget.css").read_text()


class CircularRoadWidget(anywidget.AnyWidget):
    """Interactive widget showing a car on a circular road with threshold-based steering."""

    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    # Steering angle - how many degrees the car turns per step when off-road
    angle = traitlets.Float(default_value=2.0).tag(sync=True)

    # Road width in pixels
    road_width = traitlets.Float(default_value=40.0).tag(sync=True)

    # Position along the computed path (0-1)
    position = traitlets.Float(default_value=0.0).tag(sync=True)

    # Total length of the path (computed by JS)
    total_length = traitlets.Float(default_value=0.0).tag(sync=True)
