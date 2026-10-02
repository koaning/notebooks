from pathlib import Path

import anywidget
import traitlets

_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "widget.css").read_text()


class CircularRoadWidget(anywidget.AnyWidget):
    """Interactive widget showing a car on a circular road with threshold-based steering.

    Supports simulating multiple parameter combinations simultaneously (Ladder of Abstraction).
    """

    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    # All angle values to simulate (degrees per step when off-road)
    angles = traitlets.List(traitlets.Float(), default_value=[2.0]).tag(sync=True)

    # All road width values to simulate (pixels)
    road_widths = traitlets.List(traitlets.Float(), default_value=[40.0]).tag(sync=True)

    # Currently selected/highlighted angle
    selected_angle = traitlets.Float(default_value=2.0).tag(sync=True)

    # Currently selected/highlighted road width
    selected_road_width = traitlets.Float(default_value=40.0).tag(sync=True)

    # Position along the selected path (0-1)
    position = traitlets.Float(default_value=0.0).tag(sync=True)

    # Computed data for all (angle, road_width) combinations
    # List of {angle, road_width, total_length}
    path_data = traitlets.List(default_value=[]).tag(sync=True)
