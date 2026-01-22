from pathlib import Path
import anywidget
import traitlets

_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "cube_widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "cube_widget.css").read_text()


class CubeWidget(anywidget.AnyWidget):
    """Interactive 3D cube widget for progressive dimension locking.

    Allows users to:
    - Start with a 3D volume (cube)
    - Lock one axis to view a 2D plane (blue, badge 1)
    - Lock a second axis to view a 1D line (orange, badge 2)
    - Lock all three axes to select a single point (green, badge 3)

    The widget outputs its current selection state via plane, line, point properties.
    """

    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    # Axis configurations - each has "name" and "values" keys
    x_axis = traitlets.Dict(
        default_value={"name": "X", "values": [0, 0.25, 0.5, 0.75, 1]}
    ).tag(sync=True)
    y_axis = traitlets.Dict(
        default_value={"name": "Y", "values": [0, 0.25, 0.5, 0.75, 1]}
    ).tag(sync=True)
    z_axis = traitlets.Dict(
        default_value={"name": "Z", "values": [0, 0.25, 0.5, 0.75, 1]}
    ).tag(sync=True)

    # Internal state synced with JS
    locked_order = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    axis_values = traitlets.Dict(
        default_value={"x": 0.5, "y": 0.5, "z": 0.5}
    ).tag(sync=True)

    # Output properties - these are computed from locked_order and axis_values
    plane = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    line = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
    point = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, x_axis=None, y_axis=None, z_axis=None, **kwargs):
        super().__init__(**kwargs)
        if x_axis is not None:
            self.x_axis = x_axis
        if y_axis is not None:
            self.y_axis = y_axis
        if z_axis is not None:
            self.z_axis = z_axis

        # Set up observers to compute plane/line/point from locked_order
        self.observe(self._update_outputs, names=["locked_order", "axis_values"])

    def _get_axis_config(self, axis_key):
        """Get the axis config for a given key (x, y, z)."""
        configs = {"x": self.x_axis, "y": self.y_axis, "z": self.z_axis}
        return configs.get(axis_key)

    def _update_outputs(self, change=None):
        """Update plane, line, point based on locked_order and axis_values."""
        locked = self.locked_order
        values = self.axis_values

        # Reset all outputs
        new_plane = None
        new_line = None
        new_point = None

        if len(locked) >= 1:
            axis_key = locked[0]
            config = self._get_axis_config(axis_key)
            new_plane = {"axis": config["name"], "value": values.get(axis_key)}

        if len(locked) >= 2:
            axis_key = locked[1]
            config = self._get_axis_config(axis_key)
            new_line = {"axis": config["name"], "value": values.get(axis_key)}

        if len(locked) >= 3:
            axis_key = locked[2]
            config = self._get_axis_config(axis_key)
            new_point = {"axis": config["name"], "value": values.get(axis_key)}

        self.plane = new_plane
        self.line = new_line
        self.point = new_point

    def reset(self):
        """Reset to volume view (unlock all axes)."""
        self.locked_order = []

    def lock_axis(self, axis_key, value=None):
        """Programmatically lock an axis."""
        if axis_key not in ["x", "y", "z"]:
            raise ValueError("axis_key must be 'x', 'y', or 'z'")
        if axis_key not in self.locked_order:
            self.locked_order = self.locked_order + [axis_key]
        if value is not None:
            values = dict(self.axis_values)
            values[axis_key] = value
            self.axis_values = values

    def unlock_axis(self, axis_key):
        """Programmatically unlock an axis."""
        if axis_key in self.locked_order:
            self.locked_order = [a for a in self.locked_order if a != axis_key]
