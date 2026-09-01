from pathlib import Path
import anywidget
import traitlets

_ASSET_DIR = Path(__file__).parent
_JS_SOURCE = (_ASSET_DIR / "recursive_image_widget.js").read_text()
_CSS_SOURCE = (_ASSET_DIR / "recursive_image_widget.css").read_text()


class RecursiveImageWidget(anywidget.AnyWidget):
    """Author an IFS rule by placing transformed copies of a parent image.

    A *placement* is ``[x, y, scale, angle_deg]``: the base box (world-width
    ``BASE``, see JS) scaled by ``scale``, rotated ``angle_deg`` clockwise, and
    centered at ``(x, y)`` in world coords ``[0, 1]``. The ``parent`` placement
    defines the base frame; each child in ``children`` is another placement.
    The recursive rule is, per child, the similarity mapping the parent box onto
    the child box.
    """

    _esm = _JS_SOURCE
    _css = _CSS_SOURCE

    # Stage size in pixels (square).
    size = traitlets.Int(320).tag(sync=True)

    # Parent PNG as a base64 data URL (set from wigglystuff Paint).
    parent_image = traitlets.Unicode("").tag(sync=True)

    # Parent placement [x, y, scale, angle_deg] — defines the base frame.
    parent = traitlets.List(
        traitlets.Float(), default_value=[0.5, 0.5, 1.0, 0.0]
    ).tag(sync=True)

    # One placement per child: [x, y, scale, angle_deg].
    children = traitlets.List(
        traitlets.List(traitlets.Float()), default_value=[]
    ).tag(sync=True)
