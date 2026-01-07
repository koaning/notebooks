import anywidget
import traitlets
import pathlib

class SlicePlotterWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "slice_plotter_widget.js"
    _css = pathlib.Path(__file__).parent / "slice_plotter_widget.css"

    # Data
    x = traitlets.List().tag(sync=True)
    y = traitlets.List().tag(sync=True)
    z = traitlets.List().tag(sync=True) # 2D array (list of lists)
    
    # State
    slice_index = traitlets.Int(0).tag(sync=True)
    slice_axis = traitlets.Unicode('x').tag(sync=True) # 'x' or 'y'
    
    def set_data(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
