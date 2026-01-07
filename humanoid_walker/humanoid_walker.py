import anywidget
import traitlets
import pathlib

class HumanoidWalker(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "humanoid_walker.js"
    _css = pathlib.Path(__file__).parent / "humanoid_walker.css"

    # Inputs (Python -> JS)
    torques = traitlets.Dict(default_value={
        'left_hip': 0.0,
        'left_knee': 0.0,
        'right_hip': 0.0,
        'right_knee': 0.0
    }).tag(sync=True)
    
    reset = traitlets.Bool(False).tag(sync=True)

    # Outputs (JS -> Python)
    state = traitlets.Dict(default_value={}).tag(sync=True)

    def apply_action(self, action):
        """
        Apply action dictionary to torques.
        action: dict with keys matching torques.
        """
        self.torques = action
        
    def reset_simulation(self):
        self.reset = True
