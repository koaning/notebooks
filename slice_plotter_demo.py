import numpy as np
from slice_plotter_widget import SlicePlotterWidget

def create_example():
    # Generate some 3D surface data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Function: z = sin(sqrt(x^2 + y^2))
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    # Create the widget
    widget = SlicePlotterWidget()
    
    # Set data (convert numpy arrays to lists)
    widget.set_data(x.tolist(), y.tolist(), Z.tolist())
    
    return widget

if __name__ == "__main__":
    print("Creating SlicePlotterWidget instance...")
    widget = create_example()
    print("Widget created successfully.")
    print("To view this widget, run this in a Jupyter Notebook:")
    print("  from slice_plotter_demo import create_example")
    print("  widget = create_example()")
    print("  widget")
