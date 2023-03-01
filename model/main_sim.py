"""
Create a model that will be used for simulations. Spatial calibration algorithm will be tested on it.
For simplicity start with map of 2D rectangle grid then expand it to more complex shapes (meshes).

Input: dimension of the rectangle, radius of sensors, shape of stimulating object

Generate unique object for each sensor, algorithm will know correct position of each sensor
Dragging OBJ on the grid will generate pressure input
How to determine pressure? Noise(?)
Create method for generation of large number of data samples
"""


# Data in .xyz format(?)
# point cloud in .txt format, which contains the X, Y, and Z coordinates of each point
# TriangleMesh(vertices: open3d.cpu.pybind.utility.Vector3dVector, triangles: open3d.cpu.pybind.utility.Vector3iVector)


# Ask for the dimension input from the user
# dim = input("Enter [width, height] of the mesh: ").split(',')
# _mesh = create_mesh(int(dim[0]), int(dim[1]))

"""
Press ‘h’ for more options. Some commonly used controls are:-
-Left button + drag : Rotate.
-Ctrl + left button + drag : Translate.
-Wheel button + drag : Translate.
-Shift + left button + drag : Roll.
-Wheel : Zoom in/out.
-R : Reset view point.
-L : Turn on/off lighting.
-Q : Close windows
"""

# Window choose your fighter for the stimuli
# open general window, show instructions on the right

from choose_your_fighter import initialize, preset

# initialize()
preset()