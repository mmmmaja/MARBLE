import numpy as np
import pyvista as pv


class Sphere:

    def __init__(self, radius):
        self.radius = radius
        self.position = np.zeros(3)

    def get_visualization(self):
        sphere = pv.Sphere(radius=self.radius, center=self.position)
        return sphere
