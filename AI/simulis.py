import pyvista as pv

class Sphere:

    def __init__(self, radius):
        self.radius = radius
        self.position = [0, 0, 0]

    def create_visualization(self):
        sphere = pv.Sphere(radius=self.radius, center=self.position)
        sphere = sphere.extract_geometry()
        return sphere
