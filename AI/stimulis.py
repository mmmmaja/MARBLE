from abc import abstractmethod
import pyvista as pv
import numpy as np


class Stimuli:

    def __init__(self):
        self.position = np.array([5.0, 5.0, 10.0])
        self.color = '#17d8db'

    @abstractmethod
    def get_visualization(self) -> None:
        """
        TODO override in subclasses
        :return: The mesh object in vtk format
        """

    @abstractmethod
    def calculate_force(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the force exerted by the stimulus on a point in space.
        TODO override in subclasses
        :param point: A 3D point in space.
        :return: A 3D vector representing the force.
        """


class Sphere(Stimuli):

    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.color = '#62fff8'

        # FIXME later, this is just for testing
        self.position = np.array([5.0, 5.0, 4.2])

    def get_visualization(self):
        # Return pyvista sphere
        sphere = pv.Sphere(
            radius=self.radius, center=self.position,
            # theta resolution is the number of points in the longitude direction.
            # phi resolution is the number of points in the latitude direction.
            theta_resolution=10, phi_resolution=10
        )
        return sphere

    def calculate_force(self, point: np.ndarray) -> np.ndarray:
        direction = self.position - point
        distance = np.linalg.norm(direction)
        force = direction / distance ** 3  # Assuming inverse square law

        # FIXME later, this is just for testing
        return force[2]


class Cylinder(Stimuli):

    def __init__(self, radius, height):
        super().__init__()
        self.radius = radius
        self.height = height
        self.color = 'e874ff'

    def get_visualization(self):
        # Return pyvista cylinder
        direction = np.array([0, 0, 1])
        cylinder = pv.Cylinder(
            radius=self.radius, height=self.height, center=self.position,
            resolution=50, direction=direction
        )
        return cylinder


class Cuboid(Stimuli):

    def __init__(self, width, length, height):
        super().__init__()

        self.width = width
        self.length = length
        self.height = height
        self.color = '#FF00FF'

    def get_visualization(self):
        # Return pyvista cuboid
        cube = pv.Cube(
            center=self.position,
            x_length=self.width, y_length=self.length, z_length=self.height
        )
        return cube
