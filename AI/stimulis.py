from abc import abstractmethod
import pyvista as pv
import numpy as np


class Stimuli:

    def __init__(self):
        self.position = np.array([18.0, 18.0, 4.0])
        self.color = '#17d8db'

    @abstractmethod
    def get_visualization(self) -> None:
        """
        TODO override in subclasses
        :return: The mesh object in vtk format
        """

    @abstractmethod
    def calculate_force(self, point: np.ndarray) -> float:
        """
        Calculate the force exerted by the stimulus on a point in space.
        TODO override in subclasses

        :param point: A 3D point in space.
        :return: A float value representing the force.
        """


class Sphere(Stimuli):

    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.color = '#62fff8'

    def get_visualization(self):
        # Return pyvista sphere
        sphere = pv.Sphere(
            radius=self.radius, center=self.position,
            # theta resolution is the number of points in the longitude direction.
            # phi resolution is the number of points in the latitude direction.
            theta_resolution=10, phi_resolution=10
        )
        return sphere

    def calculate_force(self, point: np.ndarray) -> float:
        """
        Calculate the force exerted by the stimulus on a point in space.
        :param point: A 3D point in space.
        :return: A float value representing the force.
        """
        distance = np.linalg.norm(self.position - point)
        # Check if the point is within the boundary of the shape of the stimulus
        if distance <= self.radius:
            # force is the inverse of the distance to the sphere's center
            return 1 / distance if distance > 0 else float('inf')
        else:
            return 0.0


class Cylinder(Stimuli):

    """
    Flat face of the cylinder is facing the mesh
    Z axis is the height of the cylinder
    """

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

    def calculate_force(self, point: np.ndarray) -> float:
        # Calculate 2D distance in the x-y plane
        distance = np.linalg.norm(self.position[:2] - point[:2])

        # Check if the point is under the flat face of the cylinder
        if distance <= self.radius and abs(point[2] - self.position[2]) <= self.height / 2:
            # distribute the pressure equally within the circular boundary
            # Intensity??
            return 1.0
        else:
            return 0.0


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

    def calculate_force(self, point: np.ndarray) -> float:

        # Get the absolute distance between the point and the center of the cuboid
        x, y, z = abs(self.position - point)

        # Check if the point is within the boundary of the shape of the stimulus
        if x <= self.width / 2 and y <= self.length / 2 and z <= self.height / 2:
            # Force is the inverse of the minimum distance to the faces of the cuboid.
            return min([1 / x, 1 / y, 1 / z]) if x > 0 and y > 0 and z > 0 else float('inf')
        else:
            return 0.0
