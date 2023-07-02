import time
from abc import abstractmethod
import pyvista as pv
import numpy as np


class Stimuli:

    def __init__(self):
        self.position = np.array([18.0, 18.0, 6.3])
        self.color = '#17d8db'
        self.visualization = self.create_visualization()

    @abstractmethod
    def create_visualization(self) -> None:
        """
        TODO override in subclasses
        :return: The mesh object in vtk format
        """

    @abstractmethod
    def calculate_pressure(self, point: np.ndarray) -> float:
        """
        Calculate the force exerted by the stimulus on a point in space.
        Assumption: The pressure decreases as the distance from the object increases.

        TODO override in subclasses

        :param point: A 3D point in space.
        :return: A float value representing the pressure between 0 and 1 (so that it can be scaled later)
        """

    @abstractmethod
    def get_area(self) -> float:
        """
        :return: The area of the stimulus (face acting on the mesh)
        """

    def move_with_key(self, key):
        # Move the stimulus with the arrow keys
        position_dt = 0.15
        position_update = np.array([0.0, 0.0, 0.0])

        if key == 'w':
            # Move the object forward
            position_update[1] += position_dt
        elif key == 's':
            # Move the object backward
            position_update[1] -= position_dt
        elif key == 'a':
            # Move the object left
            position_update[0] -= position_dt
        elif key == 'd':
            # Move the object right
            position_update[0] += position_dt

        # Add the events for moving along z axis
        elif key == 'plus':
            # Move the object up
            position_update[2] += position_dt
        elif key == 'minus':
            # Move the object down
            position_update[2] -= position_dt

        if np.linalg.norm(position_update) > 0:
            # Only move if there is a change in position
            self.move(position_update)
            return True
        else:
            return False

    def move(self, position_update):

        self.position += position_update
        # translate the visualization
        self.visualization.translate(position_update)


class Sphere(Stimuli):

    def __init__(self, radius):
        self.radius = radius
        self.color = '#62fff8'
        super().__init__()

    def create_visualization(self):
        # Return pyvista sphere
        sphere = pv.Sphere(
            radius=self.radius, center=self.position,
            # theta resolution is the number of points in the longitude direction.
            # phi resolution is the number of points in the latitude direction.
            theta_resolution=10, phi_resolution=10
        )
        return sphere

    def get_area(self):
        return None

    def calculate_pressure(self, point: np.ndarray) -> float:
        """
        Calculate the pressure exerted by the stimulus on a point in space.
        :param point: A 3D point in space.
        :return: A float value representing the pressure.
        """
        distance = np.linalg.norm(self.position - point)
        # Check if the point is within the boundary of the shape of the stimulus
        if distance <= self.radius:
            if distance == 0:  # Avoid division by zero
                return 1.0
            # force is the inverse of the distance to the sphere's center
            return 1.0 / distance
        else:
            return 0.0


class Cylinder(Stimuli):
    """
    Flat face of the cylinder is facing the mesh
    Z axis is the height of the cylinder
    """

    def __init__(self, radius, height):
        self.radius = radius
        self.height = height
        self.color = 'e874ff'
        super().__init__()

    def create_visualization(self):
        # Return pyvista cylinder
        direction = np.array([0, 0, 1])
        cylinder = pv.Cylinder(
            radius=self.radius, height=self.height, center=self.position,
            resolution=50, direction=direction
        )
        # translate the cylinder so that the flat face is facing the mesh

        return cylinder

    def calculate_pressure(self, point: np.ndarray) -> float:
        # Calculate 2D distance in the x-y plane
        distance = np.linalg.norm(self.position[:2] - point[:2])

        # Check if the point is under the flat face of the cylinder
        if distance <= self.radius and abs(point[2] - self.position[2]) <= self.height / 2:
            # distribute the pressure equally within the circular boundary
            return 1.0
        else:
            return 0.0

    def get_area(self):
        return np.pi * self.radius ** 2


class Cuboid(Stimuli):

    def __init__(self, width, length, height):
        self.width = width
        self.length = length
        self.height = height
        self.color = '#FF00FF'

        super().__init__()

    def create_visualization(self):
        # Return pyvista cuboid
        cube = pv.Cube(
            center=self.position,
            x_length=self.width, y_length=self.length, z_length=self.height
        )
        return cube

    def calculate_pressure(self, point: np.ndarray) -> float:

        # Get the absolute distance between the point and the center of the cuboid
        x, y, z = abs(self.position - point)

        # Check if the point is within the boundary of the shape of the stimulus
        if x <= self.width / 2 and y <= self.length / 2 and z <= self.height / 2:
            return 1.0
            # Force is the inverse of the minimum distance to the faces of the cuboid.
            # return min([1 / x, 1 / y, 1 / z]) if x > 0 and y > 0 and z > 0 else 1.0
        else:
            return 0.0

    def get_area(self):
        return self.width * self.length
