from abc import abstractmethod
import pyvista as pv
import numpy as np


class Stimuli:

    def __init__(self):
        self.position = np.array([-2.2, -2.2, 4.4])
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

    def recompute_position(self, picker, cell_id):
        # It will return the ids of the 8 points that make up the hexahedron
        cell_points_ids = picker.GetActor().GetMapper().GetInput().GetCell(cell_id).GetPointIds()

        # The points list will contain the coordinates of the points that belong to the cell
        points = []
        for i in range(cell_points_ids.GetNumberOfIds()):
            point_id = cell_points_ids.GetId(i)
            # Map the point id to the coordinates of the mesh cells
            points.append(picker.GetActor().GetMapper().GetInput().GetPoint(point_id))

        # Remove the bottom layer of points (Points with z coordinate == 0)
        points = [point for point in points if point[2] != 0]
        if len(points) == 0:
            return False

        # Get the average of the points
        self.position = np.mean(points, axis=0)
        return True


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
            theta_resolution=20, phi_resolution=20
        )
        return sphere

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
            # INVERSE SQUARE LAW
            return 1 / (distance ** 2)
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
            resolution=70, direction=direction
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
        else:
            return 0.0
