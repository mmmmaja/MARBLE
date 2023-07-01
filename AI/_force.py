from abc import abstractmethod
import numpy as np

"""
This script contains the classes that represent the forces acting on the mesh.
there are different types of forces that can be applied to the mesh.

The ForceHandler class is a parent class for all the forces.

    The VolumeForce class is a force that is applied to the whole mesh.
    (Force is specified in the GUI class)
    
    The StimuliForce class is a force that is applied with a stimulus.
    (Force is dependant on the stimulus shape and position)
    
    The CellSpecificForce is a force that is applied to a specific cell in the mesh.
    (Force is specified in the GUI class)
    
    MeshIntersectionForce is the same as CellSpecificForce but we need to map the cell_id to the mesh first

"""


class ForceHandler:

    @abstractmethod
    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        """
        TODO override in subclasses
        :param vertex_coordinates: A 3D coordinates of the vertex in the mesh
        :return: A float value representing the force acting on the vertex
        """


class VolumeForce(ForceHandler):

    # Stable force that is applied to the whole mesh

    def __init__(self, force):
        self.force = force
        super().__init__()

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        return self.force


def is_inside(vertex, vertex_coordinates):
    """
    :param vertex: A 3D vertex in the mesh
    :param vertex_coordinates: 3D coordinates of the rectangle plane
    :return: True if the vertex is inside the plane, False otherwise
    """
    x, y, z = vertex
    x1, y1, z1 = vertex_coordinates[0]
    x2, y2, z2 = vertex_coordinates[1]
    x3, y3, z3 = vertex_coordinates[2]
    x4, y4, z4 = vertex_coordinates[3]

    # Check if the vertex is within the rectangle's boundaries
    if min(x1, x2, x3, x4) <= x <= max(x1, x2, x3, x4) and \
            min(y1, y2, y3, y4) <= y <= max(y1, y2, y3, y4) and \
            min(z1, z2, z3, z4) <= z <= max(z1, z2, z3, z4):
        return True
    else:
        return False


class StimuliForce(ForceHandler):

    def __init__(self, stimuli):
        super().__init__()
        self.stimuli = stimuli

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        force = self.stimuli.calculate_force(vertex_coordinates)
        if force > 0:
            print('force: ', force, 'position: ', vertex_coordinates)
        return force


class MeshIntersectionForce(ForceHandler):

    def __init__(self, cell_id, picker, force):
        super().__init__()
        self.cell_id = cell_id
        self.picker = picker
        self.force = force

        self.force_dict = self.create_force_dict()

    def create_force_dict(self):

        # It will return the ids of the 8 points that make up the hexahedron
        cell_points_ids = self.picker.GetActor().GetMapper().GetInput().GetCell(self.cell_id).GetPointIds()

        # The points list will contain the coordinates of the points that belong to the cell
        points = []
        for i in range(cell_points_ids.GetNumberOfIds()):
            point_id = cell_points_ids.GetId(i)
            # Map the point id to the coordinates of the mesh cells
            points.append(self.picker.GetActor().GetMapper().GetInput().GetPoint(point_id))

        # Remove the bottom layer of points (Points with z coordinate == 0)
        points = [point for point in points if point[2] != 0]

        # Create the force dictionary (vertex_id : force [N])
        force_dict = [[points, self.force]]
        return force_dict

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        for entry in self.force_dict:
            coords, force = entry[0], entry[1]
            if is_inside(vertex_coordinates, coords):
                return force
        return 0.0


class CellSpecificForce(ForceHandler):

    def __init__(self, vertex_ids, force):
        super().__init__()
        self.vertex_ids = vertex_ids
        self.force = force

        self.force_dict = self.create_force_dict()

    def create_force_dict(self):
        return [[self.vertex_ids, self.force]]

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        for entry in self.force_dict:
            coords, force = entry[0], entry[1]
            if is_inside(vertex_coordinates, coords):
                return force
        return 0.0
