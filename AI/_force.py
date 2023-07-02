from abc import abstractmethod
import numpy as np

"""
This script contains the classes that represent the forces acting on the mesh.
there are different types of forces that can be applied to the mesh.

The force (force strength) is specified in the GUI class.
Not sure if this  is the right way to do it tho...

The ForceHandler class is a parent class for all the forces.

    The VolumeForce class is a force that is applied to the whole mesh.
    
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
        TODO experiment with vector [x, y, z] force
        """


class VolumeForce(ForceHandler):

    def __init__(self, force):
        """
        Stable force that is applied to the whole mesh
        :param force: float value representing the force strength
        """
        self.force = force
        super().__init__()

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        # Force is the same for all the vertices
        return self.force


def is_inside(vertex, vertex_coordinates):
    """
    :param vertex: A 3D vertex in the mesh
    :param vertex_coordinates: 3D coordinates of the rectangle plane
    :return: True if the vertex is inside the plane, False otherwise
    """
    if len(vertex_coordinates) != 4:
        return False

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

    def __init__(self, stimuli, force_strength):
        """
        Force that is applied to the mesh with a stimulus
        :param stimuli: Stimuli object that has its own pressure function specified
        :param force_strength: float value representing the force strength
        """
        super().__init__()
        self.stimuli = stimuli
        self.force_strength = force_strength

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        # Scale the force with the force strength
        force = self.stimuli.calculate_force(vertex_coordinates) * self.force_strength
        return force


class MeshIntersectionForce(ForceHandler):

    def __init__(self, cell_id, picker, force):
        """
        Force that is applied to a specific cell in the mesh
        :param cell_id: ID of the cell in the mesh that was picked
        :param picker: vtkCellPicker object that was used to pick the cell
        :param force: float value representing the force strength
        """
        super().__init__()
        self.cell_id = cell_id
        self.picker = picker
        self.force = force

        self.affected_points = self.get_affected_points()

    def get_affected_points(self):

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
        return points

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        if is_inside(vertex_coordinates, self.affected_points):
            return self.force
        return 0.0


class CellSpecificForce(ForceHandler):

    def __init__(self, affected_points, force):
        super().__init__()
        self.affected_points = affected_points
        self.force = force

    def get_force(self, vertex_coordinates: np.ndarray) -> float:
        if is_inside(vertex_coordinates, self.affected_points):
            return self.force
        return 0.0
