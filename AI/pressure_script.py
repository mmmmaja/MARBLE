from abc import abstractmethod
import numpy as np
from AI.stress_script import StressRelaxation

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


class PressureHandler:

    @abstractmethod
    def get_pressure(self, vertex_coordinates: np.ndarray) -> float:
        """
        TODO override in subclasses
        :param vertex_coordinates: A 3D coordinates of the vertex in the mesh
        :return: A float value representing the pressure acting on the vertex

        Inverse square law
        """


class VolumePressure(PressureHandler):

    def __init__(self, pressure_strength):
        """
        Stable force that is applied to the whole mesh
        :param pressure_strength: float value representing the strength of the pressure applied
        """
        self.pressure_strength = pressure_strength
        super().__init__()

    def get_pressure(self, vertex_coordinates: np.ndarray) -> float:
        # Pressure is the same for all the vertices
        return self.pressure_strength


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


class StimuliPressure(PressureHandler):

    def __init__(self, stimuli, pressure_strength, material):
        """
        Force that is applied to the mesh with a stimulus
        :param stimuli: Stimuli object that has its own pressure function specified
        :param pressure_strength: float value representing the strength of the pressure applied
        """
        super().__init__()
        self.stimuli = stimuli
        self.pressure_strength = pressure_strength
        self.material = material

    def get_pressure(self, vertex_coordinates: np.ndarray) -> float:
        # Scale the force with the force strength
        pressure = self.stimuli.calculate_pressure(vertex_coordinates) * self.pressure_strength

        return pressure


class MeshIntersectionPressure(PressureHandler):

    def __init__(self, cell_id, picker, pressure_strength):
        """
        Force that is applied to a specific cell in the mesh
        :param cell_id: ID of the cell in the mesh that was picked
        :param picker: vtkCellPicker object that was used to pick the cell
        :param pressure_strength: float value representing the strength of the pressure applied
        """
        super().__init__()
        self.cell_id = cell_id
        self.picker = picker
        self.pressure_strength = pressure_strength

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

    def get_pressure(self, vertex_coordinates: np.ndarray) -> float:
        if is_inside(vertex_coordinates, self.affected_points):
            return self.pressure_strength
        return 0.0


class CellSpecificPressure(PressureHandler):

    def __init__(self, affected_points, pressure_strength):
        super().__init__()
        self.affected_points = affected_points
        self.pressure_strength = pressure_strength

    def get_pressure(self, vertex_coordinates: np.ndarray) -> float:
        if is_inside(vertex_coordinates, self.affected_points):
            return self.pressure_strength
        return 0.0


"""
Handle applying the pressure to the mesh
"""

# I need to hold the reference to the timer class and destroy it
# when the simulation of the relaxation process is over
stress_relaxation_ref = None


def apply_volume_pressure(fenics, gui, relaxation=True):
    force_handler = VolumePressure(gui.PRESSURE)
    apply_pressure(fenics, gui, force_handler, relaxation)


def apply_stimuli_pressure(fenics, gui, stimuli, picker, cell_id, relaxation=False):
    if stimuli.recompute_position(picker, cell_id):
        force_handler = StimuliPressure(stimuli, gui.PRESSURE, fenics.rank_material)
        apply_pressure(fenics, gui, force_handler, relaxation)


def apply_cell_specific_pressure(fenics, gui, cell, relaxation=True):
    if not cell:
        return
    force_handler = CellSpecificPressure(cell.points, gui.PRESSURE)
    apply_pressure(fenics, gui, force_handler, relaxation)


def apply_pressure(fenics, gui, force_handler, relaxation):
    """
    :param fenics: FENICS class
    :param gui: GUI class
    :param force_handler: ForceHandler class instance
    :param relaxation: boolean that indicates if the stress relaxation process should be started
    Function that applies a vertex specific force or volume (stable) force across the whole mesh
    """
    global stress_relaxation_ref

    # Calculate the displacement
    u = fenics.apply_pressure(force_handler)

    # UPDATE plot and meshes
    fenics.mesh_boost.update_mesh(u)
    gui.draw_mesh(fenics.mesh_boost.current_vtk)
    gui.draw_sensors()
    gui.plotter.update()

    if relaxation:
        # Start the stress relaxation process
        # Stop existing relaxation process
        if stress_relaxation_ref is not None:
            stress_relaxation_ref.stop()

        stress_relaxation_ref = StressRelaxation(
            gui, fenics.mesh_boost, fenics.rank_material, u0=u, F0=gui.PRESSURE
        )
        stress_relaxation_ref.initiate()
