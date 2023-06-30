import sys

import vtk
from pyvista import Cell

from AI.gui import GUIHandler
from PyQt5.QtWidgets import QApplication
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from AI.simulis import *
from AI.stress_relaxation import StressRelaxation

FORCE = 2.5


def apply_force(fenics, gui, cell_coords=None, F=FORCE, relaxation=False):
    """
    Function that applies a vertex specific force or volume (stable) force across the whole mesh
    """
    if cell_coords is not None:
        vertex_ids = fenics.mesh_boost.get_vertex_ids_from_coords(cell_coords)
        print("Triggered cell force")
        # If the list is empty
        if len(vertex_ids) == 0:
            return
    else:
        vertex_ids = None
        print("Triggered volume force")

    # Immediately apply the force to the body (might change it later and push it to the loop)
    print("FORCE applied: ", F)
    # Calculate the displacement
    u = fenics.apply_force(vertex_ids, F)
    # Update plot and meshes
    fenics.mesh_boost.update_vtk(u)
    gui.draw_mesh()

    if relaxation:
        # Start the stress relaxation process
        stress_relaxation = StressRelaxation(
            gui, fenics, u0=u, F0=F, vertex_ids=vertex_ids
        )
        stress_relaxation.initiate()


class NoRotateStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None, gui=None, fenics=None, *args, **kwargs):
        self.gui = gui
        self.fenics = fenics

        self.mouse_pressed = False

        # Create a cell picker for the mesh
        self.picker = vtk.vtkCellPicker()
        self.picker.AddPickList(self.gui.mesh_actor)

        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)

        super().__init__(*args, **kwargs)

    def left_button_press_event(self, obj, event):
        self.mouse_pressed = True
        self.OnLeftButtonDown()  # Let VTK handle any remaining left button down operations

    def left_button_release_event(self, obj, event):
        self.mouse_pressed = False
        self.OnLeftButtonUp()  # Let VTK handle any remaining left button up operations

    def middle_button_press_event(self, obj, event):
        pass

    def right_button_press_event(self, obj, event):
        pass

    def mouse_move_event(self, obj, event):
        """
        Function that is triggered when the mouse is moved.
        If `is_dragging` is True, it updates the position of the stimuli.
        """
        pass
        if self.mouse_pressed:
            x, y = self.GetInteractor().GetEventPosition()
            self.pick_cell(x, y)

    def pick_cell(self, x, y):
        self.picker.Pick(x, y, 0, self.gui.plotter.renderer)
        cell_id = self.picker.GetCellId()
        if cell_id != -1:
            print(f"Cell {cell_id} picked.")

            # It will return the ids of the 8 points that make up the hexahedron
            cell_points_ids = self.picker.GetActor().GetMapper().GetInput().GetCell(cell_id).GetPointIds()

            # The points list will contain the coordinates of the points that belong to the cell
            points = []
            for i in range(cell_points_ids.GetNumberOfIds()):
                point_id = cell_points_ids.GetId(i)
                points.append(self.picker.GetActor().GetMapper().GetInput().GetPoint(point_id))

            # Remove the bottom layer of points (Points with z coordinate == 0)
            points = [point for point in points if point[2] != 0]
            apply_force(self.fenics, self.gui, points)


class Main:

    def __init__(self, fenics, stimuli):
        self.fenics = fenics
        self.stimuli = stimuli
        # Create the GUI
        self.gui = self.create_GUI()

    def create_GUI(self):
        gui = GUIHandler(self.fenics.mesh_boost, self.fenics.rank_material, self.stimuli)
        # Add the interactive events
        gui.plotter.add_key_event('space', lambda: apply_force(self.fenics, gui))
        gui.plotter.enable_cell_picking(
            callback=lambda cell: apply_force(self.fenics, gui, cell_coords=cell.points),
            point_size=30, line_width=5, font_size=10,
            style='wireframe', color='white', through=False
        )
        # If the enter button is pressed, the interactive mode is toggled
        gui.plotter.add_key_event('m', self.toggle_interactive)
        gui.plotter.show()
        return gui

    def toggle_interactive(self):
        print("Interactive mode toggled")

        if self.gui.plotter.interactor.GetInteractorStyle().__class__ == NoRotateStyle:
            self.gui.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.gui.update_mode_text("Interactive")
        else:
            self.gui.update_mode_text("Activation")
            self.gui.plotter.interactor.SetInteractorStyle(NoRotateStyle(
                gui=self.gui,
                fenics=self.fenics,
            ))


app = QApplication(sys.argv)
_stimuli = Sphere(radius=1.5)
_mesh_boost = GridMesh(30, 30, z_function=concave)
_fenics = FENICS(_mesh_boost, rubber)
Main(_fenics, _stimuli)
app.exec_()
