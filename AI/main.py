import vtk

from AI.gui import GUIHandler
from PyQt5.QtWidgets import QApplication
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from AI.stress_relaxation import StressRelaxation
from AI.simulis import *
from PyQt5.QtCore import QObject, pyqtSignal


class NoRotateStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None, gui=None, *args, **kwargs):
        self.gui = gui
        self.mouse_pressed = False

        self.picker = vtk.vtkCellPicker()
        self.picker.AddPickList(self.gui.mesh_actor)
        print("Picker added")

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
            # TODO apply force to cell
            # Here you can activate your cell



FORCE = 2.5


class Main:

    def __init__(self, fenics, stimuli):
        self.fenics = fenics
        self.stimuli = stimuli
        # Create the GUI
        self.gui = self.create_GUI()

    def create_GUI(self):
        gui = GUIHandler(self.fenics.mesh_boost, self.fenics.rank_material, self.stimuli)
        # Add the interactive events
        gui.plotter.add_key_event('space', self.apply_force)
        gui.plotter.enable_cell_picking(
            callback=self.apply_force,
            point_size=30, line_width=5, font_size=10,
            style='wireframe', color='white', through=False
        )
        # If the enter button is pressed, the interactive mode is toggled
        gui.plotter.add_key_event('m', self.toggle_interactive)
        gui.plotter.show()
        return gui

    def apply_force(self, cell=None, F=FORCE):
        """
        Function that applies a vertex specific force or volume (stable) force across the whole mesh
        """
        if cell is not None:
            vertex_ids = self.fenics.mesh_boost.get_vertex_ids_from_coords(cell.points)
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
        u = self.fenics.apply_force(vertex_ids, F)
        # Update plot and meshes
        self.fenics.mesh_boost.update_vtk(u)
        self.gui.draw_mesh()

        # # Start the stress relaxation process
        # stress_relaxation = StressRelaxation(
        #     self.gui, self.fenics, u0=u, F0=F, vertex_ids=vertex_ids
        # )
        # stress_relaxation.initiate()

    def toggle_interactive(self):
        print("Interactive mode toggled")

        if self.gui.plotter.interactor.GetInteractorStyle().__class__ == NoRotateStyle:
            self.gui.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.gui.update_mode_text("Interactive")
        else:
            self.gui.update_mode_text("Activation")
            self.gui.plotter.interactor.SetInteractorStyle(NoRotateStyle(
                gui=self.gui
            ))


app = QApplication(sys.argv)
_stimuli = Sphere(radius=1.5)
_mesh_boost = GridMesh(30, 30, z_function=concave)
_fenics = FENICS(_mesh_boost, rubber)
Main(_fenics, _stimuli)
app.exec_()
