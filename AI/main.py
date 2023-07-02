import vtk
from PyQt5.QtWidgets import QApplication
from AI.GUI_handler import GUI
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from AI.stimulis import *
from AI.stress_script import StressRelaxation
import io
import sys
from AI._pressure import *


# Set to True to enable the terminal output,
# otherwise the output will be redirected to the log file (maybe it is faster this way)
TERMINAL_OUTPUT = True

# I need to hold the reference to the timer class and destroy it
# when the simulation of the relaxation process is over
stress_relaxation_ref = None


class NoRotateStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    This class was added to override the default mouse events of the vtkInteractorStyleTrackballCamera class.
    It disables the rotation of the mesh when the left mouse button is pressed.
    Triggered in Activation mode
    """

    def __init__(self, parent=None, gui=None, fenics=None, stimuli=None, *args, **kwargs):
        self.gui = gui
        self.fenics = fenics
        self.stimuli = stimuli

        # Mouse pressed flag for mesh activation
        self.mouse_pressed = False

        # Create a cell picker for the mesh (to map the mouse coordinates to the mesh)
        self.picker = vtk.vtkCellPicker()
        self.picker.AddPickList(self.gui.mesh_actor)

        # Override the default mouse events (disable rotation)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("RightButtonPressEvent", self.right_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("KeyPressEvent", self.key_event)
        self.AddObserver("CharEvent", self.key_event)

        super().__init__(*args, **kwargs)

    def key_event(self, obj, event):
        # This event will change the position of the stimuli

        # Get the key that was pressed
        key = self.GetInteractor().GetKeySym()
        if self.stimuli.move_with_key(key):
            # Update the visualization
            apply_stimuli(self.fenics, self.gui, self.stimuli)
            self.gui.draw_stimuli()

    def left_button_press_event(self, obj, event):
        self.mouse_pressed = True

    def left_button_release_event(self, obj, event):
        self.mouse_pressed = False

    def middle_button_press_event(self, obj, event):
        # Disable the middle button events
        pass

    def right_button_press_event(self, obj, event):
        # Disable the right button events
        pass

    def mouse_move_event(self, obj, event):
        """
        TODO adjust to stimuli
        Function that is triggered when the mouse is moved.
        If self.mouse_pressed is True, it updates the position of the stimuli.
        """
        if self.mouse_pressed:
            # Get the mouse coordinates and pick the cell
            x, y = self.GetInteractor().GetEventPosition()
            self.pick_cell(x, y)

    def pick_cell(self, x, y):
        """
        Function that picks the cell that was clicked and applies a force to it.
        :param x: x coordinate of the mouse
        :param y: y coordinate of the mouse
        """

        # Pick the cell that was clicked
        self.picker.Pick(x, y, 0, self.gui.plotter.renderer)
        cell_id = self.picker.GetCellId()

        # If the cell exists
        if cell_id != -1:
            # It will return the ids of the 8 points that make up the hexahedron
            cell_points_ids = self.picker.GetActor().GetMapper().GetInput().GetCell(cell_id).GetPointIds()

            # The points list will contain the coordinates of the points that belong to the cell
            points = []
            for i in range(cell_points_ids.GetNumberOfIds()):
                point_id = cell_points_ids.GetId(i)
                # Map the point id to the coordinates of the mesh cells
                points.append(self.picker.GetActor().GetMapper().GetInput().GetPoint(point_id))

            # Remove the bottom layer of points (Points with z coordinate == 0)
            points = [point for point in points if point[2] != 0]
            if len(points) == 0:
                return

            # Get the average of the points
            average_point = np.mean(points, axis=0)

            # Update the position of the stimuli to the clicked cell
            self.stimuli.position = average_point
            
            force_handler = StimuliPressure(self.stimuli, self.gui.FORCE, self.fenics.rank_material)
            # Apply the force to the mesh
            apply_force(self.fenics, self.gui, force_handler, relaxation=False)


def apply_force(fenics, gui, force_handler, relaxation=True):
    """
    :param fenics: FENICS class
    :param gui: GUI class
    :param force_handler: ForceHandler class instance
    :param relaxation: boolean that indicates if the stress relaxation process should be started
    Function that applies a vertex specific force or volume (stable) force across the whole mesh
    """
    global stress_relaxation_ref

    # Calculate the displacement
    u = fenics.apply_force(force_handler)

    # UPDATE plot and meshes
    fenics.mesh_boost.update_mesh(u)

    gui.draw_mesh(fenics.mesh_boost.current_vtk)
    gui.plotter.update()

    if relaxation:
        # Start the stress relaxation process
        # Stop existing relaxation process
        if stress_relaxation_ref is not None:
            stress_relaxation_ref.stop()

        stress_relaxation_ref = StressRelaxation(
            gui, fenics.mesh_boost, fenics.rank_material, u0=u, F0=gui.FORCE
        )
        stress_relaxation_ref.initiate()


def apply_stimuli(fenics, gui, stimuli):
    force_handler = StimuliPressure(stimuli, gui.FORCE, fenics.rank_material)
    # Apply the force to the mesh
    apply_force(fenics, gui, force_handler, relaxation=False)


class Main:

    def __init__(self, mesh_boost, stimuli, rank_material):
        self.stimuli = stimuli
        self.fenics = FENICS(mesh_boost, rank_material)

        self.gui = GUI(mesh_boost.current_vtk, rank_material, stimuli)
        self.add_interactive_events()

    def add_interactive_events(self):
        # Enable cell picking
        self.gui.plotter.enable_cell_picking(
            # if cell is not none, apply force to the cell
            callback=lambda cell:
            apply_force(
                self.fenics, self.gui, CellSpecificPressure(cell.points, self.gui.FORCE), relaxation=True,
            ) if cell is not None else None,
            font_size=10, point_size=30, line_width=5,
            color='white', style='wireframe', through=False
        )

        # Add the event on the press of the space bar, apply the force
        self.gui.plotter.add_key_event('space', lambda: apply_force(
            self.fenics, self.gui, VolumePressure(self.gui.FORCE), relaxation=True,
        ))

        # If the enter button is pressed, the interactive mode is toggled
        self.gui.plotter.add_key_event('Control_L', self.toggle_interactive)
        self.gui.plotter.show()

    def toggle_interactive(self):
        print("Interactive mode toggled")

        if self.gui.plotter.interactor.GetInteractorStyle().__class__ == NoRotateStyle:
            self.gui.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            self.gui.add_mode_text("Interactive")
        else:
            self.gui.add_mode_text("Activation")
            self.gui.plotter.interactor.SetInteractorStyle(NoRotateStyle(
                gui=self.gui,
                fenics=self.fenics,
                stimuli=self.stimuli
            ))


if not TERMINAL_OUTPUT:
    # create a text trap and redirect stdout
    text_trap = io.StringIO()
    sys.stdout = text_trap

app = QApplication(sys.argv)

_mesh_boost = GridMesh(30, 30, z_function=flat, layers=3)
# _mesh_boost = ArmMesh()

# _stimuli = Sphere(radius=2.0)
# _stimuli = Cylinder(radius=2.0, height=1.0)
_stimuli = Cuboid(7.0, 4.0, 2.0)


Main(_mesh_boost, _stimuli, rubber)
app.exec_()


"""
TODO interpolation -> less resolution
Do not remove the actor but update it

TODO different stimuli activation
relaxation in a normal activation
ROBOTIC ARM
"""
