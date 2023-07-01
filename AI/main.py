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

    def __init__(self, parent=None, gui=None, fenics=None, *args, **kwargs):

        self.gui = gui
        self.fenics = fenics

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

        super().__init__(*args, **kwargs)

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

            # Get the ids of the vertices that belong to the cell
            vertex_ids = self.fenics.mesh_boost.get_vertex_ids_from_coords(points)

            # Create the force dictionary (vertex_id : force [N])
            force = [
                [vertex_ids, self.gui.FORCE]
            ]
            # Apply the force to the mesh
            apply_force(self.fenics, self.gui, force, relaxation=False)


def apply_force(fenics, gui, force, relaxation=True):
    """
    :param fenics: FENICS class
    :param gui: GUI class
    :param force: numpy array of the form [ [vertex_ids, force] ] or a single force value
    :param relaxation: boolean that indicates if the stress relaxation process should be started
    Function that applies a vertex specific force or volume (stable) force across the whole mesh
    """
    global stress_relaxation_ref

    # Calculate the displacement
    u = fenics.apply_force(force)

    # UPDATE plot and meshes
    fenics.mesh_boost.update_mesh(u)

    gui.draw_mesh(fenics.mesh_boost.current_vtk)
    gui.plotter.update()

    if relaxation:
        # Start the stress relaxation process
        # Stop existing relaxation process
        if stress_relaxation_ref is not None:
            stress_relaxation_ref.stop()

        # if force is a float then vertex_ids = None
        if isinstance(force, float) or isinstance(force, int):
            vertex_ids = None
        else:
            # vertex Ids are entries of the first column of the force array
            vertex_ids = [force[i][0] for i in range(len(force))]
        stress_relaxation_ref = StressRelaxation(
            gui, fenics.mesh_boost, fenics.rank_material, u0=u, F0=gui.FORCE, vertex_ids=vertex_ids
        )
        stress_relaxation_ref.initiate()


def apply_stimuli(fenics, gui, stimuli):
    # vertex_ids : force
    FORCE = []

    # Specifies when not to apply the force (for the speed of the simulation)
    FORCE_LIM = 0.2

    # Get the hexahedral cells
    # It consists of a list of 4 points (3D coordinates) for each cell (top facet)
    hexa_cells = fenics.mesh_boost.vtk_cells_coordinates

    # Check the distance between the centre of the face and the stimuli
    for top_face in hexa_cells:
        # Find the centre of the rectangle face
        face_center = np.mean(top_face, axis=0)
        # Calculate the force from the stimuli to the centre of the face
        force = stimuli.calculate_force(face_center)

        # Apply the force just if the magnitude is greater than the limit
        if np.linalg.norm(force) > FORCE_LIM:
            vertices = fenics.mesh_boost.get_vertex_ids_from_coords(top_face)
            # Transform the coordinates of the cell to the vertex ids
            FORCE.append([vertices, force])

    # Apply the force to the mesh
    apply_force(fenics, gui, FORCE, relaxation=False)


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
                self.fenics, self.gui, relaxation=True,
                force=[[self.fenics.mesh_boost.get_vertex_ids_from_coords(cell.points), self.gui.FORCE]],
            ) if cell is not None else None,
            font_size=10, point_size=30, line_width=5,
            color='white', style='wireframe', through=False
        )

        # Add the event on the press of the space bar, apply the force
        self.gui.plotter.add_key_event('space', lambda: apply_force(
            self.fenics, self.gui, relaxation=True, force=self.gui.FORCE
        ))

        # If the enter button is pressed, the interactive mode is toggled
        self.gui.plotter.add_key_event('m', self.toggle_interactive)
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
                fenics=self.fenics
            ))


if not TERMINAL_OUTPUT:
    # create a text trap and redirect stdout
    text_trap = io.StringIO()
    sys.stdout = text_trap

app = QApplication(sys.argv)
_mesh_boost = GridMesh(30, 30, z_function=wave, layers=3)
_stimuli = Sphere(radius=1.0)


Main(_mesh_boost, _stimuli, rubber)
app.exec_()


"""
TODO interpolation -> less resolution
Do not remove the actor but update it

TODO different stimuli activation
relaxation in a normal activation
ROBOTIC ARM
"""
