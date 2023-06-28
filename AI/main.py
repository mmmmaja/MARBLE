import sys
import vtk
from PyQt5.QtWidgets import QApplication
from AI.GUI_handler import GUI
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from AI.stress_script import StressRelaxation

FORCE = 2.5

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
        Function that is triggered when the mouse is moved.
        If self.mouse_pressed is True, it updates the position of the stimuli.
        """
        if self.mouse_pressed:
            # Get the mouse coordinates and pick the cell
            x, y = self.GetInteractor().GetEventPosition()
            self.pick_cell(x, y)

    def pick_cell(self, x, y):
        # Pick the cell that was clicked
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
            apply_force(self.fenics, self.gui, points, relaxation=False)


def apply_force(fenics, gui, cell_coords=None, F=FORCE, relaxation=True):
    """
    Function that applies a vertex specific force or volume (stable) force across the whole mesh
    """

    global stress_relaxation_ref

    if cell_coords is not None:
        vertex_ids = fenics.mesh_boost.get_vertex_ids_from_coords(cell_coords)
        print("Triggered cell force")

        # If the list is empty
        if len(vertex_ids) == 0:
            print("No vertices found")
            return

    else:
        vertex_ids = None
        print("Triggered volume force")

    # Immediately apply the force to the body (might change it later and push it to the loop)
    print("FORCE applied: ", F)
    # Calculate the displacement
    u = fenics.apply_force(vertex_ids, F)
    # Update plot and meshes
    gui.update(u, fenics.mesh_boost)

    if relaxation:
        # Start the stress relaxation process
        # Stop existing relaxation process
        if stress_relaxation_ref is not None:
            stress_relaxation_ref.stop()
        stress_relaxation_ref = StressRelaxation(
            gui, fenics, u0=u, F0=F, vertex_ids=vertex_ids
        )
        stress_relaxation_ref.initiate()


class Main:

    def __init__(self, fenics):
        self.fenics = fenics

        self.gui = GUI(self.fenics.rank_material, fenics.mesh_boost.current_vtk)
        self.add_interactive_events()

        """
        Timer for updating the state during fenics calculations (Starts a new thread)
        Documentation:
        https://doc.qt.io/qtforpython-5/PySide2/QtCore/QTimer.html
        """

    def add_interactive_events(self):
        # Enable cell picking
        self.gui.plotter.enable_cell_picking(
            callback=lambda cell_id: apply_force(self.fenics, self.gui, cell_id.points),
            font_size=10,
            color='white',
            point_size=30,
            style='wireframe',
            line_width=5,
            through=False
        )

        # Add the event on the press of the space bar, apply the force
        self.gui.plotter.add_key_event(
            'space', lambda: apply_force(self.fenics, self.gui)
        )

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
                fenics=self.fenics,
            ))


app = QApplication(sys.argv)
_mesh_boost = GridMesh(30, 30, z_function=concave)
_fenics = FENICS(_mesh_boost, rubber)
Main(_fenics)
app.exec_()
