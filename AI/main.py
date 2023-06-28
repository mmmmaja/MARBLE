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


def apply_force(fenics, gui, cell=None, F=FORCE, relaxation=True):
    """
    Function that applies a vertex specific force or volume (stable) force across the whole mesh
    """

    global stress_relaxation_ref

    if cell is not None:
        vertex_ids = fenics.mesh_boost.get_vertex_ids_from_coords(cell.points)
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
            callback=lambda cell_id: apply_force(self.fenics, self.gui, cell_id),
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
        self.gui.plotter.show()


app = QApplication(sys.argv)
_mesh_boost = GridMesh(30, 30, z_function=concave)
_fenics = FENICS(_mesh_boost, rubber)
Main(_fenics)
app.exec_()
