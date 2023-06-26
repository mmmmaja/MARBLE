import time
import pyvistaqt as pvqt  # For updating plots real time
from PyQt5.QtWidgets import QApplication
from AI.fenics import *
from AI.mesh_converter import *
from AI.material_handler import *
from PyQt5.QtCore import QObject, QTimer

FORCE = 30.5


def add_mesh(plotter, vtk_mesh, rank_material):
    """
    :param plotter: pyvistaqt plotter object
    :param vtk_mesh: mesh in .vtk format (instance of mesh_boost)
    :param rank_material: Holds all the physical properties of the material of the mesh

    Took it out of the Main to create a separate timer class
    Adds the mesh in the .vtk format to the plotter
    """
    _visual_properties = rank_material.visual_properties
    plotter.add_mesh(
        vtk_mesh,
        show_edges=True,
        smooth_shading=True,
        show_scalar_bar=False,
        edge_color=_visual_properties['edge_color'],
        color=_visual_properties['color'],
        specular=_visual_properties['specular'],
        metallic=_visual_properties['metallic'],
        roughness=_visual_properties['roughness'],
        name='initial_mesh'
    )
    plotter.enable_lightkit()


def update(u, plotter, mesh_boost, rank_material):
    # update the mesh
    mesh_boost.update_vtk(u)
    plotter.clear()
    add_mesh(plotter, mesh_boost.current_vtk, rank_material)
    plotter.update()


class StressRelaxation:
    """
    Standard Linear Solid (SLS) model for stress relaxation
    Viscosity -  the material's resistance to flow or change shape

    τ = η / E
        τ is the relaxation time
        η is the viscoelastic damping coefficient (or viscosity in the SLS model)
        E is the elastic modulus (Young's modulus)

    TODO read more here:
    https://en.wikipedia.org/wiki/Viscoelasticity

    If doesn't work try:
    Generalized Maxwell model
    """

    def __init__(self, plotter, fenics, u0, F0, vertex_ids):

        # Time step: how many milliseconds between each update
        self.dt = 10  # ms
        # Current time of the stress relaxation simulation
        self.t = 0
        self.PRESS_TIME = 2 * 1000  # n seconds in ms

        self.plotter = plotter
        # The solver
        self.fenics = fenics
        # The maximum displacement of the body
        self.u0 = u0
        # The initial force applied to the body
        self.F0 = F0
        # Where the mesh was pressed
        self.vertex_ids = vertex_ids

        self.relaxation_timer = None
        self.wait_timer = None

    def initiate(self):
        # Timer for relaxation process, but don't start yet
        self.relaxation_timer = QTimer()
        self.relaxation_timer.timeout.connect(self.timer_loop)

        print("Start waiting process")
        # Timer for initial 5 seconds wait
        self.wait_timer = QTimer()
        self.wait_timer.setSingleShot(True)  # Ensure the timer only triggers once
        self.wait_timer.timeout.connect(self.start_relaxation)
        self.wait_timer.start(self.PRESS_TIME)  # Wait for the given interval (simulate the press)

    def start_relaxation(self):
        print("Start relaxation process")
        # This function will be called after the wait timer is finished
        # Start the relaxation process here
        self.relaxation_timer.start(self.dt)  # period of dt milliseconds

    def timer_loop(self):
        """
        Stress relaxation behavior is described by the equation:
        u(t) = u0 * (1 - exp(-t/τ))
            u0 is the maximum displacement,
            t is the current time of the simulation,
            τ is the relaxation time of the material.
        :return:
        """

        # calculate the current force
        F = self.F0 * np.exp(-self.t / self.fenics.rank_material.time_constant)
        print("FORCE: ", F)

        # calculate the displacement
        u = self.fenics.apply_force(self.vertex_ids, F)

        # update
        update(u, self.plotter, self.fenics.mesh_boost, self.fenics.rank_material)

        # advance the time variable
        self.t += self.dt

        # Disable the timer when the u is close to 0
        if F < 1e-2:
            print("Stop relaxation process")
            self.relaxation_timer.stop()
            self.relaxation_timer.deleteLater()


class Main:

    def __init__(self, fenics):
        self.fenics = fenics
        self.mesh_boost = fenics.mesh_boost
        self.rank_material = fenics.rank_material

        self.plotter = None

        self.create_plot()

        """
        Timer for updating the state during fenics calculations (Starts a new thread)
        Documentation:
        https://doc.qt.io/qtforpython-5/PySide2/QtCore/QTimer.html
        """
        self.timer = None

    def create_plot(self):
        # Creates a plotter object and sets all the initial settings
        self.plotter = pvqt.BackgroundPlotter()
        add_mesh(self.plotter, self.mesh_boost.current_vtk, self.rank_material)

        # Add the material name to the plotter
        text = self.rank_material.name
        text += '\nE: ' + str(self.rank_material.young_modulus)
        text += '\nv: ' + str(self.rank_material.poisson_ratio)

        self.plotter.add_text(text, position='lower_left', font_size=8, color='white', shadow=True)

        # Enable cell picking
        self.plotter.enable_cell_picking(
            callback=self.apply_force,
            font_size=10,
            color='white',
            point_size=30,
            style='wireframe',
            line_width=5,
            through=False
        )

        # Add the event on the press of the space bar
        self.plotter.add_key_event('space', self.apply_force)
        self.plotter.show()

    def apply_force(self, cell=None, F=FORCE):
        """
        Function that applies a vertex specific force or volume (stable) force across the whole mesh
        """
        if cell is not None:
            vertex_ids = self.mesh_boost.get_vertex_ids_from_coords(cell.points)
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
        u = self.fenics.apply_force(vertex_ids, F)
        # Update plot and meshes
        update(u, self.plotter, self.mesh_boost, self.rank_material)

        # Start the stress relaxation process
        stress_relaxation = StressRelaxation(self.plotter, self.fenics, u0=u, F0=F, vertex_ids=vertex_ids)
        stress_relaxation.initiate()


app = QApplication(sys.argv)
_mesh_boost = GridMesh(30, 30, z_function=concave)
_fenics = FENICS(_mesh_boost, rubber)
Main(_fenics)
app.exec_()
