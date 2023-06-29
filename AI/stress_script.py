from PyQt5.QtWidgets import QApplication
from AI.mesh_converter import *
from PyQt5.QtCore import QTimer

FORCE_LIMIT = 1e-2


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

    TODO try the simple visualization of the stress relaxation process not involving fenics class
    """

    def __init__(self, gui, fenics, u0, F0, vertex_ids):
        # Time step: how many milliseconds between each update
        # When having 40 ms program is not freezing, all smaller values freeze the program
        self.dt = 20  # ms
        # Current time of the stress relaxation simulation
        self.t = 0
        self.PRESS_TIME = 4 * 1000  # n seconds in ms

        self.gui = gui
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
        # Timer for initial n seconds wait
        self.wait_timer = QTimer()
        self.wait_timer.setSingleShot(True)  # Ensure the timer only triggers once
        self.wait_timer.timeout.connect(self.start_relaxation)
        self.wait_timer.start(self.PRESS_TIME)  # Wait for the given interval (simulate the press)

    def start_relaxation(self):
        print("Start relaxation process")
        # This function will be called after the wait timer is finished
        # Start the relaxation process here
        self.relaxation_timer.start(self.dt)  # period of dt milliseconds

    def get_displacement(self):
        """
        Stress relaxation behavior is described by the equation:
        u(t) = u0 * exp(-t/τ)
            u0 is the maximum displacement,
            t is the current time of the simulation,
            τ is the relaxation time of the material.
        :return: the current displacement dependant of the time t of the simulation
        """
        return self.u0 * np.exp(-self.t / self.fenics.rank_material.time_constant)

    def timer_loop(self):
        """
        This function is called every dt milliseconds
        Updates the displacement of the mesh and the GUI
        """

        # calculate the displacement
        u = self.get_displacement()

        # OVERRIDE the GUI
        self.fenics.mesh_boost.override_mesh(u)
        self.gui.draw_mesh(self.fenics.mesh_boost.current_vtk)
        self.gui.plotter.update()

        QApplication.processEvents()

        # advance the time variable
        self.t += self.dt

        # Disable the timer when the magnitude of u is close to 0
        if np.linalg.norm(u) < FORCE_LIMIT:
            print("Stop relaxation process")
            self.relaxation_timer.stop()

    def stop(self):
        """
        Stop the relaxation process and delete the timers
        Otherwise the timers will keep running in the background and display will keep freezing
        """
        if self.relaxation_timer is not None:
            if self.relaxation_timer.isActive():
                print("Stop relaxation process")
                self.relaxation_timer.stop()
                self.relaxation_timer.deleteLater()
                self.relaxation_timer = None

        if self.wait_timer is not None:
            if self.wait_timer.isActive():
                self.wait_timer.stop()
                self.wait_timer.deleteLater()
                self.wait_timer = None
