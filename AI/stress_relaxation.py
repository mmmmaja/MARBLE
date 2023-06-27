import numpy as np
from PyQt5.QtCore import QTimer

# Time step: how many milliseconds between each update
DT = 5  # ms


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

    If it doesn't work try:
    Generalized Maxwell model
    """

    def __init__(self, gui, fenics, u0, F0, vertex_ids):
        # Current time of the stress relaxation simulation
        self.t = 0
        self.PRESS_TIME = 1 * 1000  # n seconds in ms

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
        print("Initiating stress relaxation process")
        # Timer for relaxation process, but don't start yet
        self.relaxation_timer = QTimer(DT)
        self.relaxation_timer.timeout.connect(self.timer_loop)

        # Start the waiting process
        # Timer for initial n seconds wait
        self.wait_timer = QTimer()
        self.wait_timer.setSingleShot(True)  # Ensure the timer only triggers once
        self.wait_timer.timeout.connect(self.start_relaxation)
        print("Waiting for ", self.PRESS_TIME, " ms")
        self.wait_timer.start(self.PRESS_TIME)  # Wait for the given interval (simulate the press)

    def start_relaxation(self):
        print("Starting stress relaxation process")
        # This function will be called after the wait timer is finished
        # Start the relaxation process here
        self.relaxation_timer.start()  # period of dt milliseconds

    def calculate_force(self):
        return self.F0 * np.exp(-self.t / self.fenics.rank_material.time_constant)

    def timer_loop(self):
        """
        Stress relaxation behavior is described by the equation:
        u(t) = u0 * (1 - exp(-t/τ))
            u0 is the maximum displacement,
            t is the current time of the simulation,
            τ is the relaxation time of the material.
        :return:
        """

        # calculate the force
        F = self.calculate_force()
        print("F: ", F)

        # calculate the displacement
        print("waiting for displacement")
        u = self.fenics.apply_force(self.vertex_ids, F)

        # update the mesh and the plotter
        self.fenics.mesh_boost.update_vtk(u)
        print("Updating mesh")
        self.gui.draw_mesh()
        print("Mesh updated")

        # advance the time variable
        self.t += DT
        print("t: ", self.t)

        # Disable the timer when the u is close to 0
        if F < 1e-2:
            print("Stop relaxation process")
            self.relaxation_timer.stop()
            self.relaxation_timer.deleteLater()
        else:
            print("Continue relaxation process")

