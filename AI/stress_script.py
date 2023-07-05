from PyQt5.QtWidgets import QApplication
from AI.mesh_converter import *
from PyQt5.QtCore import QTimer


class StressRelaxation:
    """
    Standard Linear Solid (SLS) model for stress relaxation
    Viscosity -  the material's resistance to flow or change shape

    TODO read more here:
    https://en.wikipedia.org/wiki/Viscoelasticity

    If doesn't work try:
    Generalized Maxwell model
    """

    # The force limit to stop the relaxation process
    FORCE_LIMIT = 1e-2

    def __init__(self, gui, mesh_boost, rank_material, u0, F0):
        # Time step: how many milliseconds between each update
        self.dt = 20  # ms
        # Current time of the stress relaxation simulation
        self.t = 0
        # The time in milliseconds to wait before starting the relaxation process
        self.PRESS_TIME = 1 * 1000  # n seconds in ms

        self.gui = gui
        self.mesh_boost = mesh_boost
        self.rank_material = rank_material

        # The maximum displacement of the body
        self.u0 = u0
        # The initial pressure applied to the body
        self.F0 = F0

        self.relaxation_timer = None
        self.wait_timer = None

    def initiate(self):
        # Timer for relaxation process, but don't start yet
        self.relaxation_timer = QTimer()
        self.relaxation_timer.timeout.connect(self.timer_loop)

        # Timer for initial n seconds wait
        self.wait_timer = QTimer()
        self.wait_timer.setSingleShot(True)  # Ensure the timer only triggers once
        self.wait_timer.timeout.connect(self.start_relaxation)
        self.wait_timer.start(self.PRESS_TIME)  # Wait for the given interval (simulate the press)

    def start_relaxation(self):
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
        return self.u0 * np.exp(-self.t / self.rank_material.time_constant)

    def timer_loop(self):
        """
        This function is called every dt milliseconds
        Updates the displacement of the mesh and the GUI
        """

        # calculate the displacement
        u = self.get_displacement()

        # OVERRIDE the GUI
        self.mesh_boost.override_mesh(u)
        self.gui.sensors.update_visualization()

        self.gui.draw_mesh()
        self.gui.draw_sensors()
        self.gui.plotter.update()

        QApplication.processEvents()

        # advance the time variable
        self.t += self.dt

        # Disable the timer when the magnitude of u is close to 0
        if np.linalg.norm(u) < self.FORCE_LIMIT:
            self.relaxation_timer.stop()

    def stop(self):
        """
        Stop the relaxation process and delete the timers
        Otherwise the timers will keep running in the background and display will keep freezing
        """
        if self.relaxation_timer is not None:
            if self.relaxation_timer.isActive():
                self.relaxation_timer.stop()
                self.relaxation_timer.deleteLater()
                self.relaxation_timer = None

        if self.wait_timer is not None:
            if self.wait_timer.isActive():
                self.wait_timer.stop()
                self.wait_timer.deleteLater()
                self.wait_timer = None


class StressRelaxation_v2:

    def __init__(self, mesh_boost, rank_material):
        self.mesh_boost = mesh_boost
        self.rank_material = rank_material
        self.time = 0

    def get_displacements_matrix(self):
        # Check for last displacements with the stimuli -
        # don't apply relaxation there

        total_displacements = self.mesh_boost.initial_vtk.points - self.mesh_boost.current_vtk.points
        total_relaxation = np.zeros(total_displacements.shape)
        # Iterate through all the vertices
        for i in range(total_displacements.shape[0]):
            displacement = total_displacements[i]
            relaxation = self.get_relaxation_displacement(displacement, self.time)
            total_relaxation[i] = relaxation
        self.mesh_boost.current_vtk.points += total_relaxation

    def get_relaxation_displacement(self, current_displacement, time):
        return current_displacement * self.relaxation_function(time)

    def relaxation_function(self, time):
        """
        Exponential decay function for relaxation
        R(t) = e^(-t/τ)

        Multiply the current displacement of the vertex by the relaxation function to get the new displacement

        :param time: time in ms
        :return: values between 0 and 1, representing the proportion of stress remaining in the material
        """
        return math.exp(-time / self.rank_material.time_constant)
