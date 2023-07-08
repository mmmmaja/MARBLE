from AI.model.mesh_converter import *
from PyQt5.QtCore import QTimer


class StressRelaxation:
    """
    Standard Linear Solid (SLS) model for stress relaxation
    Viscosity -  the material's resistance to flow or change shape

    Exponential decay function for relaxation
        R(t) = e^(-t/τ)
    """

    # The force limit to stop the relaxation process
    FORCE_LIMIT = 1e-2

    def __init__(self, gui, mesh_boost, rank_material):
        # Time step: how many milliseconds between each update
        self.dt = 20  # ms
        # Current time of the stress relaxation simulation
        self.t = 0

        self.gui = gui
        self.mesh_boost = mesh_boost
        self.rank_material = rank_material

        self.relaxation_timer = None
        self.wait_timer = None

    def initiate(self, wait=False):

        # Timer for relaxation process, but don't start yet
        self.relaxation_timer = QTimer()
        self.relaxation_timer.timeout.connect(self.thread_iteration)

        if wait:
            # The time in milliseconds to wait before starting the relaxation process
            wait_time = 1000
            # Timer for waiting before starting the relaxation process
            self.wait_timer = QTimer()
            self.wait_timer.timeout.connect(self.start_relaxation)
            self.wait_timer.setSingleShot(True)
            self.wait_timer.start(wait_time)
        else:
            self.start_relaxation()

    def start_relaxation(self):
        self.relaxation_timer.start(self.dt)  # period of dt milliseconds

    def exponential_decay(self, t):
        """
        Exponential decay function for relaxation
        R(t) = e^(-t/τ)
        :param t: current time of the simulation
        :return: the relaxation factor between 0 and 1,
            representing the proportion of stress remaining in the material
        """
        return np.exp(-t / self.rank_material.time_constant)

    def get_relaxation_displacement(self, u, t):
        """
        Stress relaxation behavior is described by the equation:
        u(t) = u0 * exp(-t/τ)
            u0 is the maximum displacement,
            t is the current time of the simulation,
            τ is the relaxation time of the material.
        :return: the current displacement dependant of the time t of the simulation
        """
        return u * np.exp(-t / self.rank_material.time_constant)

    def thread_iteration(self):
        self.relax_iteration(self.t)

    def active_iteration(self, t, stimuli):
        self.relax_iteration(t, stimuli.get_affected_mesh_indices(self.mesh_boost.current_vtk))

    def relax_iteration(self, t, unaffected_points=None):
        """
        :param t: current time of the simulation
        :param unaffected_points: the points that are not affected by the stress relaxation,
        because the stimuli is there

        This function is called every dt milliseconds
        Updates the displacement of the mesh and the GUI
        """

        # Get the displacement of the mesh on this iteration
        u = self.mesh_boost.current_vtk.points - self.mesh_boost.initial_vtk.points

        # calculate the displacement
        relaxation = self.get_relaxation_displacement(u, t)

        if unaffected_points:
            # Set the relaxation of the points that are not affected by the stress relaxation to 0
            relaxation[unaffected_points] = 0.0

        # OVERRIDE the GUI
        self.mesh_boost.override_mesh(relaxation)
        self.gui.sensors.update_visualization()

        self.gui.draw_mesh()
        self.gui.draw_sensors()
        self.gui.plotter.update()

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

