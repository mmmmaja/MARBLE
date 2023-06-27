from PyQt5.QtCore import QObject, QTimer

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
        print("Initiate stress relaxation")
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