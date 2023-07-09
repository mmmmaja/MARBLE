from __future__ import absolute_import
from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Integral, Equation, Equations, Problem, Material)
from sfepy.discrete.fem import FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect, ScipyIterative
from sfepy.solvers.nls import Newton
from AI.model.mesh_converter import *


"""
Good resource for FEM:  
https://quantpaleo.earth.indiana.edu/Lectures/Finite%20Element%20Analysis.pdf

In this file the solver for the Sfepy library is defined.

Displacements and reaction forces are the fundamental quantities that are being solved in any FEM computation. 
Both stresses and strains are calculated as post-processing quantities once a converged solution is obtained 
for the nodal displacements.

https://www.simscale.com/blog/stress-and-strain/
"""


def create_force_function(force_handler):
    """
    :param force_handler: a ForceHandler object that specifies a force at each point of the mesh
    :return: force function to create the material with the force term
    """

    def force_fun(ts, coors, mode=None, **kwargs):
        """
            Define a function that represents the spatial distribution of the force

            :param ts:
                a TimeStepper object that holds the current time step information.
            :param coors:
                is a NumPy array that contains the coordinates of the points where the function should be evaluated.
                In SfePy, when a function is evaluated, it's not evaluated at every point in the domain.
                Instead, it's evaluated at a specific set of points, called quadrature points,
                that are used for numerical integration.
                The coors array contains the coordinates of these quadrature points.
            :param mode:
                is a string that tells the function what it should return.
                When mode is 'qp', it means that the function is being asked to return its values at the quadrature points.
            :param kwargs:
            :return: force function to create the material with the force term
            """
        if mode == 'qp':  # querying values at quadrature points
            values = np.zeros((coors.shape[0], 1, 1), dtype=np.float64)

            for i in range(coors.shape[0]):
                # If the coordinate is in the force dictionary, get the corresponding force value
                values[i] = force_handler.get_pressure(coors[i])

            return {'val': values}

    return force_fun


def get_solver(iterative=True):
    nls_status = IndexedStruct()
    if iterative:
        ls = ScipyIterative({
            'method': 'cgs',  # Conjugate Gradient Squared method
            'i_max': 1000,  # maximum number of iterations
            'eps_a': 1e-8,  # absolute tolerance
        })
        return Newton({}, lin_solver=ls, status=nls_status)
    else:
        return Newton({}, lin_solver=ScipyDirect({}), status=nls_status)


class FENICS:

    def __init__(self, mesh, rank_material, sensors):

        # Boundary conditions will be specified later
        self.boundary_conditions = None
        # Object with all the mesh properties
        self.mesh_boost = mesh
        # Material with physical properties
        self.rank_material = rank_material
        # Sensors to measure the output
        self.sensors = sensors

        self.DOMAIN, self.omega, self.material, self.integral = None, None, None, None

        # The top and bottom regions of the mesh
        self.top, self.bottom = None, None

    def get_force_term(self, force_handler, v):
        """
        :param force_handler: a ForceHandler object that specifies a force at each point of the mesh
        :param v: The test variable
        :return: The force term associated with the given regions of the mesh

        The documentation for the Term class:
        https://sfepy.org/doc-devel/terms_overview.html
        """

        # Get the force function acting on the mesh
        force_fun = create_force_function(force_handler)
        # Register the function
        f = Material(name='f', function=force_fun)
        # Define the force term for the equation
        force_term = Term.new(
            'dw_surface_ltr(f.val, v)',
            integral=self.integral, region=self.top, v=v, f=f
        )
        return force_term

    def apply_pressure(self, force_handler):
        """
        Inspired by:
        https://github.com/sfepy/sfepy/blob/master/doc/tutorial.rst
        https://github.com/sfepy/sfepy/issues/740

        :param force_handler: a ForceHandler object that specifies a force at each point of the mesh
        :return: displacement u of the mesh for each vertex in x, y, z direction
        """

        """
        Set up the problem's variables, equations, materials and solvers
        """
        try:
            # The domain of the mesh (allows defining regions or subdomains)
            self.DOMAIN = FEDomain(name='domain', mesh=self.mesh_boost.sfepy_mesh)
        except RuntimeError:
            # Bad orientation of the mesh error
            # Return 0 displacement if the mesh is not valid
            return np.zeros((self.mesh_boost.sfepy_mesh.coors.shape[0], 3))

        # Omega is the entire domain of the mesh
        self.omega = self.DOMAIN.create_region(name='Omega', select='all')

        # Create the material for the mesh
        self.material = Material(name='m', values=self.rank_material.get_properties())

        # Create an Integral over the domain
        # Integrals specify which numerical scheme to use.
        self.integral = Integral('i', order=1)

        # 1) Define the REGIONS of the mesh
        self.top, self.bottom = self.mesh_boost.get_regions(self.DOMAIN)

        # 2) Define the field of the mesh (finite element approximation)
        # approx_order indicates the order of the approximation (1 = linear, 2 = quadratic, ...)
        field = Field.from_args(
            name='field', dtype=np.float64, shape='vector', region=self.omega, approx_order=1
        )

        # 3) Define the field variables
        # 'u' is the displacement field of the mesh (3D)
        u = FieldVariable(name='u', kind='unknown', field=field)
        # v is the test variable associated with u
        v = FieldVariable(name='v', kind='test', field=field, primary_var_name='u')

        # 4) Define the terms of the equation

        # Define the elasticity term of the material with specified material properties
        elasticity_term = Term.new(
            name='dw_lin_elastic_iso(m.lam, m.mu, v, u)',
            integral=self.integral, region=self.omega, m=self.material, v=v, u=u
        )
        # Get the specific force terms
        force_term = self.get_force_term(force_handler, v)

        # 5) Create equations
        equations = [Equation('balance', elasticity_term + force_term)]
        # Initialize the equations object
        equations = Equations(equations)

        # 6) Define the problem
        PROBLEM = Problem(name='elasticity', equations=equations, domain=self.DOMAIN)

        # Add the boundary conditions to the problem and add the solver
        boundary_conditions = EssentialBC('fix_bottom', self.bottom, {'u.all': 0.0})
        PROBLEM.set_bcs(ebcs=Conditions([boundary_conditions]))
        PROBLEM.set_solver(get_solver())

        # 7) Solve the problem
        variables = PROBLEM.solve(
            post_process_hook_final=self.stress_strain
        )

        dim = 3
        # Get the displacement field of the mesh in three dimensions (x, y, z)
        u = variables()
        # Reshape so that each displacement vector is in a row [x, y, z] displacements
        u = u.reshape((int(u.shape[0] / dim), dim))

        print('maximum displacement x:', np.abs(u[:, 0]).max())
        print('maximum displacement y:', np.abs(u[:, 1]).max())
        print('maximum displacement z:', np.abs(u[:, 2]).max())

        return u

    def stress_strain(self, pb, state):
        """
        This function is called after the problem is solved.
        Calculate and output strain and stress for given displacements.

        :param pb: The Problem instance which was solved.
        :param state: The state variable (displacement) obtained by solving the problem
        :return:
        """
        ev = pb.evaluate
        # strain = ev(
        #     'ev_cauchy_strain.3.Omega(u)',
        #     mode='el_avg'
        # )

        stress = ev(
            'ev_cauchy_stress.3.Omega(m.D, u)',
            mode='el_avg',
            copy_materials=False
        )
        stress_tensor_np = np.array(stress.data)
        stress_tensor_np = np.squeeze(stress_tensor_np)

        for sensor in self.sensors.sensor_list:
            sensor.set_readings(stress_tensor_np)

        """
        In a three-dimensional space, the stress tensor is represented as a 3x3 matrix, 
        where each element of the matrix represents a specific directional component of the stress.
        σ = 
        [σ_xx, σ_xy, σ_xz]
        [σ_yx, σ_yy, σ_yz]
        [σ_zx, σ_zy, σ_zz]
        """


        """
        Stress and strain values are tensor fields, 
        But the sensor is not able to capture their detail, it only measures the total force applied to it. 
        This is why take an average of the stress tensor field and then multiply it by the area to get a force reading, 
        mimicking the sensor's output.
        """


"""
Regarding the sensors outputs:

The stresses and forces inside the material are derived from the displacements in Fenics. 
The FEA solves for the displacements that balance the external forces with the internal forces. 

The sensors measure internal forces, so the internal forces are derived from the displacements.
(!)
"""

"""
Strain is a measure of deformation representing the change in size and shape of a material body. 
(CHANGE IN LENGTH PER UNIT LENGTH)
"""

"""
Stress is a measure of the internal forces in a material body. 
(FORCE PER UNIT AREA)
So essentially, stress is what sensors would measure.
"""