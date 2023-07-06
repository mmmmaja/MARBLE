from __future__ import absolute_import
from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Integral, Equation, Equations, Problem, Material)
from sfepy.discrete.fem import FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect, ScipyIterative
from sfepy.solvers.nls import Newton
from AI.mesh_converter import *


"""
In this file the solver for the Sfepy library is defined.
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
            'eps_a': 1e-10,  # absolute tolerance
        })
        return Newton({}, lin_solver=ls, status=nls_status)
    else:
        return Newton({}, lin_solver=ScipyDirect({}), status=nls_status)


class FENICS:

    def __init__(self, mesh, rank_material):

        # Boundary conditions will be specified later
        self.boundary_conditions = None
        # Object with all the mesh properties
        self.mesh_boost = mesh
        # Material with physical properties
        self.rank_material = rank_material

        self.DOMAIN, self.omega, self.material, self.integral = None, None, None, None

        # The top and bottom regions of the mesh
        self.top, self.bottom = None, None

        self.innit()

    def innit(self):
        """
        Set up the problem's variables, equations, materials and solvers
        """

        # The domain of the mesh (allows defining regions or subdomains)
        self.DOMAIN = FEDomain(name='domain', mesh=self.mesh_boost.sfepy_mesh)

        # Omega is the entire domain of the mesh
        self.omega = self.DOMAIN.create_region(name='Omega', select='all')

        # Create the material for the mesh
        self.material = Material(name='m', values=self.rank_material.get_properties())

        # Create an Integral over the domain
        # Integrals specify which numerical scheme to use.
        self.integral = Integral('i', order=1)

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
        variables = PROBLEM.solve()

        dim = 3
        # Get the displacement field of the mesh in three dimensions (x, y, z)
        u = variables()
        # Reshape so that each displacement vector is in a row [x, y, z] displacements
        u = u.reshape((int(u.shape[0] / dim), dim))

        print('maximum displacement x:', np.abs(u[:, 0]).max())
        print('maximum displacement y:', np.abs(u[:, 1]).max())
        print('maximum displacement z:', np.abs(u[:, 2]).max())

        return u
