from __future__ import absolute_import
from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Integral, Equation, Equations, Problem, Material)
from sfepy.discrete.fem import FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect, ScipyIterative
from sfepy.solvers.nls import Newton
from AI.mesh_converter import *


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

        # Meshio format is valid for SFepy library
        self.boundary_conditions = None
        self.mesh_boost = mesh
        # Material with physical properties
        self.rank_material = rank_material

        self.DOMAIN, self.omega, self.material = None, None, None
        self.integral = None

        self.innit()

    def innit(self):
        """
        Set up the problem's variables, equations, materials and solvers
        """
        # The domain of the mesh (allows defining regions or subdomains)
        self.DOMAIN = FEDomain(name='domain', mesh=self.mesh_boost.sfepy_mesh)

        # Omega is the entire domain of the mesh
        self.omega = self.DOMAIN.create_region(name='Omega', select='all')

        # Define the materials
        # Create the material for the mesh
        self.material = Material(name='m', values=self.rank_material.get_properties())

        # Create an Integral over the domain
        # Integrals specify which numerical scheme to use.
        self.integral = Integral('i', order=1)

    def get_force_terms(self, force, v):
        """
        :param force: numpy array of the form [ [vertex_ids, force] ] or a single force value
        :param v: The test variable
        :return: The force terms associated with the given regions

        Check out the documentation for the Term class here:
        https://sfepy.org/doc-devel/terms_overview.html
        """

        # Apply the force to the entire top face of the mesh
        # Check if the force is a single value or a dictionary
        if isinstance(force, int) or isinstance(force, float):
            print('Apply volume force')
            top, bottom = self.mesh_boost.get_regions(self.DOMAIN)
            region = top
            # Create a material that will be the force applied to the body
            f = Material(name='f', val=force)
            force_terms = [Term.new(
                    'dw_surface_ltr(f.val, v)', integral=self.integral, region=region, f=f, v=v
            )]
        else:
            print('Apply vertex specific force')
            # In this case, the force is a dictionary of the form [ [vertex_ids, force] ]
            # Create the force terms for each vertex
            force_terms = []
            for i in range(len(force)):
                vertex_ids, force_value = force[i][0], force[i][1]

                # Create a region with the vertices of the cell
                expr = 'vertex ' + ', '.join([str(ID) for ID in vertex_ids])
                try:
                    # Create the region with the given expression
                    region = self.DOMAIN.create_region(name='region', select=expr, kind='facet')
                    # Create a material that will be the force applied to the body
                    f = Material(name='f', val=force_value)
                    # Create the force term
                    # FIXME its gonna be a problem if there are more than one force term
                    force_terms.append(Term.new(
                            'dw_surface_ltr(f.val, v)', integral=self.integral, region=region, f=f, v=v
                    ))
                except ValueError:
                    # The region is empty, so it is not created
                    continue

        return force_terms

    def apply_force(self, force):
        """
        Elasticity problem solved with FEniCS.

        Inspired by:
        https://github.com/sfepy/sfepy/blob/master/doc/tutorial.rst
        https://github.com/sfepy/sfepy/issues/740

        :param force: numpy array of the form [ [vertex_ids, force] ] or a single force value
        :return: displacement u of the mesh for each vertex in x, y, z direction
        """

        # 1) Define the REGIONS of the mesh
        top, bottom = self.mesh_boost.get_regions(self.DOMAIN)

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
        force_terms = self.get_force_terms(force, v)

        # If there are no force terms, return zero displacement array
        if len(force_terms) == 0:
            return np.zeros(self.mesh_boost.sfepy_mesh.coors.shape)

        print('number of force terms: ', len(force_terms))

        # Create equations
        equations = [Equation('balance_' + str(i), elasticity_term + force_term) for i, force_term in
                     enumerate(force_terms)]
        # Initialize the equations object
        equations = Equations(equations)

        # 9) Define the problem
        PROBLEM = Problem(name='elasticity', equations=equations, domain=self.DOMAIN)

        # Add the boundary conditions to the problem and add the solver
        boundary_conditions = EssentialBC('fix_bottom', bottom, {'u.all': 0.0})
        PROBLEM.set_bcs(ebcs=Conditions([boundary_conditions]))
        PROBLEM.set_solver(get_solver())

        # 10) Solve the problem
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
