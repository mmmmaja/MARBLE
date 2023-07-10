from __future__ import absolute_import

import numpy as np
from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Equation, Equations, Problem, Material, Integral, Field)
from sfepy.discrete.fem import Mesh, FEDomain
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect, ScipyIterative
from sfepy.solvers.nls import Newton
# from AI.model.mesh_converter import *


"""
Good resource for FEM:  
https://quantpaleo.earth.indiana.edu/Lectures/Finite%20Element%20Analysis.pdf
https://www.simscale.com/blog/stress-and-strain/

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
                # Get the pressure at the current quadrature point
                values[i] = force_handler.get_pressure(coors[i])

            return {'val': values}

    return force_fun


def get_solver(iterative=True):
    """
    :param iterative: True if iterative solver is used, False if direct solver is used
    :return: Non-linear solver that solves non-linear system of equations
    """

    # Status of the non-linear solver
    nls_status = IndexedStruct()

    if iterative:
        # Conjugate Gradient Squared method. Usually used for large systems of equations
        ls = ScipyIterative({
            'method': 'cgs',  # Conjugate Gradient Squared method
            'i_max': 1000,  # maximum number of iterations
            'eps_a': 1e-6,  # absolute tolerance
        })
        return Newton({}, lin_solver=ls, status=nls_status)

    else:
        # Can be slower than iterative solver for large systems of equations
        # but will always converge and find a solution
        return Newton({}, lin_solver=ScipyDirect({}), status=nls_status)


class FENICS:

    def __init__(self, fenics_mesh, rank_material, sensors):

        # Object with all the mesh properties for the FEM solver
        self.fenics_mesh = fenics_mesh

        # Material with physical properties
        self.rank_material = rank_material

        # Sensors to measure the output, will be updated after computation is done
        self.sensors = sensors

        # Variables that will be used in the solver
        self.top, self.bottom, self.integral = None, None, None

    def apply_pressure(self, force_handler):
        """
        Inspired by:
        https://github.com/sfepy/sfepy/blob/master/doc/tutorial.rst
        https://github.com/sfepy/sfepy/issues/740

        Compute the displacement of the mesh due to the applied pressure and update the sensors

        :param force_handler: a ForceHandler object that specifies a force at each point of the mesh
        :return: displacement u of the mesh for each vertex in x, y, z direction
        """

        field = self.fenics_mesh.field
        DOMAIN = self.fenics_mesh.DOMAIN
        omega = self.fenics_mesh.omega

        # Create an Integral over the domain
        # Integrals specify which numerical scheme to use.
        self.integral = Integral('i', order=1)

        # 1) Define the REGIONS of the mesh
        self.top, self.bottom = self.fenics_mesh.get_regions(DOMAIN)

        # Create the material for the mesh
        material = Material(name='m', values=self.rank_material.get_properties())

        # 3) Define the field variables
        # 'u' is the displacement field of the mesh (3D)
        u = FieldVariable(name='u', kind='unknown', field=field)
        # v is the test variable associated with u
        v = FieldVariable(name='v', kind='test', field=field, primary_var_name='u')

        # 4) Define the terms of the equation

        # Define the elasticity term of the material with specified material properties
        elasticity_term = Term.new(
            name='dw_lin_elastic_iso(m.lam, m.mu, v, u)',
            integral=self.integral, region=omega, m=material, v=v, u=u
        )
        # Get the specific force terms
        force_term = self.get_force_term(force_handler, v)

        # 5) Create equations
        equations = [Equation('balance', elasticity_term + force_term)]
        # Initialize the equations object
        equations = Equations(equations)

        # 6) Define the problem
        PROBLEM = Problem(name='elasticity', equations=equations, domain=DOMAIN)

        # Add the boundary conditions to the problem and add the solver
        boundary_conditions = EssentialBC('fix_bottom', self.bottom, {'u.all': 0.0})
        PROBLEM.set_bcs(ebcs=Conditions([boundary_conditions]))
        PROBLEM.set_solver(get_solver())

        # 7) Solve the problem
        variables = PROBLEM.solve(
            post_process_hook_final=self.calculate_stress
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

    def calculate_stress(self, pb, state):
        """
        The sensors measure internal forces, so the internal forces are derived from the displacements.
        Stresses are calculated as post-processing quantities once a converged solution is obtained.

        :param pb: The Problem instance which was solved.
        :param state: The state variable (displacement) obtained by solving the problem
        """
        ev = pb.evaluate

        stress = ev(
            'ev_cauchy_stress.3.Omega(m.D, u)',
            mode='el_avg',
            copy_materials=False
        )
        """
        Stress is a measure of the internal forces in a material body. 
        (FORCE PER UNIT AREA)
        In a three-dimensional space, the stress tensor is represented as a 3x3 matrix, 
        where each element of the matrix represents a specific directional component of the stress.
        σ = 
        [σ_xx, σ_xy, σ_xz]
        [σ_yx, σ_yy, σ_yz]
        [σ_zx, σ_zy, σ_zz]
        I am just interested in diagonal elements of the stress tensor, which are the normal stresses.
        Here, just taking the z component of the stress tensor.
        """
        # Calculate the stress tensor and flatten it
        stress_tensor_np = np.squeeze(np.array(stress.data))

        for sensor in self.sensors.sensor_list:
            sensor.set_readings(stress_tensor_np)


"""
This class is used to create a mesh from a vtk file and to create the sfepy instances of the mesh.
"""


def transform_mesh(vtk_mesh):
    """
    This is the stupid way to create a mesh from a vtk file but there is no other way to do it.
    (Sfepy mesh cannot be updated once it is created)

    :param vtk_mesh: The vtk mesh instance
    :return: The sfepy mesh instance
    """
    path = '../meshes/current_mesh.vtk'
    # Save the mesh to a file
    vtk_mesh.save(path)
    # Load the mesh as a sfepy mesh
    return Mesh.from_file(path)

class SfepyMesh:

    def __init__(self, mesh_boost):

        # Create the mesh from the vtk instance of the mesh
        self.fenics_mesh = transform_mesh(mesh_boost.current_vtk)

        # These define regions where displacement is fixed (bottom) and where pressure is applied (top)
        self.top_region_ids = mesh_boost.top_region_ids
        self.bottom_region_ids = mesh_boost.bottom_region_ids

        # This need to be defined for the sfepy mesh and are created in the create() method
        self.DOMAIN, self.omega, self.field = None, None, None

    def create(self):
        """
        Create the sfepy instances of the mesh
        :return: True if the mesh is valid, False otherwise
        """
        try:
            # Create the mesh instances that will be used to solve the problem

            # The domain of the mesh (allows defining regions or subdomains)
            self.DOMAIN = FEDomain(name='domain', mesh=self.fenics_mesh)
            # Omega is the entire domain of the mesh
            self.omega = self.DOMAIN.create_region(name='Omega', select='all')
            # Create the finite element field
            self.field = Field.from_args(
                name='field', dtype=np.float64, shape='vector', region=self.omega, approx_order=1
            )
        except RuntimeError:
            # Mesh is not valid
            return False
        return True

    def validate(self, threshold=0.035):
        """
        Validate the mesh by checking the determinant of the Jacobian matrix at each point of the mesh.
        :param threshold: The threshold for the determinant of the Jacobian matrix
        :return: True if the mesh is valid, False otherwise
        """
        # Get the mapping of the mesh
        try:
            geo, _ = self.field.get_mapping(
                self.field.region, Integral('i', order=2), 'volume'
            )
        except ValueError:
            # Mesh is not valid
            return False

        """
        The Jacobian matrix collects all first-order partial derivatives of a multivariate function.
        It is used to map from a reference element (with standard coordinates) to the physical one in the actual mesh. 
        It is necessary for integration over the physical domain.
        
        Determinants of the Jacobian matrices give a measure of how much the mapping distorts volumes. 
        If the determinant of the Jacobian is negative at a given point, 
        that means the mapping at that point is "inverting" the reference element. 
        This is usually a sign that the mesh is badly distorted.
        https://www.lusas.com/user_area/theory/jacobian.html
        """

        # Negative determinant might indicate that the mesh element is inverted or highly distorted.
        det_jacobians = geo.det.flatten()
        # Get the minimum determinant of the Jacobian matrix
        min_det = det_jacobians.min()

        # Check if the mesh is inverted
        if min_det < threshold:
            print('INVALID MESH', min_det)
            return False
        else:
            print('valid mesh', min_det)
            return True

    def get_neighbouring_cells(self, vertex_id):
        """
        This method is used to interpolate the stress obtained in a solver.
        Since the stress is calculated at the center of each cell, the stress at a vertex is calculated by
        interpolating the stress of the cells that share the vertex.

        :param vertex_id: The id of the vertex
        :return: The ids of the cells that share the vertex
        """
        conn = self.fenics_mesh.get_conn(desc='3_8')
        # get all cells that share the vertex
        cells = [i for i, conn in enumerate(conn) if vertex_id in conn]
        return cells

    def get_regions(self, domain):
        """
        https://sfepy.org/doc-devel/users_guide.html
        From documentation:
        Regions serve to select a certain part of the computational domain using topological entities of the FE mesh.
        They are used to define the boundary conditions, the domains of terms and materials etc.

        Those regions will be used to define the boundary conditions and the force term.
        :return: top and bottom regions
        """

        expr_base = 'vertex ' + ', '.join([str(i) for i in self.top_region_ids])
        top = domain.create_region(name='Top', select=expr_base, kind='facet')

        # Create a bottom region (Where the boundary conditions apply so that the positions are fixed)
        # Define the cells by their Ids and use vertex <id>[, <id>, ...]
        expr_extruded = 'vertex ' + ', '.join([str(i) for i in self.bottom_region_ids])
        bottom = domain.create_region(name='Bottom', select=expr_extruded, kind='facet')

        return top, bottom
