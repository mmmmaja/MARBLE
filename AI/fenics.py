from __future__ import absolute_import
import sys
from sfepy.base.base import IndexedStruct
from sfepy.solvers.ts_solvers import TimeSteppingSolver
from AI.material_handler import *
from fea_.advanced_mesh_2 import *
import numpy as np
from sfepy.discrete import (FieldVariable, Integral, Function, Equation, Equations, Problem, Material, Region)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from AI.mesh_converter import *


class FENICS:

    def __init__(self, mesh, rank_material):
        # Get the trivial mesh of the sensor array

        # Meshio format is valid for SFepy library
        self.mesh_boost = mesh
        self.rank_material = rank_material

        # 1) Get the domain of the mesh (allows defining regions or subdomains)
        self.DOMAIN = FEDomain(name='domain', mesh=self.mesh_boost.meshio_mesh)

    def apply_vertex_force(self, point_ID):
        F = 10
        force = Material(name='f', val=F)

        # Define the cells by their Ids and use vertex <id>[, <id>, ...]
        expr = 'vertex ' + str(point_ID)

        # Create a region for the vertex
        force_region = self.DOMAIN.create_region(name='vertex_force', select=expr, kind='vertex')

        self.solve(force_region, force)

    def apply_volume_force(self, domain, integral, v):
        F = 10
        force = Material(name='f', val=F)
        top, bottom = self.mesh_boost.get_regions(domain)
        return Term.new(
            'dw_surface_ltr(f.val, v)', integral=integral, region=top, f=force, v=v
        )

    def apply_cell_force(self, cell):

        # cell_id = cell.GetCellId()  # Replace this with how you get the cell id
        cell_id = 10  # Replace this with how you get the cell id
        val = 10

        # Create a region for this cell
        expr = 'cell ' + str(cell_id)
        cell_region = self.DOMAIN.create_region(name='cell_force', select=expr, kind='cell')

        print(cell_region)

        # Create a material for the cell
        force = Material(name='f', val=val)

        self.solve(cell_region, force)


    def solve(self, region=None, f=None):
        """
        Elasticity problem solved with FEniCS.
        (the displacement of each node in the mesh)

        Inspired by:
        https://github.com/sfepy/sfepy/blob/master/doc/tutorial.rst
        https://github.com/sfepy/sfepy/issues/740

        :return: Displacement of the mesh for each node (shape: (num_sensors, 3)) in 3D
        """

        # 2) Define the REGIONS of the mesh
        # Omega is the entire domain of the mesh
        omega = self.DOMAIN.create_region(name='Omega', select='all')
        top, bottom = self.mesh_boost.get_regions(self.DOMAIN)

        # Works only for the 1st order
        ORDER = 1

        # 3) Define the field of the mesh (finite element approximation)
        # approx_order indicates the order of the approximation (1 = linear, 2 = quadratic, ...)
        field = Field.from_args(
            name='field', dtype=np.float64, shape='vector', region=omega, approx_order=ORDER
        )

        # 4) Define the field variables
        # 'u' is the displacement field of the mesh (3D)
        u = FieldVariable(name='u', kind='unknown', field=field)
        # v is the test variable associated with u
        v = FieldVariable(name='v', kind='test', field=field, primary_var_name='u')

        # 5) Define the materials
        # Create the material for the mesh
        material = Material(name='m', values=self.rank_material.get_properties())

        # 6) Create an Integral over the domain
        # Integrals specify which numerical scheme to use.
        integral = Integral('i', order=ORDER)

        # 7) Define the terms of the equation
        # They are combined to form the equation

        # Define the elasticity term of the material with specified material properties
        elasticity_term = Term.new(
            name='dw_lin_elastic_iso(m.lam, m.mu, v, u)', integral=integral, region=omega, m=material, v=v, u=u
        )

        if region is None or f is None:
            force_term = self.apply_volume_force(self.DOMAIN, integral, v)
        else:
            force_term = Term.new(
                'dw_surface_ltr(f.val, v)', integral=integral, region=region, f=f, v=v
            )
        # force_term = self.apply_volume_force(self.DOMAIN, integral, v)
        # force_term = self.apply_vertex_force(DOMAIN, integral, v)

        # 8) Define the equation with force term
        equations = Equations([Equation('balance', elasticity_term + force_term)])

        # 9) Get the solver for the problem
        # Nonlinear solver
        nls_status = IndexedStruct()
        newton_solver = Newton({}, lin_solver=ScipyDirect({}), status=nls_status)

        # 9) Define the problem
        PROBLEM = Problem(name='elasticity', equations=equations, domain=self.DOMAIN)

        # Add the boundary conditions to the problem
        boundary_conditions = EssentialBC('fix_bottom', bottom, {'u.all': 0.0})
        PROBLEM.set_bcs(ebcs=Conditions([boundary_conditions]))

        PROBLEM.set_solver(newton_solver)

        # 10) Solve the problem
        status = IndexedStruct()
        variables = PROBLEM.solve(status=status)

        dim = 3
        # Get the displacement field of the mesh in three dimensions (x, y, z)
        u = variables()
        # Reshape so that each displacement vector is in a row [x, y, z] displacements
        u = u.reshape((int(u.shape[0] / dim), dim))
        print('maximum displacement x:', np.abs(u[:, 0]).max())
        print('maximum displacement y:', np.abs(u[:, 1]).max())
        print('maximum displacement z:', np.abs(u[:, 2]).max())

        return u


if __name__ == "__main__":

    # rectangle_wave = TriangularMesh(
    #     RectangleMesh(20, 15, z_function=wave)
    # )
    # elbow = MeshFromFile(
    #     'C:/Users/majag/Desktop/marble/MARBLE/AI/meshes/elbow.mesh'
    # )
    # perfusion = MeshFromFile(
    #     'C:/Users/majag/Desktop/marble/MARBLE/AI/meshes/perfusion_micro3d.mesh'
    # )
    #
    # extruded_rectangle_wave = ExtrudedTriangularMesh(
    #     RectangleMesh(10, 5, z_function=wave)
    # )

    # cuboid = MeshFromFile(
    #     'C:/Users/majag/Desktop/marble/MARBLE/AI/meshes/block.mesh'
    # )

    # circle_wave = ExtrudedTriangularMesh(RectangleMesh(10, 15, z_function=flat))
    # circle_wave = ExtrudedTriangularMesh(CircleMesh(10, 15, z_function=wave))
    #
    # hexa_mesh = MeshFromFile(
    #     'C:/Users/majag/Desktop/marble/MARBLE/AI/meshes/acoustics_mesh3d.mesh'
    # )
    #
    hexa = GridMesh(10, 10, z_function=wave)

    display_mesh()
    # FENICS(hexa, rubber).solve()
    FENICS(hexa, rubber).apply_cell_force(10)

