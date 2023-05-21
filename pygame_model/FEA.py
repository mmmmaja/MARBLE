import sys

import gmsh
import numpy as np
import pyvista as pv
import numpy.typing
import meshio
from pygame_model import advanced_mesh

import sfepy

from sfepy.base.base import Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Integrals,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import FEDomain, Field, Mesh
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson, lame_from_youngpoisson
from sfepy.mechanics.tensors import get_von_mises_stress

gmsh.initialize()


def create_mesh(_mesh):
    # Create a new model
    gmsh.model.add("t1")

    # 1. Add points to the mesh
    point_tags = []
    for s in _mesh.SENSOR_ARRAY:
        point_tags.append(gmsh.model.geo.addPoint(s.real_position[0], s.real_position[1], s.real_position[2]))

    line_tags = []

    # used to define a loop composed of multiple curves or line segments
    curve_loop_tags = []

    surface_filling_tags = []

    for t in _mesh.delaunay_points:
        line_tags.append(
            gmsh.model.geo.addLine(point_tags[t[0]], point_tags[t[1]])
        )
        line_tags.append(
            gmsh.model.geo.addLine(point_tags[t[1]], point_tags[t[2]])
        )
        line_tags.append(
            gmsh.model.geo.addLine(point_tags[t[2]], point_tags[t[0]])
        )

        curve_loop_tags.append(
            gmsh.model.geo.addCurveLoop([line_tags[-3], line_tags[-2], line_tags[-1]])
        )
        surface_filling_tags.append(
            gmsh.model.geo.addSurfaceFilling([curve_loop_tags[-1]])
        )

    surface_loop = gmsh.model.geo.addSurfaceLoop(surface_filling_tags)
    # volume = gmsh.model.geo.addVolume([surface_loop])

    # Create the relevant Gmsh data structures
    # from Gmsh model.
    gmsh.model.geo.synchronize()

    # Generate mesh:
    gmsh.model.mesh.generate()

    # Save the mesh to a file
    gmsh.write("mesh.msh")

    # Creates  graphical user interface
    if 'close' not in sys.argv:
        gmsh.fltk.run()

    # It finalizes the Gmsh API
    gmsh.finalize()


def fea(material):
    # Read a finite element mesh, that defines the domain and FE approximation.
    mesh = Mesh.from_file('mesh.msh')

    # Create a domain that consists of the mesh
    domain = FEDomain('domain', mesh)

    # We call the entire domain omega
    omega = domain.create_region('Omega', 'all')

    # Define a field variable with name 'u' and with 3 components
    # (3 DOFs per node) and approximation order 1 (line elements).
    field = Field.from_args('fu', np.float64, 'vector', omega, 1)

    # Define the displacement field
    u = FieldVariable('u', 'unknown', field)

    # Define the test field
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    # Define the integral type Volume/Surface and quadrature rule
    integral = Integral('i', order=2)


csv_mesh = advanced_mesh.csvMesh('meshes_csv/web.csv')
create_mesh(csv_mesh)


# 1. Preprocessing

# 1.1 Define the geometry of the mesh

# 1.2 Assign material properties to the mesh


class Mesh_Material:

    def __init__(self, density, young_modulus, poisson_ratio):
        """
        :param density: [g/cm^3]
            Measure of material's mass per unit volume

        :param young_modulus: [Gpa]
             Property of the material that tells us how easily it can stretch and deform
             and is defined as the ratio of tensile stress (σ) to tensile strain (ε)

        :param poisson_ratio: [Gpa]
            Poisson's ratio is a measure of the Poisson effect,
            the phenomenon in which a material tends to expand in directions perpendicular
            to the direction of compression
        """
        self.material = self.define(density, young_modulus, poisson_ratio)

    def define(self, density, young_modulus, poisson_ratio):
        # Compute 2 frst Lamé parameters from Young's modulus and Poisson's ratio.
        lam, mu = lame_from_youngpoisson(young_modulus, poisson_ratio)

        # Compute stiffness tensor corresponding to Young's modulus and Poisson's ratio.
        mtx = stiffness_from_youngpoisson(2, young_modulus, poisson_ratio, plane='stress')

        return Material('mesh material', lam=lam, mu=mu, D=mtx, rho=density)


silicon = Mesh_Material(2.329, 140, 0.265)
fea(silicon)

# 2. Discretization

# Generate a finite element mesh: Convert your triangular mesh into a finite element mesh.
# You can use existing mesh generation libraries in Python, such as meshpy or gmsh, to create the finite element mesh.
# Ensure that the generated mesh is compatible with the FEA software or library you plan to use.

# use SfePy
