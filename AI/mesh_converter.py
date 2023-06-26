from AI.mesh_helper import *
from abc import abstractmethod
import numpy as np
import meshio
import pyvista as pv
from sfepy.discrete.fem import Mesh

# Thickness of the mesh (Applies to the extruded meshes only)
THICKNESS = 0.53


def convert_to_vtk(path):
    return pv.UnstructuredGrid(path)


class MeshBoost:

    def __init__(self, path=None):

        # Path to the .mesh file
        if path is None:
            self.path = PATH
        else:
            self.path = path

        self.meshio_mesh = self.create_mesh()

        self.initial_vtk = convert_to_vtk(self.path)
        self.current_vtk = self.initial_vtk.copy()

    @abstractmethod
    def create_mesh(self) -> None:
        """
        TODO override in subclasses
        :return: The mesh object in Meshio format
        """

    def get_regions(self, domain):

        min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
        min_y, max_y = domain.get_mesh_bounding_box()[:, 1]

        tol = 1e-1
        eps_x, eps_y = tol * (max_x - min_x), tol * (max_y - min_y)

        bottom = domain.create_region('bottom', 'vertices in y < %.10f' % (min_y + eps_y), 'facet')
        top = domain.create_region('top', 'vertices in y > %.10f' % (min_y - eps_y), 'facet')
        return top, bottom

    def update_vtk(self, u):
        """
        Updates the mesh with the new displacement
        mesh in the .vtk format

        :param u: displacement of the mesh
        :return: copy of the mesh with updated displacements
        """

        # Create a copy of the mesh
        self.current_vtk = self.initial_vtk.copy()

        # Apply displacement to the mesh points
        self.current_vtk.points += u

        # Check for negative z values
        # If present assign zero
        self.current_vtk.points[:, 2] = np.where(self.current_vtk.points[:, 2] < 0, 0, self.current_vtk.points[:, 2])

    def get_vertex_ids_from_coords(self, cell_coords):
        """
        Given a mesh and cell coordinates, find the matching vertex IDs in the mesh.
        """
        mesh_points = self.current_vtk.points
        vertex_ids = []

        for cell_point in cell_coords:
            for i, mesh_point in enumerate(mesh_points):
                if np.allclose(cell_point, mesh_point):
                    vertex_ids.append(i)
                    break

        return vertex_ids


class MeshFromFile(MeshBoost):

    def __init__(self, path):
        """
        Read the mesh from a file (supports .mesh and .vtk files)
        :param path: path to the .mesh file
        """
        self.path = path
        super().__init__(path)

    def create_mesh(self):
        return Mesh.from_file(self.path)


class GridMesh(MeshBoost):

    def __init__(self, width, height, z_function=flat, cell_distance=1):
        """
        Defines the mesh as a grid of sensors
        :param width: dimension of the grid
        :param height: dimension of the grid
        :param z_function: function that defines the height of the grid
        :param cell_distance: distance between each cells in a grid

        TODO add sensors later
        """
        self.width = width
        self.height = height

        self.z_function = z_function
        self.cell_distance = cell_distance

        super().__init__()

    def create_mesh(self):

        # Create the vertices of the tetrahedrons
        # Define two regions of the mesh
        top_vertices, bottom_vertices = [], []
        for i in range(self.height):
            for j in range(self.width):

                top_vertices.append([
                    i * self.cell_distance,
                    j * self.cell_distance,
                    self.z_function(i, j, self.width, self.height) + THICKNESS
                ])
                bottom_vertices.append([
                    i * self.cell_distance,
                    j * self.cell_distance,
                    0
                ])

        # Assign the same z-coordinate to all the bottom vertices (the smallest top z-coordinate)
        Z = np.amin(np.array(top_vertices)[:, 2])
        if Z < 0:
            for v in top_vertices:
                v[2] -= Z

        # Combine the top and bottom vertices
        vertices = np.concatenate([top_vertices, bottom_vertices], axis=0)
        n = len(vertices) // 2

        # Create the cells (give the indices of vertices of each tetrahedron)
        cells = []
        for i in range(self.height - 1):
            for j in range(self.width - 1):

                a_bottom, b_bottom = i * self.width + j, i * self.width + j + 1
                c_bottom, d_bottom = (i + 1) * self.width + j + 1, (i + 1) * self.width + j

                a_top, b_top = a_bottom + n, b_bottom + n
                c_top, d_top = c_bottom + n, d_bottom + n

                cells.append([
                    a_bottom, b_bottom, c_bottom, d_bottom,
                    a_top, b_top, c_top, d_top
                ])

        # Create hexahedron mesh
        mesh = meshio.Mesh(points=vertices, cells={"hexahedron": cells})
        meshio.write(PATH, mesh)
        return Mesh.from_file(PATH)

    def get_regions(self, domain):
        """
        https://sfepy.org/doc-devel/users_guide.html
        From documentation:
        Regions serve to select a certain part of the computational domain using topological entities of the FE mesh.
        They are used to define the boundary conditions, the domains of terms and materials etc.
        :return: top and bottom regions
        """

        # vertices = np.concatenate([top_vertices, bottom_vertices], axis=0)
        # Get the number of vertices
        n = self.meshio_mesh.n_nod // 2

        # Create a top region (Where the displacements happen)
        top_range = range(n)
        expr_base = 'vertex ' + ', '.join([str(i) for i in top_range])
        top = domain.create_region(name='Top', select=expr_base, kind='facet')

        # Create a bottom region (Where the boundary conditions apply so that the positions are fixed)
        bottom_range = range(n, 2 * n)
        # Define the cells by their Ids and use vertex <id>[, <id>, ...]
        expr_extruded = 'vertex ' + ', '.join([str(i) for i in bottom_range])
        bottom = domain.create_region(name='Bottom', select=expr_extruded, kind='facet')

        return top, bottom


