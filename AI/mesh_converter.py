import sys
import sfepy
from AI.mesh_helper import *
from abc import abstractmethod
import numpy as np
import meshio
import pyvista as pv
from sfepy.discrete.fem import Mesh
from copy import deepcopy

# Thickness of the mesh
THICKNESS = 1.53


def convert_to_vtk(path):
    """
    Convert a mesh file to a vtk object (UnstructuredGrid)
    :param path: Path to the mesh file
    """
    return pv.UnstructuredGrid(path)


class MeshBoost:

    # This is a parent class for all the meshes in the project

    def __init__(self, path=PATH):
        # Path to the .mesh file
        self.path = path

        # The mesh object in Sfepy format for calculations
        # The minimum and maximum z values of the top layer of the mesh
        self.sfepy_mesh, self.z_min, self.z_max = self.create_mesh()
        # The mesh object in .vtk format for visualization
        self.initial_vtk = convert_to_vtk(self.path)

        # Represents the current displacements of the mesh
        # Mind be changed later
        self.current_vtk = self.initial_vtk.copy()

        # The coordinates of the cells in the mesh in the top layer
        self.vtk_cells_coordinates = self.create_top_cells_coordinates()

    @abstractmethod
    def create_mesh(self) -> [sfepy.discrete.fem.mesh.Mesh, float, float]:
        """
        TODO override in subclasses
        :return: The mesh object in Meshio format
        """

    @abstractmethod
    def create_top_cells_coordinates(self) -> np.ndarray:
        """
        TODO override in subclasses
        :return: The coordinates of the cells in the mesh
        """

    def get_regions(self, domain):

        # Get the boundaries of the mesh
        min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
        min_y, max_y = domain.get_mesh_bounding_box()[:, 1]

        # Create a tolerance for the boundaries
        tol = 1e-1
        eps_x, eps_y = tol * (max_x - min_x), tol * (max_y - min_y)

        # Create the regions (they do not really represent top and the bottom regions,
        # but rather just some fixed points -> this method is meant to be overriden in the child classes anyway)
        bottom = domain.create_region('bottom', 'vertices in y < %.10f' % (min_y + eps_y), 'facet')
        top = domain.create_region('top', 'vertices in y > %.10f' % (min_y - eps_y), 'facet')
        return top, bottom

    def override_mesh(self, u):
        # Override the vtk version of the mesh
        # Copy the initial mesh
        self.current_vtk.points = self.initial_vtk.points.copy() + u
        # Check for negative z values
        # If present assign zero
        # self.current_vtk.points[:, 2] = np.where(self.current_vtk.points[:, 2] < 0, 0, self.current_vtk.points[:, 2])

    def update_mesh(self, u):
        # Update the vtk version of the mesh
        # Add displacement to the mesh points
        print('vtk: ', self.current_vtk.points.shape, 'u: ', u.shape)
        self.current_vtk.points += u

        # Check for negative z values
        # If present assign zero
        # self.current_vtk.points[:, 2] = np.where(self.current_vtk.points[:, 2] < 0, 0, self.current_vtk.points[:, 2])

        # 2) Update the sfepy version of the mesh
        # Save this mesh as a .vtk file
        """
        FIXME
            raise RuntimeError('elements cannot be oriented! (%s)' % key)
        RuntimeError: elements cannot be oriented! (3_8)
        # """
        # self.current_vtk.save(PATH)
        # self.sfepy_mesh = Mesh.from_file(PATH)

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


class GridMesh(MeshBoost):

    def __init__(self, width, height, z_function=flat, layers=2):
        """
        Defines the mesh as a grid of vertices
        Consists of HEXA elements(!)

        :param width: dimension of the grid
        :param height: dimension of the grid
        :param z_function: function that defines the height of the grid
        :param layers: number of layers in the grid (in z direction)
        """
        self.width, self.height = width, height
        self.z_function, self.layers = z_function, layers

        super().__init__()

    def create_mesh(self):

        # Create the vertices of the tetrahedrons
        # Define two regions of the mesh
        all_layers = []
        for i in range(self.layers):
            all_layers.append([])

        # Add the vertices to the regions
        for i in range(self.height):
            for j in range(self.width):
                all_layers[0].append([
                    i, j, self.z_function(i, j, self.width, self.height)
                ])
                for k in range(1, self.layers):
                    all_layers[k].append([
                        i, j, (self.layers - k - 1) * THICKNESS
                    ])

        # ADJUST THE TOP LAYER (where the deformations will be applied)

        # Convert all_layers[0] to numpy array for vectorized operation
        top_layer = np.array(all_layers[0])

        # Adjust the z-coordinates and ensure all are non-negative
        top_layer[:, 2] = np.maximum(top_layer[:, 2] - np.amin(top_layer[:, 2]), 0)

        # Add the thickness to the top vertices
        top_layer[:, 2] += (self.layers - 1) * THICKNESS

        # Get the min and max z-coordinates in the top layer
        z_min, z_max = np.amin(top_layer[:, 2]), np.amax(top_layer[:, 2])

        # Convert the adjusted numpy array back to list and assign it back to all_layers[0]
        all_layers[0] = top_layer.tolist()

        # Combine all the vertices
        vertices = [vertex for vertices in all_layers for vertex in vertices]
        n = len(vertices) // self.layers

        # Create the cells (give the indices of vertices of each hexahedron)
        cells = []
        for i in range(self.height - 1):
            for j in range(self.width - 1):
                # Go through all the layers
                for k in range(self.layers - 1):
                    a_top, b_top = i * self.width + j + k * n, i * self.width + j + 1 + k * n
                    c_top, d_top = (i + 1) * self.width + j + 1 + k * n, (i + 1) * self.width + j + k * n
                    cells.append([
                        a_top, b_top, c_top, d_top,
                        a_top + n, b_top + n,  c_top + n, d_top + n
                    ])

        # Create hexahedron mesh in the meshio format
        mesh = meshio.Mesh(points=vertices, cells={"hexahedron": cells})
        meshio.write(PATH, mesh)
        # Convert the meshio mesh to sfepy mesh
        return Mesh.from_file(PATH), z_min, z_max

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
        n = self.sfepy_mesh.n_nod // self.layers

        # Create a top region (Where the displacements happen)
        top_range = range(n)
        expr_base = 'vertex ' + ', '.join([str(i) for i in top_range])
        top = domain.create_region(name='Top', select=expr_base, kind='facet')

        # Create a bottom region (Where the boundary conditions apply so that the positions are fixed)
        bottom_range = range((self.layers - 1) * n, self.layers * n)
        # Define the cells by their Ids and use vertex <id>[, <id>, ...]
        expr_extruded = 'vertex ' + ', '.join([str(i) for i in bottom_range])
        bottom = domain.create_region(name='Bottom', select=expr_extruded, kind='facet')

        return top, bottom

    def create_top_cells_coordinates(self):
        # Ensure the mesh cells are hexahedrons (12 nodes per cell)

        hexa_cells = []

        # Extract the hexahedron cells
        # The first number in each cell is the number of points (8 for hexahedrons)
        cells = self.current_vtk.cells.reshape(-1, 9)[:, 1:]

        # Loop through all the cells
        for i in range(self.current_vtk.n_cells):
            hexahedron = cells[i]
            cell_points = self.current_vtk.points[hexahedron]  # Get the cell points by using the indices
            # Get the top face of the hexahedron (4 points)
            top_face = cell_points[np.argsort(cell_points[:, 2])[-4:]]

            # Check if the z coordinate of all points in the cell are within the range
            if self.z_min <= top_face[:, 2].min() <= top_face[:, 2].max() <= self.z_max:
                hexa_cells.append(top_face)

        return hexa_cells
