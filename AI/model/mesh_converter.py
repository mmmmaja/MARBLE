import pyvista
import sfepy
from AI.model.mesh_helper import *
from abc import abstractmethod
import numpy as np
import meshio
import pyvista as pv
from sfepy.discrete.fem import Mesh


def convert_to_vtk(path):
    """
    Convert a mesh file to a vtk object (UnstructuredGrid)
    :param path: Path to the mesh file
    """
    return pv.UnstructuredGrid(path)


class MeshBoost:

    # Limit in the deformation of the mesh in the z direction
    Z_DEFORMATION_LIMIT = 2e-2

    # This is a parent class for all the meshes in the project

    def __init__(self):
        # Path to the .mesh file
        self.path = '../meshes/mesh.mesh'

        # The mesh object in Sfepy format for calculations
        # The minimum and maximum z values of the top layer of the mesh
        self.sfepy_mesh = self.create_mesh()
        # The mesh object in .vtk format for visualization
        self.initial_vtk = convert_to_vtk(self.path)

        # Represents the current displacements of the mesh
        # Mind be changed later
        self.current_vtk = self.initial_vtk.copy()

    @abstractmethod
    def create_mesh(self) -> sfepy.discrete.fem.mesh.Mesh:
        """
        TODO override in subclasses
        :return: The mesh object in Meshio format
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
        self.current_vtk.points[:, 2] = np.where(
            self.current_vtk.points[:, 2] < self.Z_DEFORMATION_LIMIT,
            self.Z_DEFORMATION_LIMIT, self.current_vtk.points[:, 2]
        )

    def update_mesh(self, u):
        # Update the vtk version of the mesh
        # Add displacement to the mesh points
        print('vtk: ', self.current_vtk.points.shape, 'u: ', u.shape)
        self.current_vtk.points += u

        # Check for negative z values
        # If present assign zero
        self.current_vtk.points[:, 2] = np.where(
            self.current_vtk.points[:, 2] < self.Z_DEFORMATION_LIMIT,
            self.Z_DEFORMATION_LIMIT, self.current_vtk.points[:, 2]
        )

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

    # Thickness of the mesh
    THICKNESS = 2.73

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
                        i, j, (self.layers - k - 1) * self.THICKNESS
                    ])

        # ADJUST THE TOP LAYER (where the deformations will be applied)

        # Convert all_layers[0] to numpy array for vectorized operation
        top_layer = np.array(all_layers[0])

        # Adjust the z-coordinates and ensure all are non-negative
        top_layer[:, 2] = np.maximum(top_layer[:, 2] - np.amin(top_layer[:, 2]), 0)

        # Add the thickness to the top vertices
        top_layer[:, 2] += (self.layers - 1) * self.THICKNESS

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
        meshio.write(self.path, mesh)
        # Convert the meshio mesh to sfepy mesh
        return Mesh.from_file(self.path)

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


class ArmMesh(MeshBoost):
    # Thickness of the mesh
    THICKNESS = 0.95

    OBJ_PATH = '../meshes/model_kfadrat.obj'

    def __init__(self):
        # Add the indices of the vertices to create the regions for the solver later
        self.top_region_ids, self.bottom_region_ids = [], []
        super().__init__()

    def create_mesh(self):
        """
        Extruding the mesh along a direction perpendicular to the faces
        :return:
        """

        # Load the input mesh
        mesh = pv.read(self.OBJ_PATH)

        # Compute normals for each face
        normals = mesh.face_normals

        # Create a dictionary to handle shared vertices
        # base vertex coordinates  : [extruded vertex coordinates, extruded vertex id]
        vertices_dict = {}

        # For each face in the input mesh find the extruded face coordinates
        for i in range(mesh.n_cells):

            # Get the point coordinates for this face
            face_coords = mesh.get_cell(i).points

            # Loop over the vertices of the base (non-extruded) face
            for k in range(4):

                # Create a unique identifier for each vertex:
                # - a tuple of the vertex's coordinates
                identifier = tuple(np.round(face_coords[k], 4))

                if identifier not in vertices_dict:
                    extruded_coords = tuple(np.round(face_coords[k] - normals[i] * self.THICKNESS, 4))
                    vertices_dict[identifier] = extruded_coords

        # Make another pass to form the connections and create the cells

        # Create the vertices and cells for the output mesh (vtk format)
        vertices, cells = [], []
        # For each face in the input mesh, create a hexahedral cell in the output mesh
        for i in range(mesh.n_cells):
            face_coords = mesh.get_cell(i).points

            # The cell is defined by eight points: four points on the base face
            # and their corresponding extruded points
            cell_base, cell_extruded = [], []
            for k in range(4):
                identifier = tuple(np.round(face_coords[k], 4))

                # Handle the base vertices
                # Check if the vertex is already in the list of vertices
                if identifier not in vertices:
                    vertices.append(identifier)
                    vertex_id = vertices.index(identifier)
                else:
                    vertex_id = vertices.index(identifier)
                cell_base.append(vertex_id)  # base point
                self.top_region_ids.append(vertex_id)

                # Handle the extruded vertices
                extruded_point = vertices_dict[identifier]
                if extruded_point not in vertices:
                    vertices.append(extruded_point)
                    vertex_id = vertices.index(extruded_point)
                else:
                    vertex_id = vertices.index(extruded_point)
                cell_extruded.append(vertex_id)  # extruded point
                self.bottom_region_ids.append(vertex_id)

            # Combine the front and back faces to form the cell
            cell = cell_base[::-1] + cell_extruded[::-1]
            cells.append(cell)

        mesh = meshio.Mesh(points=vertices, cells={"hexahedron": cells})
        meshio.write(self.path, mesh)
        return Mesh.from_file(self.path)

    def get_regions(self, domain):

        expr_base = 'vertex ' + ', '.join([str(i) for i in self.top_region_ids])
        top = domain.create_region(name='Top', select=expr_base, kind='facet')

        # Create a bottom region (Where the boundary conditions apply so that the positions are fixed)
        # Define the cells by their Ids and use vertex <id>[, <id>, ...]
        expr_extruded = 'vertex ' + ', '.join([str(i) for i in self.bottom_region_ids])
        bottom = domain.create_region(name='Bottom', select=expr_extruded, kind='facet')

        return top, bottom


def display_obj_file(path):
    reader = pyvista.get_reader(path)
    mesh = reader.read()
    mesh.plot(cpos='yz', show_scalar_bar=False)
