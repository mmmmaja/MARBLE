import sys
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from fea_.components import Triangle
from scipy.sparse.linalg import spsolve


class FEA:

    def __init__(self, mesh, material):
        self.mesh = mesh

        # Triangle elements (Check later if elements are not too big)
        self.elements = self.create_elements()
        self.material = material

        # Define global stiffness matrix
        self.K = self.create_global_stiffness_matrix()

    def solve(self, F):
        """
        Solve the system of equations Ku = F
        :return: The displacement vector u contains the displacements in x and y direction for each node in the mesh
        """

        # Solve for displacements
        u = spsolve(self.K, F)

    def create_elements(self):
        elements = []

        # Array of indexes of 3 senors from SENSOR_ARRAY
        for sensor_indices in self.mesh.delaunay_points:
            p1 = self.mesh.SENSOR_ARRAY[sensor_indices[0]]
            p2 = self.mesh.SENSOR_ARRAY[sensor_indices[1]]
            p3 = self.mesh.SENSOR_ARRAY[sensor_indices[2]]
            elements.append(Triangle(p1, p2, p3))
        return elements

    def create_global_stiffness_matrix(self):
        """
        If most of the elements in the matrix are zero then the matrix is called a sparse matrix.
        It is wasteful to store the zero elements in the matrix since they do not affect the results
        of the computation.

        Compressed Sparse Row (CSR) Matrix
        :return:
        """

        # Create a sparse matrix
        # COO stores a list of (row, column, value) tuples
        data, rows, cols = [], [], []

        # loop through all elements
        for element in self.elements:

            # calculate the stiffness matrix for this element
            k_element = element.define_stiffness_matrix(self.material)

            # get the global indices for the nodes of this element
            # 3x3 matrix so the overall length is 9
            node_indices = np.array(element.get_global_DOF_indices()).flatten()

            # loop through the entries of the stiffness matrix
            for i in range(len(k_element)):
                for j in range(len(k_element[i])):
                    # append the data and the row/column indices to the lists
                    data.append(k_element[i][j])
                    rows.append(node_indices[i])
                    cols.append(node_indices[j])

        # Number of sensors in the mesh
        # Since we have 3 DOF per node, we multiply by 3
        num_nodes = len(self.mesh.SENSOR_ARRAY) * 3

        # create the CSR matrix
        K = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        return K


class Material:

    def __init__(self, density, young_modulus, poisson_ratio, thickness):
        """
        :param density: [g/cm^3]
            Measure of material's mass per unit volume

        :param young_modulus: [Gpa]
             Property of the material that tells us how easily it can stretch and deform
             and is defined as the ratio of tensile stress (σ) to tensile strain (ε)

             A material with a high Young's modulus will be very stiff (like steel),
             while a material with a low Young's modulus will be very flexible (like rubber).

        :param poisson_ratio: [Gpa]
            Poisson's ratio is a measure of the Poisson effect,
            the phenomenon in which a material tends to expand in directions perpendicular
            to the direction of compression.

            It is a property of the material that describes how a material tends to shrink
            in one direction when being stretched in another.
            For most materials, it's a value between 0 and 0.5.

        :param thickness: [mm]
        """

        self.density = density
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.thickness = thickness


# MAIN

