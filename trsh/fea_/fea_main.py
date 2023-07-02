import sys
import numpy as np
from scipy.sparse import coo_matrix
from trsh.fea_.components import Triangle
from scipy.sparse.linalg import spsolve
from AI.ai import pressure_image


class FEA:

    def __init__(self, mesh, material, stimuli):
        self.mesh = mesh
        self.material = material
        self.stimuli = stimuli

        # Triangle elements (Check later if elements are not too big)
        self.elements = self.create_elements()

        # Define global stiffness matrix
        self.K = self.create_global_stiffness_matrix()

    def introduce_displacement(self, u):
        """
        Update the geometry of the mesh with the displacements from the previous step
        :param u: The displacement vector u contains the displacements in xyz direction for each node in the mesh
        """

        # Reshape the displacement vector into a 3xN matrix
        u = u.reshape((3, -1)).T
        # update the geometry with the displacements from the previous step
        self.mesh.update_geometry(u)

    def create_elements(self):
        """
        Create a list of elements from the mesh of nodes
        Here using a triangular mesh so each element is a triangle
        """

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

    def show_pressure(self):
        # Get the force vector
        F = np.reshape(self.get_force(), (-1, 3))
        for i in range(len(self.mesh.SENSOR_ARRAY)):
            # FIXME this is not real pressure but the force in z axis applied
            self.mesh.SENSOR_ARRAY[i].pressure = F[i, 2]
        pressure_image.form_contact_image(self.mesh)
        sys.exit(1)

    def apply_pressure(self):

        # Get the force vector
        F = self.get_force()

        """
        Solve the system of equations Ku = F using the Newton-Raphson method
        K U = F
        """
        # initial guess for the displacements
        U = np.zeros_like(F)

        # tolerance for the relative change in the solution between iterations
        tol = 1e-6

        # maximum number of iterations
        max_iter = 100

        for _ in range(max_iter):
            # compute the residual force R
            # The residual force is force that is left over after the applied force is removed.
            R = self.K.dot(U) - F

            # Residual is supposed to get as close to zero as possible
            print("Residual: ", np.linalg.norm(R))

            # check if the solution has converged
            if np.linalg.norm(R) < tol:
                break

            # Create a Tangent stiffness matrix
            # The tangent stiffness matrix is the derivative of the residual force with respect to the displacement.
            # compute the change in displacements
            delta_u = spsolve(self.K, -R)

            # compute the change in displacements
            delta_u = spsolve(self.K, -R)

            # update the displacements
            U += delta_u

            # update the geometry and the stiffness matrix
            self.introduce_displacement(U)
            self.K = self.create_global_stiffness_matrix()

    def get_force(self):
        """
        The pressure is applied to the nodes of the elements that are activated.
        The pressure is applied in the z-direction, so it will only affect the z-displacement of the nodes.
        The pressure is applied as a force, so it will be converted to a force using the area of the element.
        :return:
        """

        # Get number of nodes in the sensor array
        num_nodes = len(self.mesh.SENSOR_ARRAY)

        # create an empty list to hold the forces for each node
        forces = np.zeros((num_nodes, 3))
        for i in range(num_nodes):
            # Get the force applied
            sensor_position = self.mesh.SENSOR_ARRAY[i].position
            force = self.stimuli.get_pressure(sensor_position)
            forces[i] = force

        # Create a force matrix of size num_nodes * 3 (# DOF)
        # The force matrix F is a vector of length equal to the number of degrees of freedom.
        F = forces.flatten()
        return F


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

    def get_properties(self):
        """
        Returns additional material properties
        :return: mu and lambda
        """
        E = self.young_modulus
        nu = self.poisson_ratio

        # # Mu is the shear modulus (shows the material's resistance to deformation)
        # mu = Constant(E / (2.0 * (1.0 + nu)))
        # # Lambda is the Lame parameter (defines the relationship between stress and strain)
        # lambda_ = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
        #
        # return mu, lambda_

