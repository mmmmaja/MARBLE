import numpy as np


# Element for the FAE method
# Each triangle in the mesh can be viewed as a flat plate
def get_material_property_matrix(material):
    E = material.young_modulus
    v = material.poisson_ratio
    G = E / (2 * (1 + v))  # Shear modulus
    # Major Poisson's ratios
    major_v = E / ((1 + v) * (1 - 2 * v))
    # Minor Poisson's ratios
    minor_v = (1 - v) / (2 * (1 - 2 * v))
    D = np.array([
        [major_v, minor_v, minor_v, 0, 0, 0],
        [minor_v, major_v, minor_v, 0, 0, 0],
        [minor_v, minor_v, major_v, 0, 0, 0],
        [0, 0, 0, G, 0, 0],
        [0, 0, 0, 0, G, 0],
        [0, 0, 0, 0, 0, G]
    ])
    return D


class Triangle:

    def __init__(self, p1, p2, p3):
        # CONSTANT STRAIN TRIANGLE
        #  consists of 3 sensor nodes
        self.nodes = [Node(p1), Node(p2), Node(p3)]

        # Larger elements are stiffer (harder to deform) than smaller ones.
        self.area = self.compute_area()

    def compute_area(self):
        """
        :return: area of the triangle in 3D
        |AB x AC| * 1/2
        """
        x1, y1, z1 = self.nodes[0].sensor.position
        x2, y2, z2 = self.nodes[1].sensor.position
        x3, y3, z3 = self.nodes[2].sensor.position

        # Find vector P1 -> P2
        P1P2_vector = [x2 - x1, y2 - y1, z2 - z1]

        # Find vector P1 -> P3
        P1P3_vector = [x3 - x1, y3 - y1, z3 - z1]

        # Take cross product of two vectors
        cross_product = np.cross(P1P2_vector, P1P3_vector)

        # Compute the magnitude of the cross product vector
        magnitude = np.sqrt(sum(pow(element, 2) for element in cross_product))

        return magnitude / 2

    def define_stiffness_matrix(self, material):
        """
        :return: 9x9 stiffness matrix
        Defines how the element will react to the applied forces
        It relates the displacements of the nodes to the forces acting on the nodes

        It assumes that the material is isotropic (the properties are the same in all directions)
        and homogeneous (the properties are the same throughout the material),
        and that the deformations are small.

        The global stiffness matrix K is then formed by adding the individual stiffness matrices k from each element
        to the appropriate locations.

        For a triangle with 3 nodes, the element stiffness matrix size has shape 9x9 (for 3D).
        """

        """
        [B]' * [D] * [B] * t * A
        
        [B] is the strain-displacement matrix
        [D] is the constitutive (material property) matrix
        t is the thickness of the plate or shell
        A is the area of the triangle
        """

        A = self.area
        B = self.get_strain_displacement_matrix()
        D = get_material_property_matrix(material)
        t = material.thickness

        return np.matmul(np.matmul(np.transpose(B), D), B) * t * A

    def get_strain_displacement_matrix(self):
        A = self.area
        multiplier = 1 / (2 * A)

        x1, y1, z1 = self.nodes[0].sensor.position
        x2, y2, z2 = self.nodes[1].sensor.position
        x3, y3, z3 = self.nodes[2].sensor.position
        # Compute factors for B matrix
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = z3 - z2
        c2 = z1 - z3
        c3 = z2 - z1
        B = np.array([
            [b1, 0, 0, b2, 0, 0, b3, 0, 0],
            [0, c1, 0, 0, c2, 0, 0, c3, 0],
            [0, 0, b1, 0, 0, b2, 0, 0, b3],
            [0, 0, c1, 0, 0, c2, 0, 0, c3],
            [c1, 0, 0, c2, 0, 0, c3, 0, 0],
            [0, b1, 0, 0, b2, 0, 0, b3, 0]
        ]) * multiplier
        return B

    def get_global_DOF_indices(self):
        """
        For a 2D problem, each node will typically have 2 degrees of freedom (DOF)
        - one for displacement in the X direction and one for displacement in the Y direction.
        :return:
        """
        return [node.DOF for node in self.nodes]


class Node:

    def __init__(self, sensor):
        # TODO later node might be not a sensor, subdivide previous elements
        self.sensor = sensor

        self.ID = sensor.ID

        # The degrees of freedom (DOF) are the displacements in the x, y, and z directions
        self.DOF = [
            self.sensor.ID * 3,  # x DOF
            self.sensor.ID * 3 + 1,  # y DOF
            self.sensor.ID * 3 + 2  # z DOF
        ]
