import numpy as np


class Triangle:

    def __init__(self, p1, p2, p3):

        # Triangle consists of 3 sensor nodes
        self.nodes = [Node(p1), Node(p2), Node(p3)]

        self.lines = [
            Line(self.nodes[0], self.nodes[1]),
            Line(self.nodes[1], self.nodes[2]),
            Line(self.nodes[2], self.nodes[0])
        ]
        # Centroid of the triangle
        self.centroid = self.compute_centroid()

        # Larger elements are stiffer (harder to deform) than smaller ones.
        self.area = self.compute_area()

        # Coefficients of the lines that are perpendicular to the sides of the triangle
        self.b1 = self.lines[0].get_perpendicular_line_slope()
        self.b2 = self.lines[1].get_perpendicular_line_slope()
        self.b3 = self.lines[2].get_perpendicular_line_slope()

        # Distances from the centroid of the triangle to the sides.
        self.c1 = self.lines[0].get_distance_from_point(self.centroid)
        self.c2 = self.lines[1].get_distance_from_point(self.centroid)
        self.c3 = self.lines[2].get_distance_from_point(self.centroid)

    def compute_area(self):
        """
        :return: area of the triangle
        (1/2) |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|
        """
        x1, y1 = self.nodes[0].location
        x2, y2 = self.nodes[1].location
        x3, y3 = self.nodes[2].location

        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    def compute_centroid(self):
        """
        :return: The geometric center of the triangle
        """
        x1, y1 = self.nodes[0].location
        x2, y2 = self.nodes[1].location
        x3, y3 = self.nodes[2].location

        return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3

    def define_stiffness_matrix(self, material):
        """
        :return: 6x6 stiffness matrix
        Defines how the element will react to the applied forces
        It relates the displacements of the nodes to the forces acting on the nodes

        It assumes that the material is isotropic (the properties are the same in all directions)
        and homogeneous (the properties are the same throughout the material),
        and that the deformations are small.

        The global stiffness matrix K is then formed by adding the individual stiffness matrices k from each element
        to the appropriate locations.
        """

        A = self.area
        E = material.young_modulus
        v = material.poisson_ratio
        b1, b2, b3 = self.b1, self.b2, self.b3
        c1, c2, c3 = self.c1, self.c2, self.c3

        matrix = [
            [b1 ** 2, b1 * b2, 2 * v * b1 * c1, b1 * b2, b2 ** 2, 2 * v * b2 * c2],
            [b1 * b2, b1 ** 2, 2 * v * b1 * c1, b2 ** 2, b1 * b2, 2 * v * b2 * c2],
            [2 * v * b1 * c1, 2 * v * b1 * c1, c1 ** 2, 2 * v * b2 * c2, 2 * v * b2 * c2, c2 ** 2],
            [b1 * b3, b1 * b2, 2 * v * b1 * c1, b3 ** 2, b2 * b3, 2 * v * b3 * c3],
            [b2 * b3, b1 * b2, 2 * v * b2 * c2, b2 * b3, b3 ** 2, 2 * v * b3 * c3],
            [2 * v * b3 * c3, 2 * v * b3 * c3, c3 ** 2, 2 * v * b2 * c3, 2 * v * b2 * c3, c3 ** 2]
        ]
        return (A * E / (4 * (1 - v ** 2))) * np.array(matrix)

    def get_global_DOF_indices(self):
        """
        For a 2D problem, each node will typically have 2 degrees of freedom (DOF)
        - one for displacement in the X direction and one for displacement in the Y direction.
        :return:
        """
        return [node.DOF for node in self.nodes]


class Line:

    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

        self.m = self.get_slope()
        self.b = self.get_intercept()

    def get_slope(self):
        x1, y1 = self.point1.location
        x2, y2 = self.point2.location

        # Check if line has equation x = c, then slope is inf
        if abs(x1 - x2) < 1e-9:
            return np.inf
        # Check if line has equation y = c, then slope is 0
        if abs(y1 - y2) < 1e-9:
            return 0
        return (y2 - y1) / (x2 - x1)

    def get_intercept(self):
        x1, y1 = self.point1.location

        if self.m == np.inf:
            return -x1
        if abs(self.m) < 1e-9:
            return y1
        return y1 - self.m * x1

    def get_perpendicular_line_slope(self):
        """
        If the slopes of the two perpendicular lines are m1, m2,
        then we can represent the relationship between the slope of perpendicular lines
        with the formula m1 * m2 = -1.
        """

        # If line has equation x = c, then perpendicular line has equation y = c
        if self.m == np.inf:
            return 0

        # If line has equation y = c, then perpendicular line has equation x = c
        if self.m == 0:
            return np.inf

        return -1 / self.m

    def get_distance_from_point(self, point):
        """
        :return: distance from the centroid of the triangle
        """
        x, y = point
        # line has equation x = c
        if self.m == np.inf:
            distance = abs(x - self.point1.location[0])
        # line has equation y = c
        elif self.m == 0:
            distance = abs(y - self.point1.location[1])
        else:
            distance = abs((self.m * x - y + self.b) / np.sqrt(1 + self.m ** 2))
        return distance


class Node:

    def __init__(self, sensor):
        self.sensor = sensor

        # The degrees of freedom (DOF) are the unknown displacements of the node
        # in the X and Y directions
        self.ID = sensor.ID
        self.DOF = [
            self.sensor.ID * 2,  # x DOF
            self.sensor.ID * 2 + 1]  # y DOF

        self.location = self.sensor.real_position[:2]
