import math
import random
from fea_.basic_mesh_2 import *
from _csv import reader


def generate_points_per_circle(radius, N, randomize=False):

    points = []
    for i in range(N):
        # Get Angle in radians
        if randomize:
            theta = 2 * np.pi * random.random()
        else:
            theta = 2 * np.pi * i / N
        # Add point to results
        points.append([radius * math.cos(theta), radius * math.sin(theta)])
    return points


class CircleMesh(Mesh):

    def __init__(self, net_number, sensors_per_net, z_function, distance_between_nets=1):
        """
        Crate a circular mesh (net-like)
        :param net_number: number of layers
        :param sensors_per_net: number of sensors per layer
        :param z_function: function that determines the height of each sensor
        :param distance_between_nets: distance between each two nets
        """
        self.net_number = net_number
        self.sensors_per_net = sensors_per_net
        self.z_function = z_function
        self.distance_between_nets = distance_between_nets

        super().__init__()

    def create(self):
        # Create net_number nets that are circular
        # Each net has sensors_per_net sensors

        # Add the center sensor
        sensor_array = [
            Sensor(position=[0.0, 0.0, 0])
        ]
        radius = 1.0
        for i in range(self.net_number):
            points = generate_points_per_circle(radius, self.sensors_per_net)
            for p in points:
                sensor_array.append(Sensor(position=[
                    p[0], p[1], self.z_function(p[0], p[1], radius, self.net_number)
                ]))
            radius += self.distance_between_nets
        return sensor_array


class RectangleMesh(Mesh):

    def __init__(self, width, height, z_function, sensor_distance=1):
        """
        :param width: dimension of the grid
        :param height: dimension of the grid
        :param sensor_distance: distance between each two sensors in a grid
        """
        self.width = width
        self.height = height

        self.z_function = z_function
        self.sensor_distance = sensor_distance

        super().__init__()

    def create(self):
        """
        :return: A sensor array mesh (grid-like)
        """

        sensor_array = []
        for i in range(self.height):
            for j in range(self.width):
                sensor_array.append(Sensor(position=[
                    i * self.sensor_distance,
                    j * self.sensor_distance,
                    self.z_function(i, j, self.width, self.height)
                ]))
        return sensor_array


def concave(i, j, width, height):
    concavity_factor = 0.05
    centre = [width / 2, height / 2]
    x = i - centre[0]
    y = j - centre[1]
    z = -concavity_factor * (x ** 2 + y ** 2)
    return z


def flat(i, j, width, height):
    return 0.0


def convex(i, j, width, height):
    concavity_factor = 0.15
    centre = [width / 2, height / 2]
    x = i - centre[0]
    y = j - centre[1]
    z = concavity_factor * (x ** 2 + y ** 2)
    return z


def wave(i, j, width, height):
    wave_factor = 3.5
    frequency = 0.2
    z = wave_factor * math.sin(i * frequency) * math.sin(j * frequency)
    return z


class csvMesh(Mesh):

    def __init__(self, path):
        self.path = path
        super().__init__()

    def create(self):
        """"
        Read positions of the sensors from a csv file
        """

        with open(self.path, 'r', newline='') as file:
            csv_reader = reader(file, delimiter=';')
            sensor_positions = None
            for row in csv_reader:
                sensor_positions = row
                break

        sensor_array = []
        for i in sensor_positions:
            coordinates = i.split(',')
            if len(coordinates) != 3:
                break
            sensor_array.append(Sensor(position=[
                float(coordinates[0]),
                float(coordinates[1]),
                float(coordinates[2]),
            ]))
        return sensor_array


