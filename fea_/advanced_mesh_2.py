from fea_.basic_mesh_2 import *
from _csv import reader


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
    return 0


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


class ARM_mesh(Mesh):

    def __init__(self):
        # TODO
        super().__init__()
