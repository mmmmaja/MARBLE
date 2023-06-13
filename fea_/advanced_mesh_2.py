from fea_.basic_mesh_2 import *
from _csv import reader


class RectangleMesh(Mesh):

    def __init__(self, width, height, sensor_distance=1):
        """
        :param width: dimension of the grid
        :param height: dimension of the grid
        :param sensor_distance: distance between each two sensors in a grid
        """
        self.width = width
        self.height = height
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
                    0.0
                ]))
        return sensor_array


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
