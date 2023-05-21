from mesh import *


class RectangleMesh(Mesh):

    def __init__(self, width, height, sensor_distance=1):
        self.width = width
        self.height = height
        self.sensor_distance = sensor_distance
        super().__init__()

    def create(self):
        sensor_array = []
        for i in range(self.height):
            for j in range(self.width):
                sensor_array.append(Sensor(real_position=[
                    i * self.sensor_distance,
                    j * self.sensor_distance,
                    0
                ]))
        return sensor_array

    def set_displayed_points(self):
        INDEX = 0

        sensor_line = []
        for i in range(self.height):
            index = i * self.width + INDEX
            sensor_line.append(self.SENSOR_ARRAY[index])
        return sensor_line


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
            sensor_array.append(Sensor(real_position=[
                float(coordinates[0]),
                float(coordinates[1]),
                float(coordinates[2]),
            ]))
        return sensor_array

    def set_displayed_points(self):
        return [self.SENSOR_ARRAY[0], self.SENSOR_ARRAY[1]]


class ARM_mesh(Mesh):

    def __init__(self):
        super().__init__()
