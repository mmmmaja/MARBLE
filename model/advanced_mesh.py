from model.basic_mesh import *


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


class RandomMesh(Mesh):

    def __init__(self,save_path,area, sensor_cnt,min_sep, seed = 1):
        np.random.seed(seed)
        self.area = area
        self.sensor_cnt = sensor_cnt
        self.min_sep = min_sep
        print(self.sensor_cnt)
        super().__init__(save_path = save_path)

    def create(self):
        """"
        Read positions of the sensors from a csv file
        """

        locs = []
        sensor_array = []
        for i in range(self.sensor_cnt):
            loc = np.zeros(len(self.area))

            while True:
                for i, r_ in enumerate(self.area):
                    loc[i] = (r_[1]-r_[0])*np.random.random() + r_[0]

                repeat = False
                for loc_ in locs:
                    if np.linalg.norm(loc_ - loc) < self.min_sep:
                        repeat = True
                        break
                if not repeat: break

            locs.append(loc)
            sensor_array.append(Sensor(real_position=loc))
        return sensor_array

    def set_displayed_points(self):
        return [self.SENSOR_ARRAY[0], self.SENSOR_ARRAY[1]]



# Here the pressure data will be saved
DATA = []


def set_frame_position(sensor_array):

    # Check if coordinates are negative and if yes, then shift the whole frame
    min_x, min_y = 0, 0
    for sensor in sensor_array:
        if sensor.real_position[0] < min_x:
            min_x = sensor.real_position[0]
        if sensor.real_position[1] < min_y:
            min_y = sensor.real_position[1]

    # Shift mesh from the corners to the center
    delta = [50, 50]
    for sensor in sensor_array:
        sensor.real_position = sensor.real_position - [min_x, min_y, 0]
        sensor.frame_position = sensor.real_position[:2] * UNIT + delta


def triangulate(sensor_array):
    # Creates a triangulation of the sensor array in order to display mesh for the points
    positions = []
    for s in sensor_array:
        positions.append(s.frame_position)

    delaunay = Delaunay(np.array(positions))  # triangulate projections

    triangles = []
    for triangle in delaunay.simplices:
        t = []
        for i in triangle:
            t.append(positions[i])
        triangles.append(t)
    return delaunay.simplices, triangles


class Mesh:

    def __init__(self,save_path = "../pygame_model/data.csv"):
        self.SENSOR_ARRAY = self.create()
        set_frame_position(self.SENSOR_ARRAY)
        self.delaunay_points, self.triangles = triangulate(self.SENSOR_ARRAY)
        self.displayed_points = self.set_displayed_points()
        self.save_path = save_path

    def create(self):
        # To be overridden by a child class
        return None

    def set_displayed_points(self):
        # To be overridden by a child class
        return None

    def press(self, stimuli):
        # Record the pressure
        for sensor in self.SENSOR_ARRAY:
            sensor.press(stimuli)

    def append_data(self):
        data = []
        for i in self.SENSOR_ARRAY:
            data.append(i.deformation)
        DATA.append(data)

    def save_data(self):

        sensor_positions = []
        for i in self.SENSOR_ARRAY:
            pos = str(i.real_position[0]) + ',' + str(i.real_position[1]) + ',' + str(i.real_position[2])
            sensor_positions.append(pos)
        with open(self.path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(sensor_positions)
            writer.writerows(DATA)


class Sensor:

    def __init__(self, real_position=None):

        self.real_position = np.array(real_position)
        self.frame_position = None

        self.activated = False
        self.deformation = 0

    def press(self, stimuli):
        distance = stimuli.get_distance(self.real_position)

        # stimuli directly presses on sensor
        if distance == 0:
            self.deformation = stimuli.deformation_at(self.real_position)
            self.activated = True

        # stimuli only deforms silicon where the sensor is on
        else:
            border_deformation = stimuli.border_deformation()

            self.deformation = stimuli.def_func.get_z(distance, border_deformation)
            self.activated = False

    def get_circle_properties(self):
        base_color = np.array([0, 0, 0])
        base_color[2] = min(255, int(base_color[2] - self.deformation * UNIT * 10))
        base_color[1] = min(255, int(base_color[1] - self.deformation * UNIT * 5))

        if self.activated:
            return [base_color, self.frame_position, 6]
        else:
            return [base_color, self.frame_position, 3]
