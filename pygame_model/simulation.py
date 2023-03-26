import random
from _csv import reader

mouse_presses = []

# update every 100 milliseconds
UPDATE_INTERVAL = 100


class ForgeRecording:

    def __init__(self, frame_dim, stimuli, sensor_mesh, duration=10):
        self.frame_dim = frame_dim
        self.stimuli = stimuli
        self.sensor_mesh = sensor_mesh
        self.duration = duration

        self.OUTPUT_PATH = 'fake_data'

        self.simulation_loop()

    def simulation_loop(self):
        self.stimuli.set_position(
            [random.randint(0, self.frame_dim[0]), random.randint(0, self.frame_dim[1])]
        )

        timer = 0
        while timer < self.duration * 1000:
            self.stimuli.set_deformation(-2 * 40)
            # Change pressure outputs of the sensors
            self.sensor_mesh.press(self.stimuli)
            self.simulate_press()

            self.sensor_mesh.append_data()
            timer += UPDATE_INTERVAL
            print(timer)

        file_name = '1'
        self.sensor_mesh.save_data(path=self.OUTPUT_PATH + '/' + file_name)

    def simulate_press(self):
        dx, dy = 1, 1
        self.stimuli.set_position(
            [random.randint(0, self.frame_dim[0]), random.randint(0, self.frame_dim[1])]
        )
        # self.stimuli.set_position([
        #     self.stimuli.position[0] + dx,
        #     self.stimuli.position[0] + dy
        # ])


class ReadRecording:

    def __init__(self):
        self.file_path = 'fake_data/1'
        self.data = self.read_data()
        self.time_index = 0

    def read_data(self):
        data = []
        with open(self.file_path, 'r', newline='') as file:
            csv_reader = reader(file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    line_count += 1
                    data.append(row)
        return data

    def read(self, sensor_mesh):
        for i in range(len(sensor_mesh.SENSOR_ARRAY)):
            sensor_mesh.SENSOR_ARRAY[i].deformation = float(self.data[self.time_index][i]) * 40
        self.time_index += 1
        print(self.time_index)
        if self.time_index >= len(self.data):
            return False
        return True

