import random
from _csv import reader
from mesh import UNIT

mouse_presses = []


class ForgeRecording:

    def __init__(self, frame_dim, stimuli, sensor_mesh, update_interval, duration=10):
        # Dimension of the frame to restrain mouse presses position
        self.frame_dim = frame_dim
        # Stimuli used in the current simulation
        self.stimuli = stimuli
        self.sensor_mesh = sensor_mesh
        # Number of milliseconds between frame updates
        self.update_interval = update_interval
        # Duration of the recording in seconds
        self.duration = duration

        # Path to the folder where the data is to be saved
        self.OUTPUT_PATH = 'fake_data'

        # Start outputting fake data
        self.simulation_loop()

    def simulation_loop(self):
        # Start with the random stimuli position
        self.stimuli.set_frame_position([
            random.randint(0, self.frame_dim[0]),
            random.randint(0, self.frame_dim[1]),
            0])
        # Keeps track on which millisecond of the recording are we on
        timer = 0
        while timer < self.duration * 1000:
            self.stimuli.set_deformation(-2)
            # Change pressure outputs of the sensors
            self.sensor_mesh.press(self.stimuli)

            self.simulate_press()
            self.sensor_mesh.append_data()

            timer += self.update_interval
            print(timer)

        file_name = '1'
        self.sensor_mesh.save_data(path=self.OUTPUT_PATH + '/' + file_name)

    def simulate_press(self):
        # Number of centimeters per second
        SPEED = 0.8

        # Displacement in this iteration is dependent on the update interval
        local_displacement = (self.update_interval * SPEED) / 1000
        # print(local_displacement)

        position = [random.randint(0, self.frame_dim[0]), random.randint(0, self.frame_dim[1]), 0]
        position = [
            self.stimuli.position[0] + local_displacement,
            self.stimuli.position[1] + local_displacement,
            0]
        # self.stimuli.set_frame_position(position)

        self.stimuli.set_position(position)
        print(position)


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
            sensor_mesh.SENSOR_ARRAY[i].deformation = float(self.data[self.time_index][i])
        self.time_index += 1
        if self.time_index >= len(self.data):
            return False
        return True

