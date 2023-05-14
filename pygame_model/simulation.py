import random
import statistics
from _csv import reader

import matplotlib

from mesh import UNIT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx

DATA_PATH = 'fake_data'

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

        file_name = '1.csv'
        self.sensor_mesh.save_data(path=DATA_PATH + '/' + file_name)

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

        self.stimuli.set_position(position)


def read_data(file_path):
    pressure_data, position = [], None
    with open(file_path, 'r', newline='') as file:
        csv_reader = reader(file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                position = row
                line_count += 1
            else:
                line_count += 1
                pressure_data.append(row)
    return pressure_data, position


class ReadRecording:

    def __init__(self):
        self.file_path = 'fake_data/1.csv'
        self.data, _ = read_data(self.file_path)
        self.time_index = 0

    def read(self, sensor_mesh):
        for i in range(len(sensor_mesh.SENSOR_ARRAY)):
            sensor_mesh.SENSOR_ARRAY[i].deformation = float(self.data[self.time_index][i])
        self.time_index += 1
        if self.time_index >= len(self.data):
            return False
        return True


def avg_pressure(data):
    sensor_num = data.shape[1]
    time_frame_num = data.shape[0]

    avg_pressures = np.zeros(sensor_num)
    for s in range(sensor_num):
        for t in range(time_frame_num):
            avg_pressures[s] += float(data[t, s])

    return avg_pressures / time_frame_num


def plot(pressure_distribution, position):
    x, y = [], []
    for p in position:
        p = p.split(',')
        x.append(float(p[0]))
        y.append(float(p[1]))

    z = pressure_distribution

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Average pressure for each sensor throughout entire recording')

    fitness = recording_fitness(pressure_distribution)
    # adding text inside the plot
    s = 'Fitness: ' + str(round(fitness, 2))
    ax.text2D(0.02, 0.9, s, bbox=dict(facecolor='cyan', alpha=0.5), transform=ax.transAxes)

    # Let the rgb colours indicate the pressure of the sensor
    cm = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # Plot the sensor points
    ax.scatter(x, y, z, c=scalarMap.to_rgba(z))
    scalarMap.set_array(z)

    # Show plot
    plt.show()


def recording_fitness(data):

    # Threshold to consider sensor not enough activated
    threshold = max(statistics.mean(data) / 4, 0.2)

    penalties = []
    for sensor_avg_pressure in data:

        if sensor_avg_pressure < threshold:
            penalty = (threshold - sensor_avg_pressure) / threshold
        else:
            penalty = 0
        penalties.append(penalty)

    # The lower value of fitness the better recording
    return sum(penalties) / len(data)


def evaluate_recording(path=DATA_PATH + '/data.csv'):
    """
    :param path: path to the recording in .csv file to be evaluated
    :return: Evaluation of the recording <0,1>
    """
    print("Your data is about to be evaluated")

    pressure_data, position = read_data(path)

    # Returns average pressure for each sensor
    pressure_distribution = avg_pressure(np.array(pressure_data))

    # Plot this distribution
    plot(pressure_distribution * (-1), position)




