import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata


def read_csv(path):
    """
    Read the data from the csv file and save it to the class variables
    :param path: path to the csv file with sensor recordings
    :return: sensor_positions, sensor_readings lists
    """
    index = 0
    sensor_positions, sensor_readings = [], []
    with open(path, 'r', newline='') as file:
        for row in file.readlines():
            # First row contains position of the sensors
            if index == 0:
                sensor_positions = row.split(',')
                # Convert from string to float
                for i in range(len(sensor_positions)):
                    sensor_positions[i] = sensor_positions[i].split(' ')
                    sensor_positions[i] = [float(x) for x in sensor_positions[i]]
            # The rest of the rows contain the sensor readings
            else:
                str_row = row.split(',')
                # Convert from string to float
                float_row = [float(x) for x in str_row]
                sensor_readings.append(float_row)
            index += 1
    return sensor_positions, sensor_readings


class ReadingManager:

    def __init__(self, path):
        """
        Visualizes the recording from a csv file
        :param path: path to the csv file with sensor recordings
        """
        self.sensor_positions, self.sensor_reading = read_csv(path)

    def visualize(self, sleep_time=0.4):
        """
        Visualize the sensor readings with the 3D surface plot
        :param sleep_time: time to sleep between each frame
        """

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # If x is clicked, the plot and app will close
        fig.canvas.mpl_connect('close_event', exit)

        # Get min and max values of Z to set consistent color scale and z-axis limits
        min_z = min([min(x) for x in self.sensor_reading])
        max_z = max([max(x) for x in self.sensor_reading])

        # Set color normalization based on overall min and max z-values
        norm = colors.Normalize(vmin=min_z, vmax=max_z)

        # Extract sensor positions
        x_values = np.array([float(x[0]) for x in self.sensor_positions])
        y_values = np.array([float(y[1]) for y in self.sensor_positions])

        # Create a meshgrid for X and Y
        xi = np.linspace(min(x_values), max(x_values), 100)
        yi = np.linspace(min(y_values), max(y_values), 100)
        X, Y = np.meshgrid(xi, yi)

        for i in range(len(self.sensor_reading)):
            # Clear previous plot
            ax.clear()

            # Set consistent z-axis limits
            ax.set_zlim([min_z, max_z])

            # Extract the sensor readings
            z_values = np.array([float(z) for z in self.sensor_reading[i]])
            # Interpolate Z values over grid
            Z = griddata((x_values, y_values), z_values, (X, Y), method='cubic')

            # Plot the surface
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma', edgecolor='none', norm=norm)

            # Add text to the plot with the current iteration
            ax.text2D(0.05, 0.95, f'Frame {i}', transform=ax.transAxes)

            plt.draw()
            plt.pause(sleep_time)

        plt.show()


reader = ReadingManager('../recordings/recording.csv')
reader.visualize()
