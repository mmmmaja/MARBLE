from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def read_csv(path):
    index = 0
    sensor_positions, sensor_readings = [], []
    with open(path, 'r', newline='') as file:
        for row in file.readlines():
            # First row contains position of the sensors
            if index == 0:
                sensor_positions = row.split(',')
                for i in range(len(sensor_positions)):
                    sensor_positions[i] = sensor_positions[i].split(' ')
                    sensor_positions[i] = [float(x) for x in sensor_positions[i]]
                print(sensor_positions)
            # The rest of the rows contain the sensor readings
            else:
                str_row = row.split(',')
                float_row = [float(x) for x in str_row]
                sensor_readings.append(float_row)
            index += 1
    print(sensor_positions)
    return sensor_positions, sensor_readings


class ReadingManager:

    def __init__(self, path):
        self.sensor_positions, self.sensor_reading = read_csv(path)

    def visualize(self, sleep_time=1):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # If x is clicked, the plot will close
        fig.canvas.mpl_connect('close_event', exit)

        min_z = min([min(x) for x in self.sensor_reading])
        max_z = max([max(x) for x in self.sensor_reading])

        for i in range(len(self.sensor_reading)):
            ax.clear()  # Clear previous plot

            ax.set_zlim([min_z, max_z])
            # Plot the sensor positions
            # Extract the sensor readings and positions
            x_values = [float(x[0]) for x in self.sensor_positions]
            y_values = [float(y[1]) for y in self.sensor_positions]
            z_values = [float(z) for z in self.sensor_reading[i]]

            # Add text to the plot
            ax.text2D(0.05, 0.95, f'Frame {i}', transform=ax.transAxes)

            # Plot the sensor positions
            ax.scatter(
                x_values,
                y_values,
                z_values
            )
            plt.draw()
            plt.pause(sleep_time)

        plt.show()


reader = ReadingManager('../recordings/recording.csv')
reader.visualize()
