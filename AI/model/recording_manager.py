import csv
from datetime import datetime
from PyQt5.QtCore import QTimer


class Recording:

    FOLDER_PATH = '../recordings'

    def __init__(self, sensors, dt=100, file_name=None):
        """
        :param sensors: Sensors object with list of sensors
        :param dt: time step in milliseconds
        :param file_name: Name of the file to save the data to (if None, then the current date and time is used)
        """

        self.sensors = sensors
        self.dt = dt
        self.file_name = file_name

        self.sensor_data = []
        self.timer = None

    def start(self):
        print("Recording Started...")
        self.timer = QTimer()
        self.timer.timeout.connect(self.record)
        self.timer.start(self.dt)  # period of dt milliseconds

    def record(self):
        self.sensor_data.append([sensor.stress for sensor in self.sensors.sensor_list])

    def stop(self):
        print("Recording Stopped...")
        self.save_data()
        self.timer.stop()
        self.timer.deleteLater()

    def save_data(self):
        """
        Save the recorded data to a .csv file

        Form of the csv file:
            First line: sensor positions  -> x y | x y | x y | ...
            Rest of the file: sensor data -> z | z | z | ...
        """

        if self.file_name is None:
            # Get current date and time
            file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
        else:
            file_name = self.file_name

        print("Saving data to: " + file_name)

        # Save the data to a .csv file
        # The first line of the file should be the sensor positions
        sensor_positions = []
        for i in range(len(self.sensors.sensor_list)):
            position = self.sensors.sensor_list[i].initial_position
            sensor_positions.append(
                str(position[0]) + ' ' + str(position[1])
            )
        # The rest of the file should be the pressure data
        path = self.FOLDER_PATH + '/' + file_name
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(sensor_positions)
            writer.writerows(self.sensor_data)


