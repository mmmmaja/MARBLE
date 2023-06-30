import numpy as np
from utilities import *


## Decider Classes - classes that are used for determining which sensors will be considered activated


class ActivationDecider:
    """
    ActivationDecider is a class that servers as an abstract for specific classes that determine what sensors
    were activated in a given timeframe.
    The class is very important, since it determines which sensors are activated, which is then used to estimate their
    mutual proximities

    """

    def decide_activated(self, sensor_values):
        """

        :param sensor_values: list of sensor pressure negative values, where index corresponds to sensor id
        :return: returns activated sensor ids and their pressures
        """
        pass



class MeanThresholdDecider(ActivationDecider):
    """
    Class that returns all sensors as activated whose pressure values were above K*mean
    K is pre-specified
    """


    def __init__(self, threshold):
        """

        :param threshold: decider returns all the sensors above the threshold value of threshold*mean. Mean is readily computed from the list of sensor pressures
        """
        self.t = threshold

    def decide_activated(self, sensor_values):
        sensor_values = np.abs(sensor_values)

        mean = np.mean(sensor_values)
        threshold = mean*self.t

        activated_sensors = []
        pressure_values = []

        if mean == 0: return activated_sensors, pressure_values

        for i, val in enumerate(sensor_values):

            if val >= threshold:
                activated_sensors.append(i)
                pressure_values.append(val)


        return activated_sensors, pressure_values


class CountDecider(ActivationDecider):
    """
    CountDecider returns only top N sensors with highest pressure values
    """

    def __init__(self, upper_n = None,min_sep = None,max_sep = None):
        """

        :param upper_n: the amount of top sensors that will be returned. For example if upper_n = 3, sensors with top 3 pressures will be returned
        """

        if upper_n is None:
            upper_n = int(np.ceil(np.power(max_sep/min_sep,2)))

        self.upper_n = upper_n

    def decide_activated(self, sensor_values):
        sensor_values = np.abs(sensor_values)

        activated_sensors = []
        pressure_values = []


        if np.mean(sensor_values) == 0: return activated_sensors, pressure_values

        for i in range(self.upper_n):
            m = 0
            mi = 0
            for ix, v in enumerate(sensor_values):
                if sensor_values[ix] > m:
                    m = sensor_values[ix]
                    mi = ix
            sensor_values[mi] = 0
            activated_sensors.append(mi)
            pressure_values.append(m)

        return activated_sensors, pressure_values


