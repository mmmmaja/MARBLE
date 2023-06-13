import numpy as np
import matplotlib.pyplot as plt
from pressure_recording_manager import *
import scipy



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

        for i, val in enumerate(sensor_values):

            if val >= threshold:
                activated_sensors.append(i)
                pressure_values.append(val)


        return activated_sensors, pressure_values


class CountDecider(ActivationDecider):
    """
    CountDecider returns only top N sensors with highest pressure values
    """

    def __init__(self, upper_n):
        """

        :param upper_n: the amount of top sensors that will be returned. For example if upper_n = 3, sensors with top 3 pressures will be returned
        """

        self.upper_n = upper_n

    def decide_activated(self, sensor_values):
        sensor_values = np.abs(sensor_values)

        activated_sensors = []
        pressure_values = []

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



def distance_distribution(sensor_positions,time_frames,decider,MAX_SEP,include_pressure = False):
    """

    :param sensor_positions: position of each sensor
    :param time_frames: all the time frames of the sensor recordings
    :param decider: the specified decider - for example CountDecider
    :param MAX_SEP: maximum separation of jointly activated sensors (determined by the outside stimuli size)
    :param include_pressure: option to return pressure values
    :return: returns distances of the pairs of activated sensors, total count of the activated sensors, count of pairs sensors below the MAX_SEP distance (optionally, pressure values)
    """
    activated_distances = []

    total_activated_cnt = 0

    below_max_sep_cnt = 0

    pressure_values = []

    for frame in time_frames:

        activated, pressures = decider.decide_activated(frame)

        total_activated_cnt += len(activated)
        for i in range(len(activated)):
            for j in range(i + 1, len(activated)):

                distance = np.linalg.norm(sensor_positions[activated[i]] - sensor_positions[activated[j]])
                if distance <= MAX_SEP: below_max_sep_cnt += 1
                activated_distances.append(distance)
                pressure_values.append(min(pressures[i],pressures[j]))

    if include_pressure:
        return activated_distances,pressure_values, total_activated_cnt, below_max_sep_cnt
    else:
        return activated_distances, total_activated_cnt, below_max_sep_cnt

def plot_pressure_expdistribution(activated_distances,percent_activated,percent_under_max_sep,MIN_SEP):
    """
    1.Plots the histogram of distances of the pairs of activated sensors
    2. Fits (using mle) exponential distribution to the distance hist
    :param activated_distances: distances of activated pairs
    :param percent_activated: percent of sensors activated
    :param percent_under_max_sep: percent of pairs that were activated, and their distances were under the MAX_SEP distance
    :param MIN_SEP: minimum separation of sensors
    :return:
    """

    mle_lambda = 1/(np.mean(activated_distances) - MIN_SEP)
    x_ = np.linspace(min(activated_distances) - MIN_SEP + 10e-3, max(activated_distances) - MIN_SEP, 100)

    y_ = scipy.stats.expon(scale=1 / mle_lambda).pdf(x_)
    for i in range(len(x_)): x_[i] += MIN_SEP

    plt.figure()
    plt.title(f"Pairwise Distance Hist for Threshold {t}-times the Mean; {percent_activated}% Activated")

    plt.hist(activated_distances, density=True, bins=20)
    plt.plot(x_, y_)
    plt.xlabel(f"Under Max separation: {round(percent_under_max_sep, 2)}%")
    plt.axvline(MIN_SEP, color='k', linestyle='dashed')
    plt.axvline(MAX_SEP, color='r', linestyle='dashed')

    plt.show()

def plot_pressure_logndistribution(activated_distances,percent_activated,percent_under_max_sep,MIN_SEP):
    """
    1.Plots the histogram of distances of the pairs of activated sensors
    2. Fits (using mle) lognormal distribution to the distance hist
    :param activated_distances: distances of activated pairs
    :param percent_activated: percent of sensors activated
    :param percent_under_max_sep: percent of pairs that were activated, and their distances were under the MAX_SEP distance
    :param MIN_SEP: minimum separation of sensors
    :return:
    """

    mle_mean = sum(np.log(activated_distances))/len(activated_distances)

    mle_std = 0
    for dist in activated_distances:
        mle_std += np.power(np.log(dist) - mle_mean,2)
    mle_std = np.sqrt(mle_std/len(activated_distances))

    print(f"mle mean: {mle_mean}, mle std: {mle_std}")

    x_ = np.linspace(10e-5, max(activated_distances), 100)

    y_ = scipy.stats.lognorm.pdf(x_,loc = mle_mean,s = mle_std)
    plt.figure()
    plt.title(f"Pairwise Distance Hist for Threshold {t}-times the Mean; {round(percent_activated,2)}% Activated")

    plt.hist(activated_distances, density=True, bins=20)
    plt.plot(x_, y_)
    plt.xlabel(f"Under Max separation: {round(percent_under_max_sep, 2)}%")
    plt.axvline(MIN_SEP, color='k', linestyle='dashed')
    plt.axvline(MAX_SEP, color='r', linestyle='dashed')

    plt.show()

def plot_3d_pressure_hist(activated_distances,pressure_values,percent_activated,percent_under_max_sep):
    """
    plot histogram of distances of activated sensors with respect to the pressure value of the pairs
    the distances of will be split into multiple bin sections according to the pressure at which they were activated
    :param activated_distances: distances of activated pairs
    :param pressure_values: pressure values for each pair
    :param percent_activated: percent of sensors activated
    :param percent_under_max_sep:percent of pairs that were activated, and their distances were under the MAX_SEP distance
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    hist, xedges, yedges = np.histogram2d(activated_distances, pressure_values, bins=[12,5])

    # for i in range(hist.shape[1]):
    #     hist[:,i] = hist[:,i]/sum(hist[:,i])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 64 bars.
    dx = dy = 0.2 * np.ones_like(zpos)
    dz = hist.ravel()

    plt.title(f"Pairwise Distance Hist for Threshold {t}-times the Mean; {round(percent_activated,2)}% Activated")
    plt.xlabel(f"Distance [Under Max separation : {round(percent_under_max_sep, 2)}]%")
    plt.ylabel(f"Pressure value")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    plt.show()



if __name__ == "__main__":

    MIN_SEP = 1
    MAX_SEP = 2

    sensor_positions, time_frames = read_recording("../pygame_model/data.csv")

    # We can plot the distance distributions of pairs of activated sensors for different ActivationDeciders
    # From this we can see, which decider is the best for determining relative positions, and additionally, we can use the mle distributions
    # to probabilistically estimate distance of two sensors when we have a new recording
    for t in range(2,20):

        t = t/2

        decider = MeanThresholdDecider(threshold=t)
        #decider = CountDecider(int(t) + 2)

        activated_distances,pressure_values, total_activated_cnt, below_max_sep_cnt = distance_distribution(sensor_positions,time_frames,decider,MAX_SEP,include_pressure=True)


        percent_activated = 100*total_activated_cnt/(len(time_frames)*len(time_frames[0]))
        percent_below_max_sep = 100*below_max_sep_cnt/(len(activated_distances))

        print(max(pressure_values))
        #plot_pressure_expdistribution(activated_distances,percent_activated,percent_below_max_sep,MIN_SEP)
        #plot_3d_pressure_hist(activated_distances,pressure_values,percent_activated,percent_below_max_sep)
        plot_pressure_logndistribution(activated_distances,percent_activated,percent_below_max_sep,MIN_SEP)