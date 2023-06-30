import numpy as np

## Pressure recording manager servers for reading, filtering recordings and sensors on various plots



def compare_location_estimates(l1,l2):
    """
    compares one list of locations of sensors with another list of sensor locations
    :param l1: first list of sensor locations
    :param l2: second list of sensor locations
    :return: returns summed euclidian distance of all sensor locations (computing the distance of the location of the sensor in list 1 and list 2)
    """
    if len(l1) != len(l2):
        raise "Both arrays of sensor locations must be the same size"

    absolute_difference = 0

    for i in range(len(l1)):
        absolute_difference += np.linalg.norm(l1[i] - l2[i])

    print(f"abs diff: {absolute_difference}, mean diff: {absolute_difference/len(l1)}")

    return absolute_difference


def filter_sensors(to_filter, sensor_locations):
    """
    Filters sensors locations with specific ids
    :param to_filter: ids of snesors to be filtered
    :param sensor_locations: all sensor locations
    :return: returns the sensors locations after filtering out the to_filer sensors
    """

    sp = []
    ids = []
    for i, p in enumerate(sensor_locations):
        if i in to_filter: continue

        sp.append(p)
        ids.append(i)

    return ids, np.array(sp)

def get_positions_of_sensors(sensor_locations,ids):
    """

    :param sensor_locations: all sensor locations
    :param ids: ids of certain sensors
    :return: returns a list of sensor locations of the sensors with chosen ids
    """

    p = []
    for i, p_ in enumerate(sensor_locations):
        if i in ids:
            p.append(p_)

    return p

def read_labeled_recording(recording_file):
    """
    Parses a recording into recirding frame arrays and locations of sensors. 1st Line includes true locations of all sensors
    :param recording_file: string of the location of the recording file
    :return: true location of sensors, pressure recordings of sensors (can be multiple frames)
    """

    file = open(recording_file)

    time_frames = []

    sensor_locations = []

    line = file.readline().strip("\n").split('","')
    for p in line:
        sensor_locations.append(np.array(list(map(float, p.strip('"').split(",")))))


    for line in file.readlines():
        time_frames.append(np.abs(np.array( list(map(float,line.strip("\n").split(","))))) )


    return np.array(sensor_locations), time_frames


def read_unlabeled_recording(recording_file):
    """
      parses a recording into recirding frame arrays and locations of sensors
      :param recording_file: string of the location of the recording file
      :return: true location of sensors, pressure recordings of sensors (can be multiple frames)
      """

    file = open(recording_file)

    time_frames = []

    for line in file.readlines():
        time_frames.append(np.array(list(map(float, line.strip("\n").split(",")))))

    return time_frames


def mean_deformation(time_frames, per_n = 1):
    """
    coputes mean deformation of all nth frames
    :param time_frames: list of time frames
    :param per_n: if for example 3, then every third frame is accounted in the mean
    :return: returns absolute mean of all frames, and list of means for all frames
    """

    absolute_mean = 0

    means = []

    for i, frame in enumerate(time_frames):

        if i % per_n == 0:
            mean = np.mean(frame)
            means.append(mean)
            print(f"Frame {i}: {mean}")
            absolute_mean += mean

    print(f"Absolute mean: {absolute_mean/len(time_frames)}")
    return absolute_mean, means

def min_deformation(time_frames, per_n = 1):
    """
    finds the minimum deformation through out frames
    :param time_frames: list of lists, where each list has the pressure values of all sensors
    :param per_n: if for example 3, then every third frame is accounted in the mean
    :return: minimum neformation through out frames, and a list of minimum deformations for all frames separately
    """
    absolute_min = 0

    mins = []

    for i, frame in enumerate(time_frames):

        if i % per_n == 0:
            min_ = np.max(frame)
            mins.append(min_)
            print(f"Frame {i} min deform: {min_}")
            absolute_min = max(min_,absolute_min)

    print(f"Absolute min deform: {absolute_min}")
    return absolute_min, mins


def max_deformation(time_frames,per_n = 1):
    """
    finds the maximum deformation through out frames
    :param time_frames: list of lists, where each list has the pressure values of all sensors
    :param per_n: if for example 3, then every third frame is accounted in the mean
    :return: maximum neformation through out frames, and a list of minimum deformations for all frames separately
    """
    absolute_max = 0

    maxs = []

    for i, frame in enumerate(time_frames):

        if i % per_n == 0:
            max_ = np.min(frame)
            maxs.append(max_)
            print(f"Frame {i} max deform: {max_}")
            absolute_max = min(max_,absolute_max)

    print(f"Absolute max deform: {absolute_max}")
    return absolute_max, maxs



def total_pressures(time_frames, mul = 1):
    """
    returns a list of the length of the recording (len(time_frames)), where each entry is the total pressure recorded that time frame
    :param time_frames: list of lists where each list is a recording of pressures
    :param mul: multiplier of the pressure values
    :return:
    """
    if len(time_frames) == 0: return None

    pressures = np.zeros(len(time_frames[0]))

    for t in time_frames:
        for i,s in enumerate(t):
            pressures[i] += s

    pressures *= mul

    return pressures


if __name__ == "__main__":

    sensor_pos, time_frames = read_labeled_recording("../pygame_model/data_2sz.csv")

    print(total_pressures(time_frames))
