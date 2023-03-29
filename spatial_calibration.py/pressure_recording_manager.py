import numpy as np


def read_recording(recording_file):


    file = open(recording_file)


    time_frames = []

    sensor_positions = []

    line = file.readline().strip("\n").split('","')
    for p in line:
        sensor_positions.append(np.array(list(map(float, p.strip('"').split(",")))))



    for line in file.readlines():
        time_frames.append(np.array( list(map(float,line.strip("\n").split(",")))) )


    return np.array(sensor_positions), time_frames


def mean_deformation(time_frames, per_n = 1):

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
    if len(time_frames) == 0: return None

    pressures = np.zeros(len(time_frames[0]))

    for t in time_frames:
        for i,s in enumerate(t):
            pressures[i] += s

    pressures *= mul

    return pressures


if __name__ == "__main__":

    sensor_pos, time_frames = read_recording("../pygame_model/data.csv")

    print(total_pressures(time_frames))
