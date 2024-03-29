from data_classification.data_analysis import *
import os
from scipy.stats import *
import time


##TODO: Will be deleted, new version in sensor_filter.py

def normal_filter_sample_sensors_global(sample, alpha=0.02):
    """
    Assumes normal distribution for time step
    Indicates which sensors throughout the whole time sequence that do not fit estimated model
    """

    # Sensor is filtered for all time steps
    p_avg, rot_avg = sample.sensor_wise_avg()

    mean = np.mean(p_avg)
    st_dev = np.std(p_avg)

    if st_dev < 0.0001:
        print(sample.label)
        print("MEANS: ", p_avg)
        print("ST DEV: ", st_dev)
        return []

    standard_norm_avg = (p_avg - mean) / st_dev
    bound = norm.ppf(1 - alpha / 2)
    # Assuming we are in normal distribution
    filtered_sensors = []
    for i, ns in enumerate(standard_norm_avg):
        # if value falls out of acceptance region,
        # add fabricated value form the distribution, we assume the sensor values are
        if not (-bound <= ns <= bound):
            filtered_sensors.append(i)

    return filtered_sensors


# When sensor is filtered, it is only for a given time step
def normal_filter_sample_sensors_local(sample: SampleDataAnalysis, alpha=0.02):
    """
    Separately for each time step
    Assumes normal distribution for time step
    Indicated which sensors do not fit estimated model
    """
    filtered_sensors = []
    for i in range(sample.num_time_steps):

        filtered_sensors_at = []

        p_values = sample.get_p_values_at_time(i)
        mean = np.mean(p_values)
        st_dev = np.std(p_values)

        standard_norm_p_values = (p_values - mean) / st_dev
        bound = norm.ppf(1 - alpha / 2)

        for j, ns in enumerate(standard_norm_p_values):
            if not (-bound <= ns <= bound):
                filtered_sensors_at.append(j)

        filtered_sensors.append(filtered_sensors_at)

    return filtered_sensors


def replace_filtered_sensors_normal_global(sample: SampleDataAnalysis, filtered_sensors, inplace=True):
    # Replaces sensors that ware filtered throughout the whole time sequence
    p_avg, rot_avg = sample.sensor_wise_avg()
    mean = np.mean(p_avg)
    st_dev = np.std(p_avg)

    if not inplace:
        sample = sample.copy()

    for f_sensor in filtered_sensors:
        for time_step in range(sample.num_time_steps):
            sample.p_sensors[time_step][f_sensor] = np.random.normal(mean, st_dev * 1)

    return sample


def replace_filtered_sensors_normal_local(sample: SampleDataAnalysis, filtered_sensors, inplace=True):
    """
    Replaces sensors that we filtered for each time stepnseparately
    """
    if not inplace:
        sample = sample.copy()

    for i, filtered_sensors_at in enumerate(filtered_sensors):
        p_sensors = sample.get_p_values_at_time(i)
        mean = np.mean(p_sensors)
        st_dev = np.std(p_sensors)

        for f_sensor in filtered_sensors_at:
            sample.p_sensors[i][f_sensor] = np.random.normal(mean, st_dev)

    return sample


## code section for actually filtering the faulty sensors and replacing them with placeholder values
if __name__ == "__main__":

    data_folder = "C:/University/Marble/Data/"
    data_dir = os.listdir(data_folder)

    for index, labeled_data in enumerate(data_dir):
        if index > 100:
            continue
        folder = data_folder + labeled_data + "/"

        label_dir = os.listdir(folder)

        if "90" in labeled_data or "45" in labeled_data:
            x = 1
        else:
            continue

        if labeled_data != "incorrect_orthosis_up_1cm_90": continue

        for sample_ in label_dir:
            print(sample_)
            if "LIN" in sample_: continue

            sample_path = folder + sample_
            sample = SampleDataAnalysis(labeled_data + "_" + sample_, file_path=sample_path)
            fs = normal_filter_sample_sensors_local(sample)
            print(fs)
            print("\n\n")

            time.sleep(5)
