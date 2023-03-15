import numpy as np
import matplotlib.pyplot as plt
from data_analysis import SampleDataAnalysis


LV_MIN = 10e-5
LV_MAX = 1
NV_MIN = 0.1
NV_MAX = 10
REC_LIM = 5



def estimate_variance_bounds(sensors):



    variances = np.zeros(len(sensors))
    for i, sensor in enumerate(sensors):
        variances[i] = np.var(sensor)


    return np.min(variances), np.max(variances)

def sensor_fit(sensor_pressures, min_v, max_v, linearized = False, n = 1):



    mean = np.mean(sensor_pressures)
    var = np.var(sensor_pressures)
    print("V: {}".format(var))
    faulty = 0

    ## if the variance is too large for the sensors to behave only via 1 distribution, probably the sensors was working and then stopped
    if var >= max_v:
        # in case there is too much recursion depth, end it for the sake of speed, assume all the sensors are faulty in this case (since varianace is big)
        if n >= REC_LIM:
            return mean,1

        sensor_pressures = np.sort(sensor_pressures)
        i = 0
        ## after sorting array splut it (using mean) - intuition is that probably the pressures above mean are the faulty recordings, and the ones below mean are correct
        while sensor_pressures[i] <= mean: i += 1

        ## determine the fraction that is below mean
        theta = (i + 1)/len(sensor_pressures)

        ## determine the MLE parameters for splitted dat [section 1, and 2]
        m1, f1 = sensor_fit(sensor_pressures[:i],min_v, max_v, linearized=linearized, n=n+1)
        m2, f2 = sensor_fit(sensor_pressures[i:],min_v, max_v, linearized=linearized, n=n+1) ##  NOT WORKING

        ## make weighted average of the mean pressure- example: m1 = 10.4, f1 = 0, m2 = 100, f2 = 1
        ## since section 2 had all sensors faulty, we do not consider the mean of that section (1-f2 = 0), section 1 on the other hand
        ## had all sensors correct thus we consider all of it in determining the mean of the data set
        ## This way, the mean of the dataset is determined ONLY using the pressure recordings that happened while the sensors itself was NOT faulty
        ## and will exclude the recordings where the sensors WAS faulty

        ## Furthermore, the ratio of amount of time when a sensor is faulty is returned
        ## Example: if the section 2 was faulty f2 = 100% of the time and section 1 f1 = 0%, and section 2 consists of 40% of all recordings
        ## then theta = 0.6; and the returned ratio is 0*0.6 +1*0.4 = 0.4, so 40% of the time sensor is faulty
        total_weight = (1-f1) + (1-f2)
        return ((1- f1)*m1 + (1-f2)*m2)/total_weight, f1*theta + f2*(1-theta)

    elif var <= min_v:
        faulty = 1
        mean = 0 ## all pressures are faulty, meaning no pressures tell us anything about the systematic noise shift when the sensors is working (since the sensor is never working)


    return mean, faulty



def construct_filter(sensors,min_v, max_v, linearized = False):


    if len(sensors[0]) < 5:
        print("[WARNING] Not enough recording per sensors to derive meaningful ")


    sensors_filter = np.zeros((len(sensors),2))
    for i, sensor in enumerate(sensors):
        shift, faulty_rate = sensor_fit(sensor,min_v, max_v, linearized = linearized)
        sensors_filter[i,0] = shift
        sensors_filter[i,1] = faulty_rate

        print("Shift: {}, FRate: {}".format(shift,faulty_rate))

    return sensors_filter

## Replace and correct sensors based on the determined systematic noise and faultyness of each sensor
## WARNING: [sensor_mark] of corrected sensors MUST correspond to the snenros in [recording] array
def rectify_recording(sensors_mark, recording, faulty_tolerance = 0.9, inplace= True):
    if not inplace:
        recording = np.copy(recording)

    for t, sensors in enumerate(recording):

        mean = np.mean(sensors)
        std = np.std(sensors)


        for i in range(len(sensors)):
            sensors[i] -= sensors_mark[i][0]

            # if more than [faulty_tolerance] of time the sensor is faulty, we replace it by placeholder value
            if sensors_mark[i][1] > faulty_tolerance:
                sensors[i] = np.random.normal(mean,std)

    return recording


## DEMO
if __name__ == "__main__":

    sample = SampleDataAnalysis("No orthosis 0",file_path="C:/University/Marble/Data/no_orthosis_0/3_RAW.xlsx")
    sensors = sample.get_p_array()

    ## Find some bounds on the pressure variances
    min_v, max_v = estimate_variance_bounds(sensors)

    print("Min Var: {}, Max Var: {}".format(min_v,max_v))

    ## filter sensors using these paramters; the coeffs 0.5 and 0.9 are chosen arbitrarily
    sensors_filter = construct_filter(sensors,min_v*0.5,max_v*0.9)


    print(sensors_filter)