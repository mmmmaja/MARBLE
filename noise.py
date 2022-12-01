import matplotlib.pyplot as plt
import numpy as np

from data_analysis import *
import os
from scipy.stats import *




def filter_sample(sample):


    p_avg, rot_avg = sample.sensorwise_avg()

    mean = sum(p_avg)/len(p_avg)
    var = 0
    for avg in p_avg:
        var += (mean - avg)**2 / len(p_avg)

    print(mean)
    print(var)
    stdev = var**(1/2)

    norm_avg = (p_avg - mean) / stdev

    alpha = 0.99
    bound = norm.ppf(alpha)
    ## assuming we are in normal distribution
    filtered_avg = np.zeros(len(p_avg))
    filtered_sensors = []
    for i in range(len(p_avg)):

        if -bound <= norm_avg[i] <= bound:
            filtered_avg[i] = p_avg[i]
        else:
            ## if value falls out of acceptance region, add fabricated value form the distribution, we assume the sensor values are
            filtered_sensors.append(i)
            filtered_avg[i] = np.random.normal(mean,stdev)

    print("filtered sensors: ", filtered_sensors)

    filtered_mean = sum(filtered_avg)/len(filtered_avg)
    filtered_var = 0
    for avg in filtered_avg: filtered_var += (avg - filtered_mean)**2 / len(filtered_avg)
    filtered_stdev = filtered_var**(1/2)

    x_spc = np.linspace(min(p_avg),max(p_avg),200)

    plt.plot(x_spc,norm.pdf(x_spc,mean,stdev))
    plt.hist(p_avg,20,density=True)
    plt.hist(filtered_avg,20,density=True)
    plt.plot(x_spc, norm.pdf(x_spc, filtered_mean, filtered_stdev))
    plt.title(sample.label)


    plt.show()
    plt.close()

    print(sample.label)


data_folder = "C:/University/Marble/Data/"
data_dir = os.listdir(data_folder)


for index, labeled_data in enumerate(data_dir):
    if index > 3: continue
    folder = data_folder + labeled_data + "/"

    label_dir = os.listdir(folder)

    if "90" in labeled_data or "45" in labeled_data: x = 1
    else: continue

    for sample_ in label_dir:

        if "LIN" in sample_: continue

        sample_path = folder + sample_
        sample = SampleDataAnalysis(labeled_data +"_"+ sample_,file_path=sample_path)
        filter_sample(sample)






