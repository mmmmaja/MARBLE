import math
import os
from enum import Enum
import numpy as np
import openpyxl
from matplotlib import pyplot as plt

from plotter import Plot


class Label(Enum):

    NO_ORTHOSIS_0 = 0
    NO_ORTHOSIS_45 = 1
    NO_ORTHOSIS_90 = 2
    CORRECT_ORTHOSIS_0 = 3
    CORRECT_ORTHOSIS_45 = 4
    CORRECT_ORTHOSIS_90 = 5
    UP_ORTHOSIS_0 = 6
    UP_ORTHOSIS_45 = 7
    UP_ORTHOSIS_90 = 8
    DOWN_ORTHOSIS_0 = 9
    DOWN_ORTHOSIS_45 = 10
    DOWN_ORTHOSIS_90 = 11
    CLOCK_ROT_ORTHOSIS_0 = 12
    CLOCK_ROT_ORTHOSIS_45 = 13
    CLOCK_ROT_ORTHOSIS_90 = 14
    COUTNERCLOCK_ROT_ORTHOSIS_0 = 15
    COUTNERCLOCK_ROT_ORTHOSIS_45 = 16
    COUTNERCLOCK_ROT_ORTHOSIS_90 = 16



class LabelDataAnalysis:


    def __init__(self, data_folder, label):

        self.data_folder = data_folder
        self.dir = os.listdir(self.data_folder)
        self.label = label

    def all_sensor_values(self, linearized = True):


        all_sensors = np.array([])
        for file in self.dir:
            if (linearized and "LIN" not in file) or (not linearized and "RAW" not in file): continue


            path = self.data_folder + "/" + file
            sample = SampleDataAnalysis(self.label, file_path=path)
            all_sensors = np.concatenate([sample.p_sensors.flatten(), all_sensors])

        return all_sensors



    def mean_variance(self,at_arm_degrees = None,time_wise = False, linearized = True):


        p_sensor_values = []
        rot_sensor_values = []
        nms = []
        for file in self.dir:
            if (linearized and "LIN" not in file) or (not linearized and "RAW" not in file): continue
            nms.append(file)
            path = self.data_folder + "/" + file
            sample = SampleDataAnalysis(self.label,file_path=path)

            p_sensors = None
            rot_sensors = None
            if at_arm_degrees is not None:
                p_sensors, rot_sensors, time_stamp = sample.sensor_values_at_arm_degrees(at_arm_degrees)
            elif time_wise is False:
                p_sensors, rot_sensors = sample.sensorwise_avg()
            elif time_wise is True:
                p_sensors, rot_sensors = sample.timewise_avg(sensor_range=[70,sample.num_p_sensors])

            p_sensor_values.append(p_sensors)
            rot_sensor_values.append(rot_sensors)


        ## MEAN
        p_sensor_avg = np.zeros(len(p_sensor_values[0]))
        i = 0
        for sensor_values in p_sensor_values:
            #print(nms[i])
            i += 1
            p_sensor_avg += sensor_values

        p_sensor_avg = p_sensor_avg/len(p_sensor_values)

        rot_sensor_avg = np.zeros(len(rot_sensor_values[0]))
        for sensor_values in  rot_sensor_values:
            rot_sensor_avg += sensor_values
        rot_sensor_avg = rot_sensor_avg/len(rot_sensor_values)
        ## Variance

        p_sensor_var = np.zeros(len(p_sensor_values[0]))
        for sensor_values in p_sensor_values:
            p_sensor_var += (p_sensor_avg - sensor_values)**2

        p_sensor_var = p_sensor_var/len(p_sensor_values)

        rot_sensor_var = np.zeros(len(rot_sensor_values[0]))
        for sensor_values in rot_sensor_values:
            rot_sensor_var += (rot_sensor_avg - sensor_values)**2
        rot_sensor_var = rot_sensor_var/len(rot_sensor_values)

        return p_sensor_avg, p_sensor_var, rot_sensor_avg, rot_sensor_var




class SampleDataAnalysis:

    ## at index -2 is arm degree, at index -1 is orthosis degree

    ## either initialize from csv file, or from prebuilt arrays
    def __init__(self, label, file_path = None,p_sensors = None,rot_sensors = None,sample_time_stamps = None):


        self.front_path_end = 48
        self.back_top_patch_end = 70

        self.label = label

        if file_path is not None:
            self.file_path = file_path
            self.sheet = openpyxl.load_workbook(self.file_path).active
            self.p_sensors,self.rot_sensors, self.sample_time_stamps = self.load_sample()
        if p_sensors is not None and sample_time_stamps is not None and rot_sensors is not None:
            self.p_sensors = p_sensors
            self.rot_sensors = rot_sensors
            self.sample_time_stamps = sample_time_stamps

        self.time_step = self.sample_time_stamps[1] - self.sample_time_stamps[0]
        self.num_p_sensors = len(self.p_sensors[0])
        self.num_rot_sensors = len(self.rot_sensors[0])
        self.num_time_steps = len(self.sample_time_stamps)

    def load_sample(self):

        num_sensors = self.sheet.max_row
        num_columns = self.sheet.max_column

        sample_time_stamps = np.zeros(num_columns)
        p_sensors = np.zeros((num_columns,num_sensors - 6))
        rot_sensors = np.zeros((num_columns,2))
        for i in range(num_columns):
            sample_time_stamps[i] = self.sheet.cell(2,i+1).value
            rot_sensors[i][0] = self.sheet.cell(num_sensors-1,i+1).value
            rot_sensors[i][1] = self.sheet.cell(num_sensors,i+1).value

            for j in range(0,num_sensors - 6):
                p_sensors[i][j] = self.sheet.cell(j + 4,i + 1).value

        return p_sensors, rot_sensors, sample_time_stamps

    def sensor_values_at_arm_degrees(self, at_arm_degrees):

        min_difference = abs(at_arm_degrees - self.rot_sensors[0][0])
        index = 0

        for  i in range(1,self.num_time_steps):
            difference = abs(at_arm_degrees - self.rot_sensors[i][0])
            if difference < min_difference:
                index = i
                min_difference = difference
            if difference < 10e-5:
                break

        return self.p_sensors[index], self.rot_sensors[index], self.sample_time_stamps[index]

    ## average for each sensor through out time range
    def sensorwise_avg(self, avg_range = None):
        if avg_range is None: avg_range = range(self.num_time_steps)

        p_sensors_avg = np.zeros(self.num_p_sensors)
        rot_sensors_avg = np.zeros(2)
        for i in avg_range:
            p_sensors_avg += self.p_sensors[i]
            rot_sensors_avg += self.rot_sensors[i]


        return p_sensors_avg/len(avg_range), rot_sensors_avg/len(avg_range)

    ## average at each time step through out sensor range
    def timewise_avg(self,sensor_range = None):
        if sensor_range is None: sensor_range = [0,self.num_p_sensors]

        p_time_stamp_avg = np.zeros(self.num_time_steps)
        rot_time_stamp_avg = np.zeros(self.num_time_steps)
        for i in range(self.num_time_steps):
            p_time_stamp_avg[i] = sum(self.p_sensors[i][sensor_range[0]:sensor_range[1]])/self.num_p_sensors
            rot_time_stamp_avg[i] = sum(self.rot_sensors[i])/self.num_rot_sensors

        return p_time_stamp_avg,rot_time_stamp_avg

    def total_avg(self):
        p_avg,rot_avg = self.sensorwise_avg(range(0,self.num_time_steps))
        return sum(p_avg)/len(p_avg), sum(rot_avg)/len(rot_avg)

    def avg_at_arm_degrees(self,at_arm_degrees):

        p_values,rot_values,time_stamp = self.sensor_values_at_arm_degrees(at_arm_degrees)

        return sum(p_values)/len(p_values), sum(rot_values)/len(rot_values)

    def get_p_value(self,column, row):
        return self.p_sensors[column][row]
    def get_rot_values(self,column,row):
        return self.rot_sensors[column][row]
    def get_p_values_at(self,column):
        return self.p_sensors[column]
    def get_rot_values_at(self,column):
        return self.rot_sensors[column]
    def get_time_at(self,column):
        return self.sample_time_stamps[column]




    ## makes sample smaller in time domain, sets k time stamps, and for each computes the average of the timestamps that fall into the new time stamp range
    def shrink_sample_time_domain(self,k_shrink):

        shrink_ratio = self.num_time_steps//k_shrink

        new_p_sensors = np.zeros((k_shrink,self.num_p_sensors))
        new_rot_sensors = np.zeros((k_shrink,self.num_rot_sensors))
        new_sample_time_stamps = np.zeros(k_shrink)
        for i in range(k_shrink):
            new_p_sensors[i], new_rot_sensors[i] = self.sensorwise_avg(range(i*shrink_ratio,(i+1)*shrink_ratio))
            new_sample_time_stamps[i] = self.sample_time_stamps[i*shrink_ratio]

        return SampleDataAnalysis(p_sensors=new_p_sensors,rot_sensors=new_rot_sensors,sample_time_stamps=new_sample_time_stamps,label=self.label)

    def extrema_pressure_time_stamp(self, in_range = None):
        if in_range is None: in_range = range(self.num_time_steps)

        max_i = in_range[0]
        max_press = sum(self.p_sensors[in_range[0]])
        min_i = in_range[0]
        min_press = sum(self.p_sensors[in_range[0]])


        for i in in_range:

            p_sum = sum(self.p_sensors[i])
            if p_sum < min_press:
                min_i = i
                min_press = p_sum
            if p_sum > max_press:
                max_i = i
                max_press = p_sum

        return self.p_sensors[min_i], self.p_sensors[max_i]

    def extrema_pressure_sensor(self):

        min_i = 0
        min_press = sum(self.p_sensors[:,0])
        max_i = 0
        max_press = sum(self.p_sensors[:,0])

        for i in range(1,self.num_p_sensors):
            p_sum = sum(self.p_sensors[:,i])
            if p_sum < min_press:
                min_i = i
                min_press = p_sum
            if p_sum > max_press:
                max_i = i
                max_press = p_sum

        return self.p_sensors[:,min_i], self.p_sensors[:,max_i]


def plot_pressure_data(sensor_array, save_fig_as = None):



    force_limit = max(sensor_array)
    plot = Plot(10,force_limit = 100)
    plt.pause(0.05)
    plot.update_pressure_plot(sensor_array[:48], sensor_array[48:70], sensor_array[70:])
    plt.title(save_fig_as)
    plt.show()

    if save_fig_as is not None:
        plt.pause(5)
        plt.savefig(save_fig_as)
        plt.close()


def plot_time_series_data(time_range,avg_pressure_per_timestep, variance = None, save_fig_as = None):
    if variance is None: variance = np.zeros(len(avg_pressure_per_timestep))

    plt.errorbar(time_range,avg_pressure_per_timestep,yerr=variance,color="blue",ecolor="lightgrey")
    plt.title("Mean sensor value per timestep")
    plt.xlabel("timestep")
    plt.ylabel("sensor value")
    plt.title(save_fig_as)

    if save_fig_as is not None:

        plt.savefig(save_fig_as)

    plt.show()


#
# data_folder = "C:/University/Marble/Data/"
# data_dir = os.listdir(data_folder)
# for index, labeled_data in enumerate(data_dir):
#     if index <= -1: continue
#     folder = data_folder + labeled_data
#
#     at_degrees = 0
#     if "90" in labeled_data: at_degrees = 90
#     elif "45" in labeled_data: at_degrees = 45
#
#
#
#     da = LabelDataAnalysis(folder, None)
#     p_mean, p_var, rot_mean, rot_var = da.mean_variance(time_wise=True, at_arm_degrees=None, linearized=False)
#     plot_time_series_data(np.linspace(0.5,10,20),p_mean,p_var, save_fig_as=labeled_data + "_back_bottom.png")
#
#     print(folder)
#     print("Degrees: ",at_degrees)
#     print("Actual: ", rot_mean[0])

# TODO: Custimize range of sensors/ time in .mean_variance()

# add to the plot circle with radius of the value of variance
# both avg and variance