import random

import numpy as np



## Various functions that relay pressure values of a pair of sensors into their estimated distance

def naive_distance_converter(pressure_a, pressure_b):



    if pressure_a > 1 and pressure_b > 1:
        return 1 + np.random.random(1)*0.5
    if pressure_a < 1 and pressure_b < 1:
        return 4 + np.random.random(1) - 0.5
    else:
        return 2 + (np.random.random(1) - 0.5)*0.25




def expon_distance_converter(pressure_a,pressure_b):

    return np.exp(-(np.abs(pressure_a) + np.abs(pressure_b))*1)*5 + 1


def std_converter(pressure_a,presure_b):

    return  0.1 - 0.1*(2/(1 + np.exp(np.abs(pressure_a) + np.abs(presure_b))) - 1)


def generate_location(a,b):

    loc = np.zeros(len(a))

    for i in range(len(a)):
        loc[i] = a[i] + np.random.random()*(b[i] - a[i])

    return loc




class LinearSeparationPenalty:


    def __init__(self, sep):

        self.sep = sep

    def pdf(self, distance):


        return 1 if distance >= self.sep else distance/self.sep


class ExpSeparationPenalty:

    def __init__(self,sep, sharpenss = 10):
        if sharpenss < 2: print("[WARNING], the penalty may be insignificant, thus allowing sensors to be very close")


        self.sep = sep
        self.sharp = sharpenss

    def pdf(self,distance):

        if distance >= self.sep:
            return 1
        else:
            return 2/(1 + np.exp(-self.sharp*(distance-self.sep)))



def lognorm_mle(samples):
    samples = np.array(samples)

    mean = np.sum(np.log(samples))/len(samples)
    var = np.sum(np.power(np.log(samples) - np.full(len(samples),mean),2))/len(samples)

    return mean, var



def normalize_pressures(frame):

    max_ = max(frame)
    if max_ == 0: return frame
    return list(map(lambda x:x/max_,frame))



class LognormPressureDistanceConverter:

    def __init__(self,min_sep,max_sep):

        self.min_sep = min_sep
        self.max_sep = max_sep

    def loc_conv(self,pressure_a,pressure_b):
        x = pressure_a
        y = pressure_b
        return np.log(self.max_sep)*np.exp(-(4/self.max_sep)*(x**2 +y**2 -(x-y)**2)) + np.log(self.min_sep)

    def scale_conv(self,pressure_a,pressure_b):
        x = pressure_a
        y = pressure_b
        return 0.5*np.exp(-(2/self.max_sep)*(x**2 + y**2 + (x - y)**2)) + 0.5



class PressureNormalConverter:

    def __init__(self,min_sep,max_sep):
        self.min_mean = min_sep
        self.max_mean = max_sep

        self.min_std = 0.25
        self.max_std = np.std([min_sep,max_sep])


    def convert(self,pressure_a,pressure_b):
        ## Assuming both pressures are in between 0 and 1


        x = (2 - (pressure_a + pressure_b)) / 2

        mean = (self.max_mean - self.min_mean) * (x) + self.min_mean
        std = (self.max_std - self.min_std) * (x) + self.min_std

        return mean, std


def matrix_least_squares(a_pts,b_pts):

    a_pts = np.array(a_pts)
    b_pts = np.array(b_pts)

    x,a,b,c = np.linalg.lstsq(a_pts,b_pts)

    return np.transpose(x)



if __name__ == "__main__":

    pass


