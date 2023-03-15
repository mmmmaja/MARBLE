import numpy as np
import random
from separation_functions import *

class SpacalAlgo:


    def __init__(self, separation_function, sensor_cnt, loc_dim = 3):

        self.loc_dim = loc_dim
        self.separation_function = separation_function
        self.sensor_cnt = sensor_cnt
        self.separation_min = separation_function.min_sep
        self.separation_max = separation_function.max_sep
        self.sensor_locations = self.init_sensor_positions(self.sensor_cnt, self.sensor_cnt*self.separation_min, loc_dim)
        self.proximity_pairs = self.init_proximity_pairs(self.sensor_cnt)

    def init_sensor_positions(self, sensor_cnt, separation_range, loc_dim):

        sensors_loc = []

        for i in range(sensor_cnt):
            i_loc = np.array([random.random()*separation_range for j in range(loc_dim)])
            sensors_loc.append(i_loc)

        return sensors_loc

    ## the pairs will be stored in array (not dict) due to complexity reasons - space complexity is O(n^2)
    def init_proximity_pairs(self,sensor_cnt):

        proximity_pairs = []

        for i in range(sensor_cnt):
            for j in range(i+1,sensor_cnt):

                proximity_pairs.append(0)

        return proximity_pairs

    ## to retrieve the pair value of 2 particular sensors we recompute the index at which the pair is located in the self.proximity_pairs array
    def get_pair_id(self, a, b):

        ## a has to be the smaller id
        if a > b:
            a,b = b,a

        ## The array of pairs is composed of n*(n-1) elements, where each sensor A is paired with only sensors of larger id, since a sensor with smaller id already paired with A (by construction)
        ## EXAMPLE:
        ## a = 2, b = 5, Since sensors with id < 2 are first in  the array of pairs, sensor with id = 0 pairs with all N sensors,
        ## then sensor with id = 1 only has to pair with N-1 sensors (since pair [0,1] was already paired)
        ## thus before a = 2, N + N - 1 indexes are taken. Then at a = 2, it first pairs with 3, then 4, and then with 5.
        ## Therefore, it takes 3 more indexes to arrive to the index corresponding to pair [2,5]
        ## Generally to arrive at indexes regarding 'a', we compute it using linear series sum formula. Finally, to arrive to 'b', we compute (b - a - 1), to know how many indexes till we get to pair [a,b]

        ## Reason why we do it this complicated is that array has O(1) complexity, and we ideally want to access as fast as possible since the complexity with regards to N sensors is O(N^2)
        id = (self.sensor_cnt + (self.sensor_cnt - a + 1)) * (a/2) + (b - a - 1)

        return id

    ## Updates the locations of pairs based on recorded pressure values
    def update_locations(self, sensors):

        activated_sensors = self.treshold_split(sensors)
        self.increment_proximity(activated_sensors)



    ## splits into activated and non-activated sensors based on pressure threshold
    ## NOTE: threshold is for now the mean of the sensor array, however there might be better options
    def treshold_split(self, sensors):

        mean_treshold = np.mean(sensors)

        activated_sensors = []

        for i, pressure in enumerate(sensors):
            if pressure > mean_treshold:
                activated_sensors.append(i)

        return activated_sensors




    ## Alternative way on for 'determining' which sensors were actually activated
    def k_means_split(self, sensors, center_shift_bound):

        mean = np.mean(sensors)
        std = np.std(sensors)


        sets = None
        last_centers = None
        centers = np.array([mean - 2*std,mean + 2*std])

        while np.linalg.norm(centers - last_centers) > center_shift_bound:

            sets = ([], [])
            for i in range(len(sensors)):


                c_ix = 0
                c_dist = abs(sensors[i] - centers[c_ix])

                for j in range(1,len(centers)):
                    dist = abs(sensors[i] - centers[j])
                    if dist < c_dist:
                        c_ix = j
                        c_dist = dist

                sets[c_ix].append(i)


            last_centers = np.copy(centers)
            for i in range(len(sets)):
                centers[i] = np.mean(sets[i])


            ## returns the set with higher activated sensors
            return sets[1]

    ## increment the counters of how many times were pairs of sensors activated (increment only on pairs of activated sensors!)
    def increment_proximity(self, activated_sensors):

        for i in range(len(activated_sensors)):
            for j in range(i+1, len(activated_sensors)):

                small = activated_sensors[i]
                big = activated_sensors[j]

                pair_id = self.get_pair_id(small,big)

                self.proximity_pairs[pair_id] += 1

    ## Compute the derivatives with respect to each coordinate parameter
    def get_coord_param_derivatives(self):

        ## derivative wrt each sensor and each of its coordinates
        diffs = [np.zeros(self.loc_dim) for i in range(self.sensor_cnt)]


        ## COST FUNCTION:
        ## C(a,b,...,n) = (a - b)^2 + (a - c)^2 + ....  + (a - n)^2 + (b - c)^2 + ... + (b - n)^2  | include each term IFF its magnitude is larger than the required distance of the pair of sensors in question
        ## EXAMPLE:
        ## take sensors a, b. If the required distance of sensors a and b is 10, and (a - b)^2 > 10^2, then include the term in the cost function
        ##
        ## Then to minimize the cost of the cost function, compute the analytical derivative wrt each sensor position and each of its coordinate
        ## dC/db (a,b,...,n) = -2(a - b) + 2(b - c) + ....
        ## NOTE: since we include each pair only once, and variable b is not the first, in some terms it is (a - b)^2,
        ## and in others it is (b - c)^2, thus to account for that some derivative terms (wrt b) have - sign


        for i in range(self.sensor_cnt):

            i_diff = diffs[i]

            for j in range(i + 1, self.sensor_cnt):

                j_diff = diffs[j]

                pair_id = self.get_pair_id(i, j)

                maximum_d = self.separation_function.f(self.proximity_pairs[pair_id])

                i_ = self.sensor_locations[i]
                j_ = self.sensor_locations[j]

                d = np.linalg.norm(i_ - j_)


                ## if distance of points i and j is larger than it should be, compute the derivative w.r.t. each coordinate of i

                ## COMMENT THIS!!!
                if d > maximum_d:
                    i_diff += (1/d)*(i_ - j_)
                    j_diff += (1/d)*(i_ - j_)*(-1)

        return diffs

    ## update the locations of all sensors based on common actions in between pairs of sensors
    def update_sensor_locations(self, learning_rate, n = 1):

        for r in range(n):

            diffs = self.get_coord_param_derivatives()
            for i, diff in enumerate(diffs):

                self.sensor_locations[i] -= diff*learning_rate




## DEMO
if __name__ == "__main__":


    sensor_cnt = 80 ## amount of physical sensors
    dim = 2 ## dimension of location vectors

    min_sep = 1.5 ## smallest distance of 2 sensors
    max_sep = 3 ## most distant 2 commonly activated sensors can be (size of the outside stimuli + slack since silicon deformation will activate neighboring sensors probably)
    separation_function = ExpSep(min_sep,max_sep)



    sensors = [0 for i in range(sensor_cnt)] ## placeholder values (irl they would be recorded from the arm or from a simulation model)
    for i in range(4): sensors[i] = 3

    alg = SpacalAlgo(separation_function,sensor_cnt,loc_dim=dim)

    ## Placeholder loop that simulates the fact that we have ultiple frames of recording
    for i in range(10):
        alg.update_locations(sensors)

    ## after we register common pairwise activations, the positions of sensors will get updated, 0.1 is the learning rate, and 'n' is the amount of update iterations
    alg.update_sensor_locations(0.1,n = 1000)









