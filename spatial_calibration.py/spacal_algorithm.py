import numpy as np
import random
from separation_functions import *
import scipy

class SpacalAlgo:


    # separation function estimates how close two sensors are based on their common activations
    # area_range describes what is the range of interest (the initial random sensors will be generated in this space)
    def __init__(self, separation_function, sensor_cnt, area_range = ((0,100),(0,100),(0,100))):

        self.area_range = area_range
        self.loc_dim = len(area_range)
        self.separation_function = separation_function
        self.sensor_cnt = sensor_cnt
        self.separation_min = separation_function.min_sep
        self.separation_max = separation_function.max_sep
        self.sensor_locations = self.init_sensor_positions(self.sensor_cnt, area_range=self.area_range)
        self.sensor_beliefs = np.array([1/sensor_cnt]*sensor_cnt)
        self.proximity_pairs = self.init_proximity_pairs(self.sensor_cnt)

        self.known_sensors = set()


    def get_pairs(self):
        return self.proximity_pairs

    def get_locations(self):
        return np.copy(self.sensor_locations)

    def set_known_sensors(self, ids, locations):

        for i, id_ in enumerate(ids):
            self.known_sensors.add(id_)
            self.sensor_locations[id_] = np.copy(locations[i])
            self.sensor_beliefs[id_] = 1

    def init_sensor_positions(self, sensor_cnt, area_range):
        sensors_loc = []

        for i in range(sensor_cnt):
            loc = []
            for dim_range in area_range:
                loc.append((dim_range[1] - dim_range[0])*random.random())
            sensors_loc.append(np.array(loc))

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
        id_ = ((self.sensor_cnt - 1) + (self.sensor_cnt - a)) * (a/2) + (b - a - 1)

        return int(id_)

    ## Updates the activations of pairs based on recorded pressure values
    def update_activations(self, sensors):

        activated_sensors = self.threshold_split(sensors)
        self.increment_proximity(activated_sensors)



    ## splits into activated and non-activated sensors based on pressure threshold
    ## NOTE: threshold is for now the mean of the sensor array, however there might be better options
    def threshold_split(self, sensors):

        mean_threshold = np.mean(sensors)*4
        activated_sensors = []

        for i, pressure in enumerate(sensors):

            # TODO: Now we work with defomations whic are all negative, thus the '<' comparison and not '>'
            if pressure < mean_threshold:
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

                i_ = self.sensor_locations[i]
                j_ = self.sensor_locations[j]


                d = np.linalg.norm(i_ - j_)

                # cost function of 2 sensors being close is relevant only for pairs that were jointly activated, and for pairs
                proximity_pair_cnt = self.proximity_pairs[pair_id]
                if proximity_pair_cnt == 0:
                    if d < self.separation_function.get_max_sep():
                        c = d - self.separation_function.get_max_sep()
                        i_diff += c*(1/d)*(i_ - j_)
                        i_diff += c*(1/d)*(i_ - j_)*(-1)
                    continue

                maximum_d = self.separation_function.f(self.proximity_pairs[pair_id])
                ## if distance of points i and j is larger than it should be, compute the derivative w.r.t. each coordinate of i
                ## EXAMPLE:
                ## 1. Let i be 3d vector of sensor 1, and j of sensor 3
                ## 2. their distance is D = sqrt( (i-j)^2 ) = sqrt( (i1-j1)^2 + (i2-j2)^2 + (i3-j3)^2 )
                ## 3. if D is bigger than it should be, it is considered a cost, and to minimize cost, we take the derivative wrt to each variable in play
                ## 3. dD/di1 = 1/D * 2*i1 ; dD/dj1 = 1/D * 2*(-j1)
                ## We can ignore the constant terms (2 in this case), and thus arrive at the equations used in the code below

                if d > maximum_d:
                    c = d - maximum_d
                    i_diff += c*(1/d)*(i_ - j_)
                    j_diff += c*(1/d)*(i_ - j_)*(-1)

                ## If distance d is smaller than is should be (the minimum distance of 2 points, then it adds to the cost)
                ## The cost term will in this case be C = (d - min_separation)^2
                ## the derivative dC/dd = 2*(d - min_separation)* derivative of d

                elif d < self.separation_function.get_min_sep():
                    c = d - self.separation_function.get_min_sep()
                    i_diff += c * (1 / d) * (i_ - j_)
                    j_diff += c * (1 / d) * (i_ - j_) * (-1)
                    continue




        return diffs

    ## update the locations of all sensors based on common actions in between pairs of sensors
    def update_sensor_locations(self, learning_rate, n = 1, decay = 1):

        for r in range(n):

            diffs = self.get_coord_param_derivatives()
            for i, diff in enumerate(diffs):

                # skip sensors that have known location
                if i in self.known_sensors: continue


                self.sensor_locations[i] -= diff*learning_rate + (np.array([random.random() - 0.5,random.random() - 0.5,0]))*np.exp(-decay*r)




## DEMO
if __name__ == "__main__":


    sensor_cnt = 80 ## amount of physical sensors
    dim = 2 ## dimension of location vectors

    min_sep = 1.5 ## smallest distance of 2 sensors
    max_sep = 3 ## most distant 2 commonly activated sensors can be (size of the outside stimuli + slack since silicon deformation will activate neighboring sensors probably)
    separation_function = ExpSep(min_sep,max_sep)



    sensors = [0 for i in range(sensor_cnt)] ## placeholder values (irl they would be recorded from the arm or from a simulation model)
    for i in range(4): sensors[i] = 3

    alg = SpacalAlgo(separation_function,sensor_cnt,area_range=((0,40),(0,40)))

    ## Placeholder loop that simulates the fact that we have ultiple frames of recording
    for i in range(10):
        alg.update_activations(sensors)

    ## after we register common pairwise activations, the positions of sensors will get updated, 0.1 is the learning rate, and 'n' is the amount of update iterations
    alg.update_sensor_locations(0.1,n = 1000)




class BayesFilterAlgo:


    def __init(self,locations,sensor_ids, stimuli_size):

        self.stimuli_size = stimuli_size
        self.locations = locations
        self.sensor_ids = sensor_ids


        self.sensor_location_belief = dict()

        for id_ in self.sensor_ids:
            probability = 1/len(locations)
            location_belief = {tuple(location):probability for location in self.locations }
            self.sensor_location_belief[id_] = location_belief


    def set_root_sensor(self,id_, location):

        locations = self.sensor_location_belief[id_]
        for loc in locations.keys():
            locations[loc] = 0
        locations[tuple(location)] = 1


    def update_belief(self,activated_sensors):

        for i, id_1 in enumerate(activated_sensors):

            locations_1 = self.sensor_location_belief[id_1]

            for j in range(i+1,len(activated_sensors)):
                id_2 = activated_sensors[j]

                locations_2 = self.sensor_location_belief[id_2]

                # given that sensors i and j were activated together, update the beliefs of their positions
                new_beliefs_2 = {l: 0 for l in locations_2}


                total_belief_1 = 0
                total_belief_2 = 0
                for loc_1 in locations_1.keys():
                    new_belief_1 = 0
                    prob_1 = locations_1[loc_1]

                    for loc_2 in locations_2.keys():

                        distance = np.linalg.norm(np.array(loc_1) - np.array(loc_2))


                        # the bigger the distance the less probability of  these two locations, die to the fact they were activated together
                        # for now draw consider exponential distribution, and determine probability that distance is bigger than [distance] variable
                        belief = 1 - scipy.stats.expon.cdf(distance,loc = 0, scale = 0.9*self.stimuli_size)
                        prob_2 = locations_2[loc_2]

                        new_belief_1 += prob_1*belief*prob_2
                        new_beliefs_2[loc_2] += prob_1*belief*prob_2

                        total_belief_2 += prob_1*belief*prob_2

                    locations_1[loc_1] = new_belief_1
                    total_belief_1 += new_belief_1

                # now we need to normalize the probabilities to sum up to 1
                for loc_2, blf in new_beliefs_2.items():
                    locations_2[loc_2] = blf/total_belief_2

                for loc_1, blf in locations_1.items():
                    locations_1[loc_1] = blf/total_belief_1





