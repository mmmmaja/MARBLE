import sys

import numpy as np
import random
from DEPRECIATED_separation_functions import *
import scipy
from pressure_recording_manager import *
from GUI_models import BeliefModel, GeneralModel


from activation_decider import *





def rand_unit_vect(dir_dim, full_dim):
    dir = 2*np.concatenate([np.random.random(dir_dim), np.zeros(full_dim - dir_dim)]) - np.concatenate([np.ones(dir_dim),np.zeros(full_dim-dir_dim)])
    dir /= np.linalg.norm(dir)

    return dir

def loc_generator(min_sep,max_sep, center, dim):

    mag = (np.random.random()*(max_sep - min_sep) + min_sep) * -1 if np.random.random() >= 0.5 else 1
    dir = rand_unit_vect(dim,len(center))

    return center + dir*mag




class BeliefSpacialAlgo:
    """
    BeliefSpacalAlgo is class that server for determining locations of a list of sensors
    The core algorithm is based on beliefs. So At start the sensors could be anywhere, so we have very low belief in their location.
    The more recordings we provide, the more proximity information to other sensors does the algorithm have. The Algo gradually shifts around
    the sensors according to the proximity information, and strengthens their belief accordingly.
    """

    # separation function estimates how close two sensors are based on their common activations
    # area_range describes what is the range of interest (the initial random sensors will be generated in this space)
    def __init__(self,decider : ActivationDecider, min_sep,max_sep, sensor_cnt, area_range=((0, 100), (0, 100), (0, 100)),conv_rate = 10e-4, seed = 0):
        """

        :param decider: Activation Decider that decides which sensors are activated and which are not
        :param min_sep: minimum distance two sensors can be at
        :param max_sep: maximum distance two sensors can be at
        :param sensor_cnt: amount of sensors
        :param area_range: the range in which sensors can appear
        :param seed: random seed, for reproducibility
        """
        self.random = random.Random()
        self.random.seed(seed)
        np.random.seed(seed)

        self.conv_rate= conv_rate

        self.decider = decider

        self.min_sep = min_sep
        self.max_sep = max_sep

        self.area_range = area_range
        self.loc_dim = len(area_range)
        self.sensor_cnt = sensor_cnt
        self.sensor_locations = self.init_sensor_positions(self.sensor_cnt, area_range=self.area_range)

        self.sensor_beliefs = np.array([0.0] * sensor_cnt)
        self.sensor_compound_beliefs = [0.0]*self.sensor_cnt

        self.known_sensors = set()



        self.conv_f = lambda x: -np.exp(-conv_rate*x) + 1


    def get_locations(self):

        return np.copy(self.sensor_locations)

    def get_beliefs(self):
        return np.copy(self.sensor_beliefs)

    def set_known_sensors(self, ids, locations):
        """
        sets pre-specified sensors to fixed locations
        :param ids: ids of sensors that we want to fix the locations of
        :param locations: locations of the sensors we want to fix in space
        :return:
        """
        for i, id_ in enumerate(ids):
            self.known_sensors.add(id_)
            self.sensor_locations[id_] = np.copy(locations[i])
            self.sensor_beliefs[id_] = 1
            self.sensor_compound_beliefs[id_] = 1000/self.conv_rate


    def init_sensor_positions(self, sensor_cnt, area_range):
        sensors_loc = []

        for i in range(sensor_cnt):
            loc = []
            for dim_range in area_range:
                loc.append((dim_range[1] - dim_range[0]) * random.random())
            sensors_loc.append(np.array(loc))

        return sensors_loc



    def is_valid_neighbor_distance(self,distance):
        """

        :param distance: distance scalar
        :return: returns true if a distance (of two paints) is within the possible distance of the two sensors, assuming they are neighbors. Neighbors in a sense that they both can be activated by outside stimuli at the same time
        """

        return self.min_sep <= distance <= self.max_sep

    def soft_max(self,pprobs):
        pprobs = np.array(pprobs)
        s = np.sum(np.exp(pprobs))
        return [np.exp(p)/s for p in pprobs]

    def get_coord_vectors(self,activated_sensors):
        """
        for each activated sensor, compute a vector where the sensor should shift
        :param activated_sensors: list of activated sensor ids
        :return: returns a lists of shift vectors for ALL sensors [Not just the activated ones]
        """


        vector_list = [np.zeros(self.loc_dim) for i in range(self.sensor_cnt)]


        for i in range(len(activated_sensors)):
            a_id = activated_sensors[i]
            a_loc = self.sensor_locations[a_id]

            vects = []
            beliefs = []



            for j in range(len(activated_sensors)):
                if i == j: continue


                b_id = activated_sensors[j]
                b_belief = self.sensor_beliefs[b_id]
                b_loc = self.sensor_locations[b_id]

                designated_loc = loc_generator(self.min_sep,self.max_sep, b_loc,dim = 2)

                v = (designated_loc - a_loc)

                dist = np.linalg.norm(b_loc - a_loc)
                v = (np.random.random()*(self.max_sep - self.min_sep) + dist - self.max_sep)*(b_loc - a_loc)/dist
                vects.append( v )
                beliefs.append(b_belief)


            soft_max_b = self.soft_max(beliefs)
            vect = np.zeros(self.loc_dim)
            for j,b in enumerate(soft_max_b): vect += vects[j]*b
            vector_list[a_id] = vect


        return vector_list

    def get_coord_correction_vectors(self, optimize = False):
        """
        coputes the correction shift vectors for all sensors. This is important, since when shifting sensors toghether according to
        proximity information, we disregard the fact that sensors must be separated at least by minimum separation distance. To correct for this
        error, we compute for each pair, if it is too close, how much should the sensors shift away
        :param optimize: if True the vectors are computed faster, but less accurately
        :return: list of shift vectors for each sensor
        """

        vectors = [np.zeros(self.loc_dim) for i in range(self.sensor_cnt)]

        for i in range(len(self.sensor_locations)):
            if i in self.known_sensors: continue

            a_loc = self.sensor_locations[i]
            a_belief = self.sensor_beliefs[i]

            # if we wish to make the correction faster, we skip sensors with higher probability of being correct
            if optimize and a_belief > random.random(): continue

            k = 0
            for j in range(len(self.sensor_locations)):
                if i == j: continue

                b_loc = self.sensor_locations[j]

                shift_dir = b_loc - a_loc
                dist = np.linalg.norm(shift_dir)
                shift_dir = shift_dir / dist

                ## If sensors are too close, we shift sensor A away from sensor B according to belief,
                ## The more belief we have in sensor A, the less we shift A, and in other iteration we will shift sensor B more
                if dist < self.min_sep:
                    k+=1
                    vectors[i] += shift_dir*(dist - self.min_sep)*(1 - a_belief + random.random())

            if k > 0:
                vectors[i] /= k

        return vectors


    def update_beliefs(self,activated_sensors):
        """
        updates the beliefs of all sensors. After sensor locations are updated according to the shift vectors, we also have to update,
        how much we believe each sensor to be at a location where it is.
        For each sensor, we go  through all the rest, and update the belief according to the beliefs of the rest of sensors.
        If a sensor was activated and shifted towards sensors that we have high belief in, the sensor was probably shifted in the right direction,
        so the belief should go up.
        :param activated_sensors: list of activated sensors
        :return:
        """
        new_beliefs = self.get_beliefs()


        for i in range(len(activated_sensors)):
            a_id = activated_sensors[i]

            if a_id in self.known_sensors: continue


            for j in range(len(activated_sensors)):
                if i == j: continue

                b_id = activated_sensors[j]
                b_belief = self.sensor_beliefs[b_id]


                self.sensor_compound_beliefs[a_id] += b_belief

            new_beliefs[a_id] = self.conv_f(self.sensor_compound_beliefs[a_id])

        self.sensor_beliefs = new_beliefs


    ## update the locations of all sensors based on common actions in between pairs of sensors
    def update_sensor_locations(self,sensors):
        """
        update locations of all sensors
        :param sensors: list of pressure values for each sensor, where the index of the array is the id of a sensor
        :return:
        """

        activated_sensors, pressure_values = self.decider.decide_activated(sensors)
        vectors = self.get_coord_vectors(activated_sensors)

        for id_ ,vect in enumerate(vectors):

            self.sensor_locations[id_] += vect*(1-self.sensor_beliefs[id_])

        self.update_beliefs(activated_sensors)

    def correct_sensor_locations(self,optimize = False):
        """
        corrects the locations of sensors, so they are not too close.
        :param optimize: if True, it will be faster but less accurate
        :return:
        """

        vectors = self.get_coord_correction_vectors(optimize= optimize)

        for id_, vect in enumerate(vectors):
            self.sensor_locations[id_] += vect



## DEMO
if __name__ == "__main__":

    MIN_SEP = 1
    MAX_SEP = 2

    sensor_positions, time_frames = read_labeled_recording("../pygame_model/rand1.csv")
    known_pts, known_positions = filter_sensors(set(range(42, 62, 3)), sensor_positions)

    known_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 91,
                 92, 93, 94, 95, 96, 97, 98, 99]
    known_pts = [0,9,90,99]
    known_pts = [0]
    known_pts = [36]
    known_positions = get_positions_of_sensors(sensor_positions, known_pts)




    model = BeliefModel(time_frames=time_frames, positions=sensor_positions, static_points=known_pts)

    decider = CountDecider(4)
    algo = BeliefSpacalAlgo(decider,MIN_SEP,MAX_SEP,len(sensor_positions),conv_rate=0.005,area_range=((0,10),(0,10),(0,0)),seed=1)

    algo.set_known_sensors(known_pts, known_positions)


    compare_location_estimates(sensor_positions, algo.get_locations())

    point_ev = [sensor_positions]
    belief_ev = [algo.get_beliefs()]

    for epoch in range(10):
        point_ev.append(algo.get_locations())
        belief_ev.append(algo.get_beliefs())
        for frame in time_frames:

            algo.update_sensor_locations(frame)
            if np.random.random() > 0.7:
                algo.correct_sensor_locations(optimize=True)



    point_ev.append(algo.get_locations())
    belief_ev.append(algo.get_beliefs())

  #  model.plot_evolution(point_ev, belief_ev)
  #  compare_location_estimates(sensor_positions, algo.get_locations())

    est_locs = algo.get_locations()
    X = matrix_least_squares(est_locs,sensor_positions)
    est_locs = [X.dot(loc) for loc in est_locs]
    GeneralModel().plot_points(est_locs)








