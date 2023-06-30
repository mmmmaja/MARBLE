import sys

import numpy as np
import random
from DEPRECIATED_separation_functions import *
import scipy
from pressure_recording_manager import *
from GUI_models import BeliefModel


from activation_decider import *


class BeliefSpacalAlgo:
    """
    BeliefSpacalAlgo is class that server for determining locations of a list of sensors
    The core algorithm is based on beliefs. So At start the sensors could be anywhere, so we have very low belief in their location.
    The more recordings we provide, the more proximity information to other sensors does the algorithm have. The Algo gradually shifts around
    the sensors according to the proximity information, and strengthens their belief accordingly.
    """
    THRESHOLD_B = 4
    OLD_BELIEF = 0.9
    NEW_BELIEF = 0.1

    # separation function estimates how close two sensors are based on their common activations
    # area_range describes what is the range of interest (the initial random sensors will be generated in this space)
    def __init__(self,decider : ActivationDecider, min_sep,max_sep, sensor_cnt, area_range=((0, 100), (0, 100), (0, 100)), seed = 0):
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

        self.decider = decider

        self.min_sep = min_sep
        self.max_sep = max_sep

        self.area_range = area_range
        self.loc_dim = len(area_range)
        self.sensor_cnt = sensor_cnt
        self.sensor_locations = self.init_sensor_positions(self.sensor_cnt, area_range=self.area_range)
        self.sensor_beliefs = np.array([0.0] * sensor_cnt,dtype=float)

        self.known_sensors = set()


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



    def get_coord_vectors(self,activated_sensors):
        """
        for each activated sensor, compute a vector where the sensor should shift
        :param activated_sensors: list of activated sensor ids
        :return: returns a list of shift vectors for ALL sensors [Not just the activated ones]
        """


        vectors = [np.zeros(self.loc_dim) for i in range(self.sensor_cnt)]


        for i in range(len(activated_sensors)):
            a_id = activated_sensors[i]
            a_belief = self.sensor_beliefs[a_id]
            a_loc = self.sensor_locations[a_id]

            vect = np.zeros(self.loc_dim)

            for j in range(len(activated_sensors)):
                if i == j: continue

                b_id = activated_sensors[j]
                b_belief = self.sensor_beliefs[b_id]
                b_loc = self.sensor_locations[b_id]

                ## For each pair of sensors that were activated, we shift one of the sensors to the other
                shift_dir = b_loc - a_loc
                dist = np.linalg.norm(shift_dir)

                ## We need to compute what is the direction we want to shift sensor A to sensor B
                shift_dir = shift_dir / dist

                max_shift = dist - self.min_sep # the closest the sensor a can be shifted towards b
                min_shift = dist - self.max_sep # the most distant shift that still shifts sensor a towards sensor b

                # if we are not so sure about the B sensor position, we want to relax moving sensor A towards B since we do not actually know where B is


                #TODO: These lines are the most important Since they determine how much will a sensor shift to a fellow activated sensor
                #TODO: according to the beliefs we have for both sensor positions respectively. tweaking the range of shift may vastly improve or worsen the performance
                relaxed_min_shift = (b_belief*min_shift) # NOTE: we may also incorporate the a_belief
                relaxed_max_shift = relaxed_min_shift + (np.sqrt(b_belief)*(max_shift - relaxed_min_shift))

                # Conversely, if we are SURE that sensor A is correctly placed, the less we want to shift it towards B event hough they were activated
                vect += shift_dir*(1-a_belief)*random.uniform(relaxed_min_shift,relaxed_max_shift) # the more am i sure of my a position, the less i shift

            vectors[a_id] = vect/(len(activated_sensors)-1)

        return vectors

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

            for j in range(len(self.sensor_locations)):
                if i == j: continue

                b_loc = self.sensor_locations[j]

                shift_dir = b_loc - a_loc
                dist = np.linalg.norm(shift_dir)
                shift_dir = shift_dir / dist

                ## If sensors are too close, we shift sensor A away from sensor B according to belief,
                ## The more belief we have in sensor A, the less we shift A, and in other iteration we will shift sensor B more
                if dist < self.min_sep:
                    vectors[i] += shift_dir*(dist - self.min_sep)*(1 + random.random()*0.5 - a_belief)


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
            a_belief = self.sensor_beliefs[a_id]

            if a_id in self.known_sensors: continue

            cross_belief = 0
            for j in range(len(activated_sensors)):
                if i == j: continue

                b_id = activated_sensors[j]
                b_belief = self.sensor_beliefs[b_id]

                cross_belief += b_belief



            new_beliefs[a_id] = (a_belief*500 + cross_belief)/500.0
            if new_beliefs[a_id] >= 0.99: new_beliefs[a_id] = 0.99

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

            self.sensor_locations[id_] += vect

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

    sensor_positions, time_frames = read_labeled_recording("../pygame_model/data_2sz.csv")
    known_pts, known_positions = filter_sensors(set(range(42, 62, 3)), sensor_positions)

    known_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 91,
                 92, 93, 94, 95, 96, 97, 98, 99]
    #known_pts = [0,9,90,99]
    #known_pts = [0]
    known_positions = get_positions_of_sensors(sensor_positions, known_pts)

    model = BeliefModel(time_frames=time_frames, positions=sensor_positions, static_points=known_pts)

    decider = MeanThresholdDecider(threshold=3)
    decider = CountDecider(4)
    algo = BeliefSpacalAlgo(decider,MIN_SEP,MAX_SEP,len(sensor_positions),area_range=((0,9),(0,9),(0,0)),seed=1)

    algo.set_known_sensors(known_pts, known_positions)

    compare_location_estimates(sensor_positions, algo.get_locations())

    point_ev = [sensor_positions]
    belief_ev = [algo.get_beliefs()]

    for epoch in range(10):
        point_ev.append(algo.get_locations())
        belief_ev.append(algo.get_beliefs())
        for frame in time_frames:
            algo.update_sensor_locations(frame)

        algo.correct_sensor_locations()

    point_ev.append(algo.get_locations())
    belief_ev.append(algo.get_beliefs())

    model.plot_evolution(point_ev, belief_ev)
    compare_location_estimates(sensor_positions, algo.get_locations())


