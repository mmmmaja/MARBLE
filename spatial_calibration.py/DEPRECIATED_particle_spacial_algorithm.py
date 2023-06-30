import sys

import numpy as np
import random
from pressure_recording_manager import *
from matplotlib import cm
import queue
import scipy
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from activation_decider import *
import queue
from scipy import stats


class WeightDistribution:


    def __init__(self, root_loc , distribution):


        self.distribution = distribution
        self.root_loc = root_loc


    def __repr__(self):
        return f"<LocationDistribution| comb_cnt: {len(self.root_loc)}, avg_rt_loc: {np.mean(self.root_loc)}>"


    def pdf(self, loc, log = True):

        dist = np.linalg.norm(loc - self.root_loc)

        pdf_ = self.distribution.pdf(dist)

        return pdf_ if not log else np.log(pdf_)




class SensorParticleSwarm:


    def __init__(self,id_, particle_cnt, area = ((0,0,0),(100,100,100))):
        self.id_ = id_
        self.dim = len(area[0])

        self.particles = np.random.uniform(area[0],area[1],(particle_cnt,self.dim))
        self.particle_weights = [1/particle_cnt]*particle_cnt
        self.swarm_belief = 1/particle_cnt

        self.reset_cnt = 0
        self.particle_cnt = particle_cnt

        self.area = area

    def __repr__(self):
        return f"id: {self.id_}, particle_len: {len(self.particles)}, belief: {self.swarm_belief}, reset_cnt:{self.reset_cnt}"


    def add_noise(self, noise_gen):

        noise = np.reshape(np.array(noise_gen.rvs(size = self.particles.size)),self.particles.shape)

        for i in range(self.dim):
            for j in range(len(self.particles)):
                nv = self.particles[j,i] + noise[j,i]

                if nv >= self.area[0][i] and nv <= self.area[1][i]:
                    self.particles[j,i] = nv



    def get_particles(self):
        return self.particles

    def get_weights(self):
        return self.particle_weights

    def get_belief(self):
        return self.swarm_belief

    def set_particle_set(self,particles, weights):

        self.particles = np.array(particles, dtype=float)
        self.particle_weights = np.array(weights, dtype=float)
        self.swarm_belief = max(weights)

        self.reset_cnt += 1


    def particle_var(self):

        return np.mean(np.var(self.particles, axis= 1))


class ParticleSpacalAlgo:
    """
    Bayes Spacal Algorithm is an alternative algorithm for determining the locations of sensors
    It is based on:
    1. Determining which sensors were activated, and analysing In case 2 sensors were activated, how distant they are (this is done on the 'training data'). The distance behavior is modelled by a probabilistic distribution.
    2. Given a few sensors that have known positions, and a new sensor that was activated with such sensors; we know due to the previously estimated probabilistic distribution where the new sensor is 'probably located'
    3. We place sensors with unknown positions continually according to 2. in specific locations, and make their locations known, and then repeat with a new set of unknown sensors
    """

    def __init__(self, decider : ActivationDecider, min_sep, max_sep, sensor_cnt,area = ((0,0,0),(100,100,100)),particle_cnt = 100, particle_decay = 10e-4,dim = 3, seed=0):
        """

        :param decider: decider that decides which sensors are activated
        :param min_sep: minimum distance of two sensors
        :param max_sep: maximum distance of two sensors, given that they are jointly activated (so it is also the size of the outside stimuli)
        :param sensor_cnt: amount of sensors
        :param dim: location dimesion
        :param seed: seed for reproducibility
        """

        self.dim = dim
        self.decider = decider
        self.random = random.Random()
        self.random.seed(seed)

        self.min_sep = min_sep
        self.max_sep = max_sep

        self.sensor_cnt = sensor_cnt
        self.particle_cnt = particle_cnt

        self.area = area


        self.distance_lim = self.min_sep*0.33

        self.particle_decay = particle_decay


        self.sensor_particle_swarms = [SensorParticleSwarm(i,particle_cnt,area) for i in range(sensor_cnt)]


    def set_known_sensors(self,indicies, locations):

        for i in range(len(indicies)):
            self.sensor_particle_swarms[i].set_particle_set(particles=[np.array(locations[i])],weights=[1])


    def update_hypothesis(self, sensor_reading):

        activated_sensors, pressure_values = self.decider.decide_activated(sensor_reading)

        percent_activated = len(activated_sensors) / self.sensor_cnt
        print(f"{percent_activated}  activated")
        if percent_activated > 0.5:
            print("[[Warning] more than half of the sensors were activated. Check if the recording is correct. This recording frame will be skipped]")

        else:

            self.update_swarms(activated_sensors,pressure_values)


    def update_swarms(self, activated_sensors, pressure_values):

        for a in activated_sensors:
            for b in activated_sensors:

                if a == b: continue

                primary_swarm = self.sensor_particle_swarms[a]
                hypothesis_swarm = self.sensor_particle_swarms[b]

                if primary_swarm.swarm_belief < hypothesis_swarm.swarm_belief: continue
                if primary_swarm.swarm_belief < 0.5: continue


                prev_var = hypothesis_swarm.particle_var()
                self.reweight_sensor_swarms(primary_swarm,hypothesis_swarm)
                self.resample_swarm(hypothesis_swarm, particle_cnt=hypothesis_swarm.particle_cnt)
                hypothesis_swarm.add_noise(stats.norm(loc = 0, scale = 0.2))
                post_var = hypothesis_swarm.particle_var()


                hypothesis_swarm.particle_cnt = int(max(np.ceil(hypothesis_swarm.particle_cnt * self.get_swarm_decay(hypothesis_swarm)*post_var/prev_var),1))


    def get_swarm_decay(self,swarm :SensorParticleSwarm):

        return np.exp(-self.particle_decay * swarm.reset_cnt)

    def add_sensor_location_gauss_noise(self, scale=1):

        for sensor in self.sensor_particle_swarms:
            sensor.add_noise(stats.norm(loc=0, scale=scale))


    def reweight_sensor_swarms(self,primary_swarm : SensorParticleSwarm,hypothesis_swarm : SensorParticleSwarm):


        weights = []
        for hypo_particle in hypothesis_swarm.get_particles():

            s = 0.5743268
            loc = 0.811029
            weight_distribution = WeightDistribution(root_loc=hypo_particle,distribution=stats.lognorm(loc  =loc, s = s))

            agg_weight = 0

            for primary_particle in primary_swarm.get_particles():
                agg_weight += weight_distribution.pdf(primary_particle,log = False)

            weights.append(agg_weight)

        weights = np.array(weights)/sum(weights)

        hypothesis_swarm.set_particle_set(hypothesis_swarm.get_particles(),weights = weights)


        return hypothesis_swarm

    def resample_swarm(self,swarm :SensorParticleSwarm, particle_cnt = None):

        if particle_cnt is None: particle_cnt = len(swarm.get_particles())

        resampled_indicies = np.random.choice(np.arange(len(swarm.get_particles())), particle_cnt, p = swarm.get_weights())

        resampled_particles = swarm.get_particles()[resampled_indicies]
        weights = [1/particle_cnt]*particle_cnt

        swarm.set_particle_set(resampled_particles,weights)

        return swarm





if __name__ == "__main__":

    MIN_SEP = 1
    MAX_SEP = 2

    sensor_positions, time_frames = read_recording("../pygame_model/data_2sz.csv")
    known_pts, known_positions = filter_sensors(set([0]), sensor_positions)

    known_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 91,
                 92, 93, 94, 95, 96, 97, 98, 99]

    #known_pts = [0,9,90,99]
    known_pts = [0]
    known_positions = get_positions_of_sensors(sensor_positions, known_pts)

    #model = SpatialModel(time_frames=time_frames, positions=sensor_positions, static_points=known_pts)
    #
    decider = MeanThresholdDecider(threshold=3)
    decider = CountDecider(4)
    algo = ParticleSpacalAlgo(decider,MIN_SEP,MAX_SEP,len(sensor_positions),area=((0,0,0),(10,10,0)), particle_cnt = 100,particle_decay=10e-3)




    algo.set_known_sensors([0],[np.array([0,0,0])])

    for frame in time_frames[1:]:

        print(frame)

        algo.update_hypothesis(frame)


    for sensor in algo.sensor_particle_swarms:
        print(sensor)
        print(sensor.particles[:3])




    # loc_dist = LocationDistribution([np.array([0,0,0]),np.array([-1.56,0,0]),np.array([-0.78,0.504,0])],[scipy.stats.lognorm(loc = 0.831029,s =1.33),scipy.stats.lognorm(loc = 0.831029,s = 1.25),scipy.stats.lognorm(loc = 0.831029,s = 0.2)])
    # loc_dist.plot(max_=True)

   # loc_dist = LocationDistribution([np.array([0,0,0]),np.array([2,0,0])],[scipy.stats.lognorm(loc = 0.831029,s = 0.6743268),scipy.stats.lognorm(loc = 0.831029,s = 0.6743268)])
    #loc_dist = LocationDistribution([np.array([0,0,0])],[scipy.stats.lognorm(loc = 0.831029,s = 0.6743268)])
    # print(loc_dist.pdf(np.array([1,0,0])))
    #loc_dist.plot(max_=False)
 #   max_X = scipy.optimize.fmin(lambda X: -loc_dist.pdf(X),np.array([2,1]))
  #  print(max_X)
    #
    # loc_dist.plot(range_=((-3,-3,0),(7,7,0)),max_ = False)
    # # print(graph.node_set[0].get_neighbors())





