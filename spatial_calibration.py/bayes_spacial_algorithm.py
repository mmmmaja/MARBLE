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


## LocationDistribution is a class that aggregates all the distinc root nodes with which unknown node was activated with
## W.R.T each root, the unknown node has probability of being somewhere
## To get the probability considering all roots at once, we multiply the probabilities tighether (of pdf for each root node)
## In the end we get pdf of an unknown node being as a particular location
## NOTE: the pdf is actually not pdf since it does not sum up to 1, however we are interested in finding location of an unknown node with the higher probability
## therefore, the normalizing constant is irrelevant
class LocationDistribution:


    def __init__(self, root_locs , distributions):
        if len(root_locs) != len(distributions):
            raise Exception("For every root, there must be a distribution!")

        self.distributions = distributions
        self.root_locs = root_locs
        self.dim = len(root_locs[0])

    def __repr__(self):
        return f"<LocationDistribution| comb_cnt: {len(self.root_locs)}, avg_rt_loc: {np.mean(self.root_locs)}, dim: {self.dim}>"

    def max(self,loc):
        max_ = 0
        for i in range(len(self.root_locs)):
            distance = np.linalg.norm(loc - self.root_locs[i])

            d_ = self.distributions[i].pdf(distance)

            max_ = max(max_,d_)

        return max_


    def pdf(self, loc, log = True):

        loc = np.concatenate([loc,np.zeros(self.dim - len(loc))])

        log_p = 0

        for i in range(len(self.root_locs)):
            distance = np.linalg.norm(loc - self.root_locs[i])


            d_ = self.distributions[i].pdf(distance)

            d_ += 10e-10
            log_p += np.log(d_)

        return np.exp(log_p) if not log else log_p

    def plot(self,range_ = ((-5,-5,0),(5,5,0)), max_ = False):

        a = np.linspace(range_[0][0],range_[1][0],100)
        b = np.linspace(range_[0][1],range_[1][1],100)

        X = np.zeros((len(a),len(b)))
        Y = np.zeros((len(b),len(a)))
        Z = np.zeros((len(a),len(b)))

        for i, x in enumerate(a):
            for j, y in enumerate(b):
                X[i,j] = x
                Y[i,j] = y
                if max_:
                    z = self.max(np.array([x,y,0]))
                else:
                    z = self.pdf(np.array([x,y,0]),log=False)
                Z[i,j] = z

        ax = plt.figure().add_subplot(projection='3d')



        # Plot the 3D surface
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.3, rstride=3, cstride=3,
                        alpha=0.35,antialiased = True)

        plt.show()



class SensorNode:
    """
    Node of a graph that represents a sensor. Each sensor node can be rooted or not. If sensor node is rooted, then we know its location.
    The 'less' it is rooted the less we are sure of its location.
    """


    def __init__(self,id_,root = 0, location = None):
        """

        :param id_: id of the sensor node
        :param root: ranges from 0 to 1, and signifies how sure are we of the sensors location
        :param location: location of the sensor node
        """
        self.id_ = id_
        self.location = location
        self.root = root

        if root == 1 and location is None:
            raise Exception("If sensor is rooted, we need to know its location")

        self.neighborhood = dict()

    def __repr__(self):

        return f"<SensorNode| id: {self.id_}, loc: {self.location}, rt:{self.root}, nghlen: {len(self.neighborhood)}>"

    def get_neighbors(self):
        """
        :return: returns a dictionary where each key is a neighboring sensor node, and each value is list of:
         1. mean the pressure due to which these two sensors were activated and thus have direct edge
         2. mean size of the outside stimuli due to which they were activated
        """
        return self.neighborhood

    def get_neighbor_nodes(self):
        """

        :return: Only returns neighboring nodes, not the edge values
        """
        return list(self.neighborhood.keys())

    def get_location(self):
        """

        :return: returns location of the sensor node
        """
        if self.location is None: return None
        return np.copy(self.location)

    def add_neighbor(self,node,pressure, max_sep):
        """
        adds a neighboring node - creates an edge between itself and another node
        :param node: the node we want to add to neighborhood
        :param pressure: the pressure at which these to were jointly activated thus are in the neighborhood
        :param max_sep: the size of the outside stimuli due to which the sensor nodes were jointly activated
        :return:
        """

        if node not in self.neighborhood:
            self.neighborhood[node] = np.array([pressure,max_sep])
            node.neighborhood[self] = np.array([pressure,max_sep])
        else:
            edge = (self.neighborhood[node] + np.array([pressure,max_sep])) / 2

            self.neighborhood[node] = edge
            node.neighborhood[self] = edge


    def set_root(self,root, location = None):
        """
        set a sensor node to be root. If root < 1, the location is set to None
        :param root: the value of how much is sensor rooted
        :return:
        """
        if root == 1 and location is None:
            raise Exception("Rooted node must have specified location")

        if root != 1: location = None
        self.root = root
        self.location = location

    def set_location(self,location):
        """
        :param location: the location of a sensor node
        :return:
        """
        self.location = location

    def make_rooted(self,location):
        """
        make sensor node rooted (set self.root = 1). For this a location is necessary
        :param location: the location of the sensor node
        :return:
        """
        self.root = 1
        self.location = location


    def is_root(self):
        """

        :return: returns True of sensor is a root (self.root == 1)
        """
        return self.root == 1

    def __hash__(self):
        return self.id_

    def __eq__(self, other):
        return self.id_ == other.id_



class VicinityGraph:
    """
    The Vicinity Graph holds all sensor nodes inside it, and server as an underlying data structure for managing and storing
    the proximity information.

    If two sensors are activated, we create an edge in the graph between their corresponding sensor nodes. The edge weigh is then a tuple
    of how much pressure happened when the sensors were activated, and what size was the outside stimuli that invoked the activation.
    This is later useful when building probability distributions of where each sensor is located, since edges of a sensor node signify,
    near which other sensors a sensor must be.
    """


    def __init__(self,sensor_cnt,dim):
        """

        :param sensor_cnt: amount of sensors
        :param dim: dimension of locations
        """

        self.node_set = dict()
        self.roots = dict()
        self.dim = dim

        for i in range(sensor_cnt):

            sensor_node = SensorNode(id_ = i)
            self.node_set[i] = sensor_node

    def get_roots(self):
        """

        :return: return all sensor nodes that are roots
        """
        return list(self.roots.values())

    def set_root(self, node : SensorNode, location):
        """
        sets a node to be root
        :param node: node to be added as root
        :param location: set the location of the new root
        :param location: set the location of the new root
        :return:
        """
        if len(location) != self.dim:
            raise Exception(f"[ERROR] location dim is expected to be {self.dim} for sensor node")

        node.make_rooted(location)
        self.roots[node.id_] = node

    def remove_root(self,node : SensorNode):
        """
        removed node from being a root
        :param node: node to be removed as root
        :return:
        """

        node.set_root(0)
        self.roots.pop(node.id_)

    def set_roots(self,ids, locations):
        """
        set a list of sensor nodes as roots
        :param ids: ids of sensors to be set to roots
        :param locations: locations of the sensors that will be set to roots
        :return:
        """

        for ix, id_ in enumerate(ids):

            node = self.node_set[id_]
            self.set_root(node,np.array(locations[ix]))

    def add_neighborhood(self,activated_sensors, pressure_values, MAX_SEP):
        """
        creates a neighborhood inbetween a list of sensors- creates a clique in-between the list of sensors.
        :param activated_sensors: list of ids of activated sensors
        :param pressure_values: pressure value of each activated sensor
        :param MAX_SEP: maximum separation of sensors - the size of outside stimuli
        :return:
        """

        for i in range(len(activated_sensors)):
            a_id = activated_sensors[i]
            node_a = self.node_set[a_id]
            for j in range(i+1,len(activated_sensors)):
                b_id = activated_sensors[j]
                node_b = self.node_set[b_id]

                pressure = min(pressure_values[i],pressure_values[j])

                node_a.add_neighbor(node_b,pressure,MAX_SEP)

    def get_forest(self):
        """

        :return: returns a list of sensors, where for each sensor in the list, it corresponds to a disconnecte dgraph from the graphs
        of the other sensors in the list
        """

        forest = []
        nonvisited_nodes = set(self.node_set.values())

        while len(nonvisited_nodes) > 0:

            root : SensorNode = None
            for e in nonvisited_nodes:
                root = e
                break

            nonvisited_nodes.remove(root)

            q = queue.Queue()
            q.put(root)

            while not q.empty():

                node : SensorNode = q.get()
                neighborhood = node.get_neighbors()

                for node, edge in neighborhood.items():
                    if node in nonvisited_nodes:
                        nonvisited_nodes.remove(node)
                        q.put(node)

            forest.append(root)

        return forest


    def get_node_degree_list(self):
        """

        :return: returns the list of edge degrees for each sensor node
        """

        degree_list = []
        for id_, node in self.node_set.items():

            degree_list.append(len(node.get_neighbors()))

        return degree_list

    def get_nodes(self):
        """
        returns list of all sensor nodes
        :return:
        """

        return list(self.node_set.values())


    def get_nearest(self,node : SensorNode, lim = None):
        """
        finds the nearest node of a specific node within some range. NOTE: For sensors, there might be other sensors that are not their direct neighborhood (direct edge).
        This is because when generating sensor locations, we might generate them incorrectly.
        :param node: node of interest
        :param lim: the limitation of how many other nodes we can check
        :return: returns the nearest node, and the distance to that node
        """
        if lim is None: lim = len(self.node_set)

        loc = node.get_location()

        visited = {node}
        q = queue.Queue()
        for n in node.get_neighbors().keys():
            q.put(n)
            visited.add(n)

        nearest_root = None
        distance = None

        while not q.empty():
            if lim <= 0: break

            neighbor : SensorNode = q.get()

            for n in neighbor.get_neighbors().keys():
                if n not in visited:
                    visited.add(n)
                    q.put(n)

            if neighbor.is_root():

                d_ = np.linalg.norm(loc - neighbor.get_location())
                if distance is None or distance > d_:
                    distance = d_
                    nearest_root = neighbor

            lim -= 1

        return nearest_root, distance




class BayesSpacalAlgo:
    """
    Bayes Spacal Algorithm is an alternative algorithm for determining the locations of sensors
    It is based on:
    1. Determining which sensors were activated, and analysing In case 2 sensors were activated, how distant they are (this is done on the 'training data'). The distance behavior is modelled by a probabilistic distribution.
    2. Given a few sensors that have known positions, and a new sensor that was activated with such sensors; we know due to the previously estimated probabilistic distribution where the new sensor is 'probably located'
    3. We place sensors with unknown positions continually according to 2. in specific locations, and make their locations known, and then repeat with a new set of unknown sensors
    """

    def __init__(self, decider : ActivationDecider, min_sep, max_sep, sensor_cnt,ngh_lim = 2,dim = 3, seed=0):
        """

        :param decider: decider that decides which sensors are activated
        :param min_sep: minimum distance of two sensors
        :param max_sep: maximum distance of two sensors, given that they are jointly activated (so it is also the size of the outside stimuli)
        :param sensor_cnt: amount of sensors
        :param dim: location dimesion
        :param seed: seed for reproducibility
        """

        self.dim = 3
        self.decider = decider
        self.random = random.Random()
        self.random.seed(seed)

        self.min_sep = min_sep
        self.max_sep = max_sep

        self.sensor_cnt = sensor_cnt


        self.ngh_lim = ngh_lim
        self.distance_lim = self.min_sep*0.33



        self.vicinity_graph = VicinityGraph(sensor_cnt,self.dim)


    def get_sensors(self):
        """

        :return: returns list of sensor Node objects
        """
        return self.vicinity_graph.get_nodes()

    def update_sensor_graph(self,sensors):
        """
        Creates new connections in the graph of sensor nodes, where each edge connection signifies that the two sensors were jointly activated
        :param sensors: list of sensor pressures, where index is the sensor id
        :return:
        """


        # determine which sensors were activated
        activated_sensors, pressure_values = self.decider.decide_activated(sensors)

        # if too many sensors were activated, warn the user that either the recording is wrong, or they are using too big outside stimuli
        percent_activated = len(activated_sensors)/self.sensor_cnt
        print(f"{percent_activated}  activated")
        if percent_activated > 0.5:
            print("[[Warning] more than half of the sensors were activated. Check if the recording is correct. This recording frame will be skipped]")
        else:
            ## create connections inbetween all sensors nodes for which the corresponding sensors were activated
            self.vicinity_graph.add_neighborhood(activated_sensors,pressure_values,self.max_sep)


    def set_roots(self, root_ids, root_locs):
        """
        sets a subset of sensor nodes to roots - so sensors whose location is 100% known
        :param root_ids: ids of sensors to set to roots
        :param root_locs: locations of the sensors
        :return:
        """

        self.vicinity_graph.set_roots(root_ids,root_locs)

    def propagate_location_estimates(self,iter_ = None):
        """
        propagates the locations of sensors through the graphs. We start with the neighbors of roots, determine their location, make them roots
        , and then repeat until all sensors are set or itet_ = 0.
        Since there are multiple possible sensors that we can determine the locations of next, we explore this as a depth fist search.
        We set a candidate root node to root, and establish its locatiomn, and then search for new candidate roots. If this 'path' does not lead
        anywhere, due to inccorect setting of sensor locations (so causing sensors to be condensed in one small region), we backtrack, and
        first try to set the locations of other sensors.
        :param iter_: amount of iterations we allow - amount of sensors we want to propagate
        :return:
        """

        if iter_ is not None and iter_ <= 0: return True

        roots = self.vicinity_graph.get_roots()

        ## if there are no roots, there is not a strarting point from which we can propagate
        if len(roots) == 0:
            raise Exception("No roots are present [For location estimation, at leas one root has to be set]")
        elif len(roots) == self.sensor_cnt: # all sensors were set to position
            print("All sensors Set!")
            return True


        # get the possible candidates for next root
        candidate_roots = self.get_ordered_candidate_rootnodes(roots)

        # if there are no candidate roots, due to the above if statements, it implies there are sensors with unknown location left,
        # but there are no roots that we can use to estimate their location
        if len(candidate_roots) == 0:
            print("No candidate roots!")
            return True # TODO: change to False (since no candidate roots are possible, but we still have nodes that are not rooted, it is not good!)

        # Try out making every candidate root an actual root, and recursively repeat
        for candidate_root, neighboring_roots in candidate_roots:

            optimal_location = self.get_new_root_location(candidate_root,neighboring_roots)

            print("next rt: ", candidate_root)
            print("opt loc: ", optimal_location)
            print("ngh rts: ",neighboring_roots)

            self.vicinity_graph.set_root(candidate_root,optimal_location)

            nearest_root, distance = self.vicinity_graph.get_nearest(candidate_root, lim=25)


            if distance > self.distance_lim:

                correct_build = self.propagate_location_estimates(iter_ - 1 if iter_ is not None else None)

                # if the lower level recursion tells us that we correctly built the locations of sensors, we do not try to reconfigure the
                # locations of sensors in a different way. We just terminate.
                if correct_build: return True

            # if adding the new root causes sensors to be too close (closer then possible), we try adding a different node as a root first
            else:
                self.vicinity_graph.remove_root(candidate_root)

        # if adding no root was successful, we return false to notify the upper level of recursion
        return False


    def get_new_root_location(self,candidate_root,neighboring_roots, optimize = True):
        """
        returns the most probable location of a sensors that is goint to be set to root
        :param next_root: the sensor in question
        :param neighboring_roots: neighboring roots, that we know are near the sensor in question
        :param optimize: if True, runs faster, bust the location of the next_root might be determined incorrectly
        :return: location of the next_root
        """

        ## Create The governing location probability distribution of an unknown sensor
        loc_dist = self.build_distribution(candidate_root,neighboring_roots)

        ## Get initial starting location
        best_loc = np.random.rand(self.dim-1)
        best_pdf = -loc_dist.pdf(best_loc,log= True)

        start_locs = self.get_starting_locations(neighboring_roots)

        for start_loc in start_locs:

            start_loc = start_loc[:2]

            ## find the location with the highest value for a location distribution - the location with the highest probability
            ## instead of the actual rpobability, we optimize upon Log(probability), since the probabilities may become very small - numerical stability
            optimal_location = scipy.optimize.minimize(lambda X: -loc_dist.pdf(X, log=True), start_loc)
            optimal_location = optimal_location.x


            pdf_ = -loc_dist.pdf(optimal_location,log = True)

            if pdf_ < best_pdf:
                best_pdf = pdf_
                best_loc = optimal_location


        best_loc = np.concatenate([best_loc,np.zeros(self.dim - len(best_loc))])
        return best_loc

    def build_distribution(self,candidate_root : SensorNode,neighboring_roots):
        """
        builds the probabilistic distriubtion of a sensor location.
        For a given sensor, it has a certain root neighbors, for which we know the locations.
        Since the sensor is close (in distance) to all of them, and we have defined probability distribution of a sensor to another sensor,
        We can combine these for all the roots w.r.t the new sensor. This will yield an adjusted location probability of the new sensor.
        :param neighboring_roots: list all sensors with known location, that were jointly activated with a sensor of interest
        :return: probabilistic distribution of a sensor location, given the neighboring roots
        """

        root_locs = []
        distributions = []
        for root in neighboring_roots:
            pressure = candidate_root.get_neighbors()[root][0]
            print(root)
            s = 0.5743268
            loc = 0.811029
            print("loc : ",loc)
            root_locs.append(root.get_location())
            distributions.append(scipy.stats.lognorm(loc = loc,s = s))

        loc_dist = LocationDistribution(root_locs, distributions)

        return loc_dist

    # NOTE: Choosing a good starting point is vital. This is because in some (nonsmall) regions the pdf will have 0 value
    # and that is because it is so small, it cannot be represented by a float
    # Generally, the the lognorm pdf have parameters such that if x (the input distance) less then the MIN_SEP distance, the pdf will return very low value
    # In this case the gradient will be zero, and optimization will not work. To fix this, we have to find a start point such that, for all pdfs for all roots
    # the distance of the start point to all roots is at least MIN_SEP
    def get_starting_locations(self,neighboring_roots):
        """
        this method determines what will be the starting point of an optimization process for a certain sensor node, where we will look for a sensor location,
        that maximizes the probability of that location.
        It is necessary to choose the starting point wisely, since based on that the gradient ascent optimization will continue.
        :param neighboring_roots: list of neighboring roots
        :return: returns a good estimate for the starting position from which we will then optimize to find the local optimum (highest probability)
        """
        neighboring_roots = list(neighboring_roots)

        corner_point = neighboring_roots[0].get_location()


        mean_loc = np.zeros(len(corner_point))
        for neighbor in neighboring_roots:
            mean_loc += neighbor.get_location()

        mean_loc /= len(neighboring_roots)

        locs = [mean_loc]

        for shift in [np.array([1,0,0]),np.array([0,1,0]),np.array([-1,0,0]),np.array([0,-1,0])]:
            locs.append(mean_loc + shift*self.min_sep)

        return locs



    def get_ordered_candidate_rootnodes(self,roots):
        """
        this method determines which nodes should we estimate the locations of next. It returns a list of all nodes that are connected
        to at least one root node. The list is ordered, so in the first positions, sensor nodes that have many connections to roots appear.
        The ordering is due to the fact that the more connected roots to a sensor node -> the more proximity information of the node -> better location estimation.
        :param roots: list of all roots nodes (sensors with known locations) in the graph
        :return: returns all possible sensors nodes that we can make to be roots
        """
        roots = set(roots)
        nodes = self.vicinity_graph.get_nodes()

        candidate_roots = []

        for node in nodes:
            if node in roots: continue

            neighbor_roots = set(node.get_neighbors().keys()).intersection(roots)

            if len(neighbor_roots) >= len(roots) or len(neighbor_roots) >= self.ngh_lim: # Note: if we have less roots than the amount of roots we want to have, if course we cannot enforce the condition
                candidate_roots.append([node,neighbor_roots])

        ## Note: we want to take care of the nodes that have the most root connections first
        candidate_roots = sorted(candidate_roots,key=lambda x: len(x[1]),reverse=True)

        return candidate_roots




if __name__ == "__main__":

    MIN_SEP = 1
    MAX_SEP = 2

    sensor_positions, time_frames = read_recording("../pygame_model/data.csv")
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
    algo = BayesSpacalAlgo(decider, MIN_SEP, MAX_SEP, len(sensor_positions),
                            seed=1)




    for frame in time_frames:

        algo.update_sensor_graph(frame)




    algo.set_roots(known_pts,known_positions)
    algo.propagate_location_estimates(iter_=3)

    print(algo.get_sensors())
    model =GraphModel()
    print(len(algo.get_sensors()))
    model.plot_graph(algo.get_sensors())


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





