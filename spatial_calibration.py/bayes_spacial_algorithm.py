import numpy as np
import random
from pressure_recording_manager import *
import scipy
from activation_decider import *
from utilities import *
import queue
from GUI_models import *


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

        log_p /= len(self.root_locs)

        return np.exp(log_p) if not log else log_p

    def plot(self,range_ = ((-10,-10,0),(10,10,0)), max_ = False):

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
        ax.plot_surface(X, Y, Z, cmap='copper', lw=0.3, rstride=3, cstride=3,
                        alpha=0.55,antialiased = True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("probability")
        plt.show()




class PEdge:
    """
    Auxiliary class for edge objects
    handles storing edge values
    Example: sensor A, and Sensor B were activated together.  We store information about the activation in the edge object.
    If sensor A, B were already activated, and their edge exists, we just add the newly recorded information to the already known data
    """


    def __init__(self, pressure_a,pressure_b, max_sep):

        self.p_a = pressure_a
        self.p_b = pressure_b

        self.sep = max_sep

        self.e_cnt = 1

    def __repr__(self):

        return f"pa: {self.p_a}, pb: {self.p_b}"

    def add(self,pressure_a,pressure_b,max_sep):

        if pressure_a + pressure_b > self.p_a + self.p_b:

            self.p_a = pressure_a
            self.p_b = pressure_b

        self.sep += max_sep

        self.e_cnt += 1

        return self

    def get_pressure_pair(self):

        return np.array([self.p_a,self.p_b])/1

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

    def add_neighbor(self,node,pressure_pair, max_sep):
        """
        adds a neighboring node - creates an edge between itself and another node
        :param node: the node we want to add to neighborhood
        :param pressure: the pressure at which these to were jointly activated thus are in the neighborhood
        :param max_sep: the size of the outside stimuli due to which the sensor nodes were jointly activated
        :return:
        """

        if node not in self.neighborhood:
            edge = PEdge(pressure_pair[0],pressure_pair[1],max_sep)
            self.neighborhood[node] = edge
            node.neighborhood[self] = edge

        else:

            edge = self.neighborhood[node].add(pressure_pair[0],pressure_pair[1],max_sep)

            self.neighborhood[node] = edge
            node.neighborhood[self] = edge


    def set_root(self,root, location = None):
        """
        set a sensor node to be root. If root < 1, the location is set to None
        :param root: the value of how much is sensor rooted
        :return:
        """
        if root > 0 and location is None:
            raise Exception("Rooted node must have specified location")

        if root == 0: location = None

        self.root = root
        self.location = location

    def set_location(self,location):
        """
        :param location: the location of a sensor node
        :return:
        """
        self.location = location

    def make_rooted(self,location, root_val = 1.0):
        """
        make sensor node rooted (set self.root = 1). For this a location is necessary
        :param root_val:
        :param location: the location of the sensor node
        :return:
        """
        self.root = root_val
        self.location = location


    def get_root_val(self):

        return self.root

    def copy(self):
        s = SensorNode(id_=self.id_,root=self.root,location=np.copy(self.location))
        s.neighborhood = self.neighborhood
        return s


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

    def set_root(self, id_, location, root_val = 1.0):
        """
        sets a node to be root
        :param root_val:
        :param id_: node to be added as root
        :param location: set the location of the new root
        :param location: set the location of the new root
        :return:
        """

        node = self.node_set[id_]
        if root_val == 0:
            print(f"[Warning] root_val = 0; .set_root() will make node {node} NOT rooted")
            self.remove_root(node)
            return

        if len(location) != self.dim:
            raise Exception(f"[ERROR] location dim is expected to be {self.dim} for sensor node")

        node.make_rooted(location, root_val)
        self.roots[node.id_] = node

    def remove_root(self,node : SensorNode):
        """
        removed node from being a root
        :param node: node to be removed as root
        :return:
        """

        node.set_root(0)
        self.roots.pop(node.id_)

    def set_roots(self,ids, locations, root_val = 1.0):
        """
        set a list of sensor nodes as roots
        :param root_val:
        :param ids: ids of sensors to be set to roots
        :param locations: locations of the sensors that will be set to roots
        :return:
        """

        for ix, id_ in enumerate(ids):

            self.set_root(id_,np.array(locations[ix]), root_val)

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

                pressure_pair = (pressure_values[i],pressure_values[j])

                node_a.add_neighbor(node_b,pressure_pair,MAX_SEP)

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

            if neighbor.get_root_val() > 0:

                d_ = np.linalg.norm(loc - neighbor.get_location())
                if distance is None or distance > d_:
                    distance = d_
                    nearest_root = neighbor

            lim -= 1

        return nearest_root, distance




class MultilatSpacialAlgo:
    """
    Bayes Spacal Algorithm is an alternative algorithm for determining the locations of sensors
    It is based on:
    1. Determining which sensors were activated, and analysing In case 2 sensors were activated, how distant they are (this is done on the 'training data'). The distance behavior is modelled by a probabilistic distribution.
    2. Given a few sensors that have known positions, and a new sensor that was activated with such sensors; we know due to the previously estimated probabilistic distribution where the new sensor is 'probably located'
    3. We place sensors with unknown positions continually according to 2. in specific locations, and make their locations known, and then repeat with a new set of unknown sensors
    """

    def __init__(self, decider : ActivationDecider, min_sep, max_sep, sensor_cnt,dim = 2, branch_factor = 3,discount = 0.9, seed=0):
        """

        :param decider: decider that decides which sensors are activated
        :param min_sep: minimum distance of two sensors
        :param max_sep: maximum distance of two sensors, given that they are jointly activated (so it is also the size of the outside stimuli)
        :param sensor_cnt: amount of sensors
        :param dim: location dimesion
        :param seed: seed for reproducibility
        """

        self.dim = dim
        self.true_dim = 3
        self.decider = decider
        self.random = random.Random()
        self.random.seed(seed)
        np.random.seed(seed)

        self.min_sep = min_sep
        self.max_sep = max_sep

        self.sensor_cnt = sensor_cnt

        self.converter = PressureNormalConverter(self.min_sep,self.max_sep)


        self.distance_lim = self.min_sep*0.33
        self.discount = discount
        self.branch_factor = branch_factor

        self.vicinity_graph = VicinityGraph(sensor_cnt,dim = self.true_dim)



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

        if np.max(np.abs(sensors)) <= 10-3:
            print("[Warning] No pressure was exhibited on sensors; time frame will be ignored")

        sensors = normalize_pressures(sensors)


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


    def set_roots(self, root_ids, root_locs, root_val = 1):
        """
        sets a subset of sensor nodes to roots - so sensors whose location is 100% known
        :param root_ids: ids of sensors to set to roots
        :param root_locs: locations of the sensors
        :return:
        """

        self.vicinity_graph.set_roots(root_ids,root_locs, root_val= root_val)


    def print_recording_evaluation(self):

        degrees = self.vicinity_graph.get_node_degree_list()
        print("Minimum sensor sensor relations: ",np.min(degrees))
        print("Maximum sensor sensor relations: ",np.max(degrees))
        print("Mean sensor sensor relations: ",np.mean(degrees))
        print("Variance of sensor sensor relations: ",np.var(degrees))

    def build_map(self,iter_ = None,find_all = False):

        self.print_recording_evaluation()

        if find_all:
            self.best_hypothesis = (None,0)

        self.propagate_location_estimates(iter_=iter_,find_all = find_all, cumulative_prob=0)

        if find_all: return self.best_hypothesis[0], self.best_hypothesis[1]

    def propagate_location_estimates(self,iter_ = None,find_all = False, cumulative_prob = 0):
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

        if iter_ is not None and iter_ <= 0:
            if find_all:
                print("CUMP: ",cumulative_prob)
                if cumulative_prob > self.best_hypothesis[1]:
                    self.best_hypothesis = (self.get_locations_snapshot(),cumulative_prob)

            return True

        roots = self.vicinity_graph.get_roots()

        ## if there are no roots, there is not a strarting point from which we can propagate
        if len(roots) == 0:
            raise Exception("No roots are present [For location estimation, at leas one root has to be set]")
        elif len(roots) == self.sensor_cnt: # all sensors were set to position
            print("All sensors Set!")
            if find_all:
                if cumulative_prob > self.best_hypothesis[1]:
                    self.best_hypothesis = (self.get_locations_snapshot(), cumulative_prob)
            return True


        # get the possible candidates for next root
        candidate_roots = self.get_ordered_ranked_candidate_rootnodes(roots)

        # if there are no candidate roots, due to the above if statements, it implies there are sensors with unknown location left,
        # but there are no roots that we can use to estimate their location
        if len(candidate_roots) == 0:
            print("No candidate roots!")
            return True # TODO: change to False (since no candidate roots are possible, but we still have nodes that are not rooted, it is not good!)


        # Size down the branching factor, only branch out if the candidate roots are do not have enough already present roots to narrow their location
        if len(candidate_roots[0][0].get_neighbors()) < 3:
            candidate_roots = candidate_roots[:1]
        else:
            candidate_roots = candidate_roots[:1]

        # Try out making every candidate root an actual root, and recursively repeat
        for candidate_root, total_rt_val in candidate_roots:

            neighboring_roots = set(self.vicinity_graph.get_roots()).intersection(set(candidate_root.get_neighbors()))
            candidate_locations = self.get_new_root_locations(candidate_root,neighboring_roots,self.branch_factor)

            # If a candidate root node does not have enough neighbors to narrow its location completely, make several branches
            if len(neighboring_roots) < 3:
                candidate_locations = candidate_locations[:(3 -  len(neighboring_roots) + 1)]
            else:
                candidate_locations = candidate_locations[:1]


            # Try out all candidate locations for a given candidate root
            for candidate_location, pdf_val in candidate_locations:


                self.vicinity_graph.set_root(candidate_root.id_,candidate_location,root_val=self.discount*total_rt_val/len(neighboring_roots))

                result = self.propagate_location_estimates(iter_ - 1 if iter_ is not None else None,find_all = find_all,cumulative_prob = cumulative_prob + pdf_val)

                if result and not find_all: return True

                self.vicinity_graph.remove_root(candidate_root)


        # if adding no root was successful, we return false to notify the upper level of recursion
        return True


    def get_new_root_locations(self,candidate_root,neighboring_roots,n, optimize = True):
        """
        returns the most probable location of a sensors that is goint to be set to root
        :param n: amount of candidate root locations
        :param candidate_root: the sensor in question
        :param neighboring_roots: neighboring roots, that we know are near the sensor in question
        :param optimize: if True, runs faster, bust the location of the next_root might be determined incorrectly
        :return: location of the next_root
        """

        ## Create The governing location probability distribution of an unknown sensor
        loc_dist = self.build_distribution(candidate_root,neighboring_roots)


        start_locs = self.get_starting_locations(neighboring_roots, loc_dist,n*3)

        candidate_locations = []

        temp_locs_list = []

        for start_loc in start_locs:

            start_loc = start_loc[:self.dim]

            ## find the location with the highest value for a location distribution - the location with the highest probability
            ## instead of the actual rpobability, we optimize upon Log(probability), since the probabilities may become very small - numerical stability
            optimal_location = scipy.optimize.minimize(lambda X: -loc_dist.pdf(X, log=True), start_loc,options={"gtol":10e-3})
            optimal_location = optimal_location.x
            optimal_location = np.concatenate([optimal_location,np.zeros(self.true_dim - self.dim)])


            pdf_ = -loc_dist.pdf(optimal_location,log = True)


            if len(temp_locs_list) == 0:
                temp_locs_list.append(optimal_location)
                candidate_locations.append((optimal_location,pdf_))

            elif self.closest_dist_to_set(temp_locs_list,optimal_location) > 10e-3:
                temp_locs_list.append(optimal_location)
                candidate_locations.append((optimal_location, pdf_))


        return sorted(candidate_locations, key = lambda x: x[1])[:n]

    def closest_dist_to_set(self,locations_set, key_loc):

        min_dist = np.linalg.norm(locations_set[0] - key_loc)

        for l in locations_set:
            dist = np.linalg.norm(l - key_loc)
            if dist < min_dist: min_dist = dist

        return min_dist

    def build_distribution(self,candidate_root : SensorNode,neighboring_roots):
        """
        builds the probabilistic distriubtion of a sensor location.
        For a given sensor, it has a certain root neighbors, for which we know the locations.
        Since the sensor is close (in distance) to all of them, and we have defined probability distribution of a sensor to another sensor,
        We can combine these for all the roots w.r.t the new sensor. This will yield an adjusted location probability of the new sensor.
        :param neighboring_roots: list all sensors with known location, that were jointly activated with a sensor of interest
        :return: probabilistic distribution of a sensor location, given the neighboring roots
        """

        all_roots = set(self.vicinity_graph.get_roots())

        root_locs = []
        distributions = []
        roots_of_interest = set()
        for root in neighboring_roots:
            roots_of_interest.update(all_roots.intersection(set(root.get_neighbor_nodes())))


            pressure_pair = candidate_root.get_neighbors()[root].get_pressure_pair()

            mean, std = self.converter.convert(pressure_pair[0],pressure_pair[1])

            root_locs.append(root.get_location())

            distributions.append(scipy.stats.norm(loc = mean,scale= std))


        roots_of_interest = roots_of_interest.difference(set(neighboring_roots))
        for rootint in roots_of_interest:
            root_locs.append(rootint.get_location())
            distributions.append(LinearSeparationPenalty(self.min_sep))



        loc_dist = LocationDistribution(root_locs, distributions)

        return loc_dist

    def get_starting_locations(self,neighboring_roots, location_distribution, n):
        """
        this method determines what will be the starting point of an optimization process for a certain sensor node, where we will look for a sensor location,
        that maximizes the probability of that location.
        It is necessary to choose the starting point wisely, since based on that the gradient ascent optimization will continue.
        :param n: pick n best locations
        :param neighboring_roots: list of neighboring roots
        :param location_distribution: probability distribution of the location of the candidate root sensor
        :return: returns a good estimate for the starting position from which we will then optimize to find the local optimum (highest probability)
        """


        center = 0
        for root in neighboring_roots: center += root.get_location()
        center /= len(neighboring_roots)

        initial_locs = self.generate_random_locs(center,n = self.sensor_cnt)

        locs_with_pdf = [ (loc, location_distribution.pdf(loc, log = True)) for loc in initial_locs ]

        top_n = sorted(locs_with_pdf,key=lambda x: x[1], reverse=True)

        top_n = top_n[:n]

        return [loc[0] for loc in top_n]



    def get_ordered_ranked_candidate_rootnodes(self,roots):
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

            sum_rt = 0
            for r in neighbor_roots: sum_rt += r.get_root_val()

            if len(neighbor_roots) > 0:
                candidate_roots.append([node,sum_rt])

        ## Note: we want to take care of the nodes that have the most root connections first
        candidate_roots = sorted(candidate_roots,key=lambda x: x[1],reverse=True)

        return candidate_roots[:self.branch_factor]


    def generate_random_locs(self,center,n = 1):


        locs = [ center + np.concatenate([4*self.max_sep*np.random.random(self.dim) - 2*np.ones(self.dim)*self.max_sep,np.zeros(self.true_dim - self.dim)]) for i in range(n)]

        return locs



    def get_locations_snapshot(self):

        l = [None]*self.sensor_cnt
        nodes = self.vicinity_graph.get_nodes()
        for node in nodes:
            l[node.id_] = node.get_location()

        return l









if __name__ == "__main__":

    MIN_SEP = 1
    MAX_SEP = 2

    sensor_positions, time_frames = read_labeled_recording("../pygame_model/data_2sz.csv")
    known_pts, known_positions = filter_sensors(set([0]), sensor_positions)

    known_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 91,
                 92, 93, 94, 95, 96, 97, 98, 99]

    known_pts = [0,9,90,99]
    known_pts = [0]
   # known_pts = [36]
    known_positions = get_positions_of_sensors(sensor_positions, known_pts)


   # model = SpatialModel(time_frames=time_frames, positions=sensor_positions, static_points=known_pts)

    decider = CountDecider(4)
    algo = MultilatSpacialAlgo(decider, MIN_SEP, MAX_SEP, len(sensor_positions),dim= 2,
                            seed=1)




    for frame in time_frames:

        algo.update_sensor_graph(frame)



    algo.set_roots(known_pts,known_positions)
    #locs, pdf = algo.build_map(iter_=1,find_all=True)
    algo.build_map(iter_=100,find_all=False)

    est_locs = algo.get_locations_snapshot()

    X =  matrix_least_squares(est_locs,sensor_positions)
    #
    est_locs = [X.dot(loc) for loc in est_locs]
    print(np.linalg.det(X[:2,:2]))
    # print(est_locs)
   #  print(algo.get_sensors())
   # print(len(algo.get_sensors()))
     #model =GraphModel()
   #  model.plot_nodes(algo.get_sensors())
   # model.plot_evolution(nodes_ev)

    model = GeneralModel()
    model.plot_points(est_locs)


    # loc_dist = LocationDistribution([np.array([0,0,0]),np.array([0,3,0]),np.array([3,0,0])],[scipy.stats.norm(loc = 2,scale =1),scipy.stats.norm(loc = 2,scale =1),scipy.stats.norm(loc = 2,scale =1)])
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





