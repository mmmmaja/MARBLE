import random
import numpy as np




class Grid:

    def __init__(self, sensor_cnt, area_grid):

        self.grid_list = list(area_grid)

        self.location_to_id = {tuple(p): -1 for p in self.grid_list}
        self.id_to_location ={i: None for i in range(sensor_cnt)}


        random.shuffle(list(area_grid))

        for i in range(sensor_cnt):
            p = tuple(area_grid.pop(-1))
            self.location_to_id[p] = i
            self.id_to_location[i] = p
    def get_sensor_loc(self,id_):
        return self.id_to_location[id_]
    def get_loc_sensor(self,location):
        return self.location_to_id[tuple(location)]
    def lock_sensor(self,id_,location):

        switch_id = self.location_to_id[location]
        switch_location = self.id_to_location[id_]

        self.location_to_id[location] = id_
        self.id_to_location[id_] = location

        self.location_to_id[switch_location] = switch_id
        if switch_id is not None:
            self.id_to_location[switch_id] = switch_location

    def get_neighborhood(self,location, radius):
        neighborhood = set()

        location = np.array(location)
        for loc in self.grid_list:
            if np.linalg.norm(np.array(loc) - location) <= radius:
                neighborhood.add(tuple(location))
        return neighborhood



class DSpacalAlgo:

    def __init__(self, min_sep, max_sep, sensor_cnt, area_grid,dim = 3, seed=0):
        random.seed(seed)

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.grid_list = list(area_grid)

        self.GRID = Grid(sensor_cnt,list(area_grid))

        self.loc_dim = dim
        self.sensor_cnt = sensor_cnt
        self.sensor_locations, self.area_grid = self.init_sensor_positions(self.sensor_cnt, grid_list=self.grid_list)
        self.sensor_beliefs = np.array([1 / sensor_cnt] * sensor_cnt)

        self.known_sensors = set()

    def init_sensor_positions(self,sensor_cnt,grid_list):
        area_grid = {tuple(p): -1 for p in grid_list}
        sensor_locations = []

        grid_list = list(grid_list)
        random.shuffle(grid_list)

        for i in range(sensor_cnt):
            p = tuple(grid_list.pop(-1))
            area_grid[p] = i
            sensor_locations.append(p)

        return sensor_locations, area_grid

    def get_locations(self):
        return np.copy(self.sensor_locations)
    
    def get_beliefs(self):
        return np.copy(self.sensor_beliefs)

    def set_known_sensors(self, ids, locations):

        for i, id_ in enumerate(ids):
            self.known_sensors.add(id_)
            self.sensor_beliefs[id_] = 1

            self.GRID.lock_sensor(id_,locations[i])

        self.reweight_beliefs()

    def reweight_beliefs(self):

        total = 0
        for i in range(len(self.sensor_beliefs)):
            if i in self.known_sensors: continue
            total += self.sensor_beliefs[i]

        for i in range(len(self.sensor_beliefs)):
            if i in self.known_sensors: continue
            self.sensor_beliefs[i] = self.sensor_beliefs[i] / total
    def threshold_split(self, sensors):

        mean_threshold = np.mean(sensors) * 4
        activated_sensors = []

        for i, pressure in enumerate(sensors):

            # TODO: Now we work with defomations which are all negative, thus the '<' comparison and not '>'
            if pressure < mean_threshold:
                activated_sensors.append(i)

        return activated_sensors


    def get_sensors_locations_in_area(self,position,radius, occupied = None):
        if occupied is None: occupied = set()
        loc_set =  set()

        position = np.array(position)

        for loc in self.grid_list:
            if tuple(loc) in occupied: continue

            if np.linalg.norm(np.array(loc) - position) <= radius:
                loc_set.add(tuple(loc))

        return loc_set

    def get_new_locations(self,activated_sensors):

        sensor_requests = []

        for i in range(len(activated_sensors)):
            a_id = activated_sensors[i]
            a_belief = self.sensor_beliefs[a_id]
            a_loc = np.array(self.sensor_locations[a_id])

            possible_locations = set()

            for j in range(len(activated_sensors)):
                if i == j: continue

                b_id = activated_sensors[j]
                b_belief = self.sensor_beliefs[b_id]
                b_loc = np.array(self.sensor_locations[b_id])

                dist = np.linalg.norm(a_loc - b_loc)

                radius = a_belief*(1.1-b_belief)*(self.max_sep)*2

                locations = self.GRID.get_neighborhood(b_loc,radius)

                possible_locations = possible_locations.intersection(locations)

            sensor_requests.append([a_id,possible_locations])

        return sensor_requests
    def update_sensor_locations(self, sensors):

        activated_sensors = self.threshold_split(sensors)
        requested_locations = self.get_new_locations(activated_sensors)

        requested_locations = sorted(requested_locations,key= lambda x: len(x[1]))


        locked_locations = set()
        for id_, locations in requested_locations:
            locations = locations - locked_locations
            print(locations)

            if len(locations) == 0:
                locked_locations.add(self.GRID.get_sensor_loc(id_))
                continue

            new_location = random.sample(locations,1)
            self.GRID.lock_sensor(id_,new_location)

            locked_locations.add(new_location)

        self.update_beliefs(activated_sensors)

    def update_beliefs(self, activated_sensors):
        new_beliefs = self.get_beliefs()

        for i in range(len(activated_sensors)):
            a_id = activated_sensors[i]
            a_belief = self.sensor_beliefs[a_id]

            if a_id in self.known_sensors: continue

            cross_belief = 1 - a_belief
            for j in range(len(activated_sensors)):
                if i == j: continue

                b_id = activated_sensors[j]
                b_belief = self.sensor_beliefs[b_id]

                cross_belief += b_belief

            cross_belief /= len(activated_sensors)

            new_beliefs[a_id] = 0.1 * cross_belief + 0.9 * a_belief

        self.sensor_beliefs = new_beliefs
