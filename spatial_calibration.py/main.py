import random
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from pressure_recording_manager import *
from spacal_algorithm import *
from spacal_algorithm_2 import *
from discrete_spacal_algo import *





if __name__ == "__main__":

    MIN_SEP = 1
    MAX_SEP = 2

    sep_function = ExpSep(MIN_SEP,MAX_SEP)

    sensor_positions, time_frames = read_recording("../pygame_model/data.csv")
    known_pts, known_positions = filter_sensors(set(range(42,62,3)),sensor_positions)

    known_pts = [0,1,2,3,4,5,6,7,8,9,10,19,20,29,30,39,40,49,50,59,60,69,70,79,80,89,90,91,92,93,94,95,96,97,98,99]
    known_positions = get_positions_of_sensors(sensor_positions,known_pts)

    model = SpatialModel(time_frames=time_frames,positions=sensor_positions,static_points=known_pts)

    algo = SpacalAlgo(sep_function,len(sensor_positions),((0,9),(0,9),(0,0)))
    #algo = SpacalAlgoTest(MIN_SEP,MAX_SEP,len(sensor_positions),area_range=((0,9),(0,9),(0,0)),seed=1)
    #algo = DSpacalAlgo(MIN_SEP,MAX_SEP,len(sensor_positions),sensor_positions)
    algo.set_known_sensors(known_pts,known_positions)

    compare_location_estimates(sensor_positions,algo.get_locations())


    point_ev = [sensor_positions]
    #belief_ev = [algo.get_beliefs()]

    for epoch in range(5):
        point_ev.append(algo.get_locations())
        #belief_ev.append(algo.get_beliefs())
        for frame in time_frames:

            algo.update_sensor_locations(frame)


   # algo.correct_sensor_locations()

    point_ev.append(algo.get_locations())
    #belief_ev.append(algo.get_beliefs())

    model.plot_evolution(point_ev,None)
    compare_location_estimates(sensor_positions, algo.get_locations())



