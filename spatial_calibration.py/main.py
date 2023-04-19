import random
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from pressure_recording_manager import *
from spacal_algorithm import *
from spacal_test_algo import *
from discrete_spacal_algo import *



class SpatialModel:


    def  __init__(self,time_frames = (), positions = (), static_points = ()):

        self.figure = plt.figure(figsize=(10,6))
        self.ax = self.figure.add_subplot(121,projection = "3d")

        plt.connect('button_press_event', self.on_click)


        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.title.set_text("Pointcloud of estimated points")


        self.points_evolution = []
        self.belief_ev = []
        self.time_step = 0
        self.belief_shown = False
        self.static_points = set(static_points)



        if len(time_frames) != 0 and len(time_frames[0]) == len(positions):
            pressures = total_pressures(time_frames=time_frames, mul=1/len(time_frames))
            positions = positions

            self.p_ax = self.figure.add_subplot(222, projection="3d")
            self.p_ax.set_xlabel("X")
            self.p_ax.set_ylabel("Y")
            self.p_ax.set_zlabel("Pressure")
            self.p_ax.title.set_text("Pressure distribution of the recording")
            self.plot_pressures(self.p_ax,positions,pressures)

    def plot_pressures(self,p_ax,positions, pressures):

        points_ = np.zeros((len(positions), 3))
        colors_ = [""] * (len(positions))
        for i, point in enumerate(positions):

            color = self.point_to_color(i)

            points_[i] = np.array(point)
            colors_[i] = color

        p_ax.bar3d(points_[:, 0], points_[:, 1], points_[:, 2], [0.5]*len(positions), [0.5]*len(positions), pressures*-2, color=colors_)


    def point_to_color(self,id_):


        np.random.seed(id_)

        color = np.round(np.random.rand(3)*255).astype(int)

        return "#" + "".join(f'{i:02X}' for i in color)



    def plot_points(self,points, beliefs = None, static_points = ()):
        if beliefs is None:
            beliefs = [1]*len(points)

        s_pts = np.zeros((len(static_points),3))
        sx = 0

        for i, point in enumerate(points):
            if i not in static_points:
                marker = "o"
                size = 50
                if self.belief_shown:
                    size = 225
                    marker = f"${round(beliefs[i],2)}$"
                self.ax.scatter(point[0], point[1], point[2],color = self.point_to_color(i),marker=marker, s= size)

            else:
                s_pts[sx,:] = point
                sx += 1

        self.ax.scatter(s_pts[:,0],s_pts[:,1],s_pts[:,2],color = "black", marker="X")

        plt.show()


    def plot_evolution(self,points_ev,beliefs_ev = None):

        self.points_evolution = points_ev
        self.belief_ev = beliefs_ev
        self.time_step = 0

        axprev = self.figure.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = self.figure.add_axes([0.92, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)
        axshowbelief = self.figure.add_axes([0.81, 0.05, 0.1, 0.075])
        bbelief = Button(axshowbelief,"Show Belief")
        bbelief.on_clicked(self.show_belief)


        self.plot_time_step(self.time_step)

    def next(self,event):

        self.time_step = (self.time_step+1) % len(self.points_evolution)
        self.plot_time_step(self.time_step)


    def prev(self,event):

        if self.time_step == 0:
            self.time_step = len(self.points_evolution) -1
        else:
            self.time_step -= 1

        self.plot_time_step(self.time_step)

    def show_belief(self,event):
        self.belief_shown = not self.belief_shown

        self.plot_time_step(self.time_step)

    def on_click(self,event):
        if event.button is MouseButton.LEFT:
            pass



    def plot_time_step(self,time_step):
        self.ax.cla()

        self.ax.title.set_text(f"TIME STEP: {time_step}")

        beliefs = None
        if self.belief_ev is not None:
            beliefs = self.belief_ev[self.time_step]

        self.plot_points(self.points_evolution[self.time_step],beliefs = beliefs,static_points=self.static_points)



def compare_location_estimates(l1,l2):
    if len(l1) != len(l2):
        raise "Both arrays of sensor locations must be the same size"

    absolute_difference = 0

    for i in range(len(l1)):
        absolute_difference += np.linalg.norm(l1[i] - l2[i])

    print(f"abs diff: {absolute_difference}, mean diff: {absolute_difference/len(l1)}")

    return absolute_difference


def filter_sensors(to_filter, sensor_positions):

    sp = []
    ids = []
    for i, p in enumerate(sensor_positions):
        if i in to_filter: continue

        sp.append(p)
        ids.append(i)

    return ids, np.array(sp)

def get_positions_of_sensors(sensor_positions,ids):
    p = []
    for i, p_ in enumerate(sensor_positions):
        if i in ids: p.append(p_)
    return p



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



