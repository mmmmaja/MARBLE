import numpy as np
import random
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from spacal_algorithm import *
from spacal_algorithm_2 import *
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

def read_recording(recording_file):

    file = open(recording_file)

    time_frames = []

    sensor_positions = []

    line = file.readline().strip("\n").split('","')
    for p in line:
        sensor_positions.append(np.array(list(map(float, p.strip('"').split(",")))))


    for line in file.readlines():
        time_frames.append(np.array( list(map(float,line.strip("\n").split(",")))) )


    return np.array(sensor_positions), time_frames


def mean_deformation(time_frames, per_n = 1):

    absolute_mean = 0

    means = []

    for i, frame in enumerate(time_frames):

        if i % per_n == 0:
            mean = np.mean(frame)
            means.append(mean)
            print(f"Frame {i}: {mean}")
            absolute_mean += mean

    print(f"Absolute mean: {absolute_mean/len(time_frames)}")
    return absolute_mean, means

def min_deformation(time_frames, per_n = 1):
    absolute_min = 0

    mins = []

    for i, frame in enumerate(time_frames):

        if i % per_n == 0:
            min_ = np.max(frame)
            mins.append(min_)
            print(f"Frame {i} min deform: {min_}")
            absolute_min = max(min_,absolute_min)

    print(f"Absolute min deform: {absolute_min}")
    return absolute_min, mins


def max_deformation(time_frames,per_n = 1):
    absolute_max = 0




    maxs = []

    for i, frame in enumerate(time_frames):

        if i % per_n == 0:
            max_ = np.min(frame)
            maxs.append(max_)
            print(f"Frame {i} max deform: {max_}")
            absolute_max = min(max_,absolute_max)

    print(f"Absolute max deform: {absolute_max}")
    return absolute_max, maxs



def total_pressures(time_frames, mul = 1):
    if len(time_frames) == 0: return None

    pressures = np.zeros(len(time_frames[0]))

    for t in time_frames:
        for i,s in enumerate(t):
            pressures[i] += s

    pressures *= mul

    return pressures


if __name__ == "__main__":

    sensor_pos, time_frames = read_recording("../pygame_model/data.csv")

    print(total_pressures(time_frames))
