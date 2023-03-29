import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from pressure_recording_manager import *
from spacal_algorithm import *



class SpatialModel:


    def  __init__(self, root_point = (0,0,0), time_frames = (), positions = (), static_points = ()):

        self.figure = plt.figure(figsize=(10,6))
        self.ax = self.figure.add_subplot(121,projection = "3d")


        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.title.set_text("Pointcloud of estimated points")


        self.points_evolution = []
        self.belief_ev = []
        self.time_step = 0
        self.root_point = np.array(root_point)
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

        points_ = np.zeros((len(points),3))
        colors_ = [""]*(len(points))


        self.ax.scatter(self.root_point[0],self.root_point[1],self.root_point[2],marker="D",color = "black", s = 80)

        for i, point in enumerate(points):

            if beliefs is not None:
                color = (np.round(np.array( [255]*3 )*beliefs[i])).astype(int)
                color = "#" + "".join(f"{i:02X}" for i in color)
            elif i not in static_points:
                color = self.point_to_color(i)
            else:
                color = "black"

            points_[i] = np.array(point)
            colors_[i] = color


        self.ax.scatter(points_[:,0], points_[:,1], points_[:,2], color=colors_)

        plt.show()


    def plot_evolution(self,points_ev,beliefs_ev = None):

        self.points_evolution = points_ev
        self.belief_ev = beliefs_ev
        self.time_step = 0

        axprev = self.figure.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = self.figure.add_axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)

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



if __name__ == "__main__":
    random.seed(0)

    MIN_SEP = 1
    MAX_SEP = 2

    sep_function = ExpSep(MIN_SEP,MAX_SEP)

    sensor_positions, time_frames = read_recording("../pygame_model/data.csv")
    ids, known_positions = filter_sensors(set(range(42,62,3)),sensor_positions)


    model = SpatialModel(time_frames=time_frames,positions=sensor_positions,static_points=ids)

    algo = SpacalAlgo(sep_function,len(sensor_positions),area_range=((0,9),(0,9),(0,0)))
    algo.set_known_sensors(ids,known_positions)

    compare_location_estimates(sensor_positions,algo.get_locations())


    point_ev = [sensor_positions]

    for epoch in range(1):
        point_ev.append(algo.get_locations())
        for frame in time_frames:

            algo.update_activations(frame)
            algo.update_sensor_locations(0.03,n=1,decay=epoch+100)

    point_ev.append(algo.get_locations())

    model.plot_evolution(point_ev)
    compare_location_estimates(sensor_positions, algo.get_locations())



