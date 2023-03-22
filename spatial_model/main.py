import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


UNIT = 40



class SpatialModel:


    def  __init__(self, root_point = (0,0,0)):

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(projection = "3d")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.figure.suptitle("Pointcloud of estimated points")


        self.points_evolution = []
        self.belief_ev = []
        self.time_step = 0
        self.root_point = np.array(root_point)


    def point_to_color(self,id_):

        np.random.seed(id_)

        color = np.round(np.random.rand(3)*255).astype(int)

        return "#" + "".join(f'{i:02X}' for i in color)



    def plot_points(self,points, beliefs = None):

        points_ = np.zeros((len(points),3))
        colors_ = [""]*(len(points))


        self.ax.scatter(self.root_point[0],self.root_point[1],self.root_point[2],marker="D",color = "black", s = 80)

        for i, point in enumerate(points):

            if beliefs is not None:
                color = (np.round(np.array( [255]*3 )*beliefs[i])).astype(int)
                color = "#" + "".join(f"{i:02X}" for i in color)
            else:
                color = self.point_to_color(i)

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

        self.time_step = min(len(self.points_evolution)-1, self.time_step+1)
        self.plot_time_step(self.time_step)

    def prev(self,event):

        self.time_step = max(0,self.time_step-1)
        self.plot_time_step(self.time_step)



    def plot_time_step(self,time_step):
        self.ax.cla()

        beliefs = None
        if self.belief_ev is not None:
            beliefs = self.belief_ev[self.time_step]

        self.plot_points(self.points_evolution[self.time_step],beliefs = beliefs)





if __name__ == "__main__":

    model = SpatialModel()

    three_pts = []
    for t in range(10):
        three_pts.append([ np.array([t*3 - 10, np.random.rand()*10,1]), np.array([0.5*t]*3), np.array([t**2- 4]*3)])

    print(three_pts)
    #model.plot_points(three_pts[0])
    model.plot_evolution(three_pts)



