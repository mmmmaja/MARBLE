from pressure_recording_manager import *
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
from activation_decider import *
from bayes_spacial_algorithm import *

## Graphical models for either final product or for showing results
class BeliefModel:
    """
    SpatialModel plots locations of sensors in 3D space; optionally, beliefs of each sensor.
    Additionally, we can plot multiple iterations of an algorithm estimating sensor locations
    The SpatialModel also plots beliefs of the locations of the sensors
    """

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
            print(pressures)
            positions = positions

            self.p_ax = self.figure.add_subplot(222, projection="3d")
            self.p_ax.set_xlabel("X")
            self.p_ax.set_ylabel("Y")
            self.p_ax.set_zlabel("Pressure")
            self.p_ax.title.set_text("Pressure distribution of the recording")
            self.plot_pressures(positions,pressures)

    def plot_pressures(self,positions, pressures):
        """
        Plots the pressures for given positions
        :param positions: positions of pressures
        :param pressures: pressure values
        :return:
        """

        points_ = np.zeros((len(positions), 3))
        colors_ = [""] * (len(positions))
        for i, point in enumerate(positions):

            color = self.point_to_color(i)

            points_[i] = np.array(point)
            colors_[i] = color

        self.p_ax.bar3d(points_[:, 0], points_[:, 1], points_[:, 2], [0.5]*len(positions), [0.5]*len(positions), pressures, color=colors_)


    def point_to_color(self,id_):


        np.random.seed(id_)

        color = np.round(np.random.rand(3)*255).astype(int)

        return "#" + "".join(f'{i:02X}' for i in color)



    def plot_points(self,points, beliefs = None, static_points = ()):
        """
        plots points in space
        :param points: array of point locations
        :param beliefs: 0-1 values signifying the belief a point is in a specified location
        :param static_points: points that are static, so should have belief 1
        :return:
        """
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
        """
        plots the evolution of point locations
        :param points_ev: list of lists of locations of points
        :param beliefs_ev: list of lists of beliefs of points
        :return:
        """

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



class GraphModel:
    """
    GraphModel plots sensor locations along with edges to other sensors that were activated jointly with them
    """

    def __init__(self, time_frames=(), positions=(), hide_unk  =True):

        self.figure = plt.figure(figsize=(10, 8))
        self.ax = self.figure.add_subplot(111, projection="3d")


        self.hide_unk = hide_unk

        plt.connect('button_press_event', self.on_click)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.title.set_text("Graph of estimated sensor locations")

        self.nodes_evolution = []
        self.time_step = 0

        if len(time_frames) != 0 and len(time_frames[0]) == len(positions):
            pressures = total_pressures(time_frames=time_frames, mul=1 / len(time_frames))
            positions = positions

            self.p_ax = self.figure.add_subplot(222, projection="3d")
            self.p_ax.set_xlabel("X")
            self.p_ax.set_ylabel("Y")
            self.p_ax.set_zlabel("Pressure")
            self.p_ax.title.set_text("Pressure distribution of the recording")
            self.plot_pressures(positions, pressures)

    def plot_pressures(self, positions, pressures):
        """
        Plots the pressures for given positions
        :param positions: positions of pressures
        :param pressures: pressure values
        :return:
        """

        points_ = np.zeros((len(positions), 3))
        colors_ = [""] * (len(positions))
        for i, point in enumerate(positions):
            color = self.point_to_color(i)

            points_[i] = np.array(point)
            colors_[i] = color

        self.p_ax.bar3d(points_[:, 0], points_[:, 1], points_[:, 2], [0.5] * len(positions), [0.5] * len(positions),
                        pressures * -2, color=colors_)

    def point_to_color(self, id_):

        np.random.seed(id_)

        color = np.round(np.random.rand(3) * 255).astype(int)

        return "#" + "".join(f'{i:02X}' for i in color)

    def plot_nodes(self, nodes):

        visited_nodes = set()
        visited_pairs = set()

        for node in nodes:

            if node.get_location() is None: continue
            if self.hide_unk and node.get_root_val() == 0: continue

            neighbors = node.get_neighbors().items()

            x = [node.get_location()[0], 0]
            y = [node.get_location()[1], 0]
            z = [node.get_location()[2], 0]



            self.ax.scatter(x[0],y[0],z[0], marker = 'o',s = 20, c = self.point_to_color(node.id_))
            self.ax.text(x[0] - 0.01, y[0] + 0.01,z[0] + 0.01, node.id_, size = 6)


            for neighbor, edge in neighbors:

                if neighbor.get_location() is None: continue
                if (neighbor.id_,node.id_) in visited_pairs: continue

                visited_pairs.add((node.id_,neighbor.id_))

                x[1] = neighbor.get_location()[0]
                y[1] = neighbor.get_location()[1]
                z[1] = neighbor.get_location()[2]

                self.ax.plot(x, y, z, linestyle="--", lw=1)


            visited_nodes.add(node.id_)

        plt.show()

    def plot_evolution(self, nodes_ev):


        self.nodes_evolution = nodes_ev
        self.time_step = 0

        axprev = self.figure.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = self.figure.add_axes([0.92, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)

        self.plot_time_step(self.time_step)

    def next(self, event):

        self.time_step = (self.time_step + 1) % len(self.nodes_evolution)
        self.plot_time_step(self.time_step)

    def prev(self, event):

        if self.time_step == 0:
            self.time_step = len(self.nodes_evolution) - 1
        else:
            self.time_step -= 1

        self.plot_time_step(self.time_step)


    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            pass

    def plot_time_step(self, time_step):
        self.ax.cla()

        self.ax.title.set_text(f"TIME STEP: {time_step}")


        self.plot_nodes(self.nodes_evolution[self.time_step])



class GeneralModel:
    """
    General Model plots locations of sensors in 3D space.
    Additionally, we can plot multiple iterations of an algorithm estimating sensor locations
    The SpatialModel also plots beliefs of the locations of the sensors
    """

    def  __init__(self,time_frames = (), positions = (), static_points = ()):

        self.figure = plt.figure(figsize=(10,6))
        self.ax = self.figure.add_subplot(projection = "3d")

        plt.connect('button_press_event', self.on_click)


        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.title.set_text("Pointcloud of estimated points")


        self.points_evolution = []
        self.time_step = 0
        self.static_points = set(static_points)



        if len(time_frames) != 0 and len(time_frames[0]) == len(positions):
            pressures = total_pressures(time_frames=time_frames, mul=1/len(time_frames))
            positions = positions

            self.p_ax = self.figure.add_subplot(222, projection="3d")
            self.p_ax.set_xlabel("X")
            self.p_ax.set_ylabel("Y")
            self.p_ax.set_zlabel("Pressure")
            self.p_ax.title.set_text("Pressure distribution of the recording")
            self.plot_pressures(positions,pressures)

    def plot_pressures(self,positions, pressures):
        """
        Plots the pressures for given positions
        :param positions: positions of pressures
        :param pressures: pressure values
        :return:
        """

        points_ = np.zeros((len(positions), 3))
        colors_ = [""] * (len(positions))
        for i, point in enumerate(positions):

            color = self.point_to_color(i)

            points_[i] = np.array(point)
            colors_[i] = color

        self.p_ax.bar3d(points_[:, 0], points_[:, 1], points_[:, 2], [0.5]*len(positions), [0.5]*len(positions), pressures*-2, color=colors_)


    def point_to_color(self,id_):


        np.random.seed(id_)

        color = np.round(np.random.rand(3)*255).astype(int)

        return "#" + "".join(f'{i:02X}' for i in color)



    def plot_points(self,points, static_points = ()):
        """
        plots points in space
        :param points: array of point locations
        :param beliefs: 0-1 values signifying the belief a point is in a specified location
        :param static_points: points that are static, so should have belief 1
        :return:
        """

        s_pts = np.zeros((len(static_points),3))
        sx = 0

        for i, point in enumerate(points):
            if point is None: continue

            if i not in static_points:
                marker = "o"
                size = 50
                self.ax.scatter(point[0], point[1], point[2],color = self.point_to_color(i),marker=marker, s= size)
                self.ax.text(point[0], point[1], point[2] + 0.05, str(i),size=11)

            else:
                s_pts[sx,:] = point
                sx += 1

        self.ax.scatter(s_pts[:,0],s_pts[:,1],s_pts[:,2],color = "black", marker="X")

        plt.show()


    def plot_evolution(self,points_ev):
        """
        plots the evolution of point locations
        :param points_ev: list of lists of locations of points
        :param beliefs_ev: list of lists of beliefs of points
        :return:
        """

        self.points_evolution = points_ev
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


    def on_click(self,event):
        if event.button is MouseButton.LEFT:
            pass



    def plot_time_step(self,time_step):
        self.ax.cla()

        self.ax.title.set_text(f"TIME STEP: {time_step}")

        self.plot_points(self.points_evolution[self.time_step],static_points=self.static_points)



class AlgorithmModel:


    def __init__(self):

        self.algo = None
        self.known_locs = False
        self.recording = None



        self.window = tk.Tk()
        self.window.geometry("750x750")
        self.menu = tk.Menu(master = self.window)


        self.menu.add_command(label="Create Algo", command=self.create_algo)
        self.menu.add_command(label="Set Locations",command=self.set_locations)
        self.menu.add_command(label="Recording",command=self.load_recording)
        self.menu.add_command(label="Run Algorithm",command=self.run_algorithm)


        self.window.config(menu=self.menu)
        self.window.mainloop()

    def clear(self):

        l = self.window.grid_slaves()

        for _ in l: _.destroy()


    def create_algo(self):

        self.clear()

        tk.Label(self.window,text="Minimum Separation").grid(row = 0, column=0)
        tk.Label(self.window,text="Stimuli Size").grid(row = 0, column=1)
        tk.Label(self.window,text="Sensor Count").grid(row = 0, column=2)
        tk.Label(self.window,text="Seed").grid(row = 0, column=3)
        min_sep = tk.Entry(self.window)
        max_sep = tk.Entry(self.window)
        sen_cnt = tk.Entry(self.window)
        seed = tk.Entry(self.window)

        min_sep.grid(row= 1,column=0)
        max_sep.grid(row= 1,column=1)
        sen_cnt.grid(row= 1,column=2)
        seed.grid(row= 1,column=3)

        seed.insert(-1,"1")

        def exe(s):
            s.algo = BayesSpacalAlgo(CountDecider(4),min_sep = float(min_sep.get()), max_sep = float(max_sep.get()),sensor_cnt = int(sen_cnt.get()),seed = int(seed.get()) )
            tk.Label(self.window,text="CREATED!").grid(row = 3)

        tk.Button(text="Create",command= lambda :exe(self) ).grid(row = 2)

    def set_locations(self):
        self.clear()

        if self.algo is None:
            tk.Label(self.window,text="[ERROR] first create the algorithm").grid()
            return



        cnt = simpledialog.askinteger(title="",prompt="Fill amount of sensors that are known")

        ids = []
        locs = []
        for i in range(cnt):

            ids.append(tk.Entry(self.window))
            locs.append(tk.Entry(self.window))

            tk.Label(text = "ID:").grid(row=2*i,column=0)
            tk.Label(text = "Location {format = \"x,y,z\"}:").grid(row=2*i,column=1)
            ids[i].grid(row = 2*i+1,column=0)
            locs[i].grid(row = 2*i+1,column=1)

        def exe(s):
            s.store_locs(ids,locs)
            tk.Label(self.window,text = "Set!").grid(row=2*cnt+1)
            s.known_locs = True


        tk.Button(self.window,text="Set",command=lambda :exe(self)).grid(row=2*cnt)


    def store_locs(self,ids,locs):

        ids = [int(i.get()) for i in ids]
        locs = [ np.array(list(map(lambda x: float(x), loc.get().strip().split(",")))) for loc in locs]

        self.algo.set_roots(ids,locs)

    def load_recording(self):

        self.clear()

        file_str = simpledialog.askstring(title="Recording file", prompt="Write recording source file")

        true_locs, self.recording= read_labeled_recording(file_str)
        tk.Label(self.window,text="Successful").grid()

    def run_algorithm(self):
        self.clear()
        if self.algo is None or self.recording is None or self.known_locs is False:
            if self.algo is None:
                tk.Label(self.window,text="[ERROR] algorithm not created").grid()
            if self.recording is None:
                tk.Label(self.window,text="[ERROR] recording not parsed").grid()
            if self.known_locs is False:
                tk.Label(self.window,text="[ERROR] At lest one sensor has to have known location").grid()
            return

        for frame in self.recording:

            self.algo.update_sensor_graph(frame)

        self.algo.propagate_location_estimates(iter_ = len(self.recording[0]))
        locations= np.array(self.algo.get_locations_snapshot())

        figure = Figure(figsize=(7.5,7.5))
        plot1 = figure.add_subplot(projection = "3d")

        for i, point in enumerate(locations):
            if point is None: continue
            plot1.scatter(point[0], point[1], point[2], color=self.id_to_color(i), marker="o", s=30)
            plot1.text(point[0], point[1], point[2]+0.01,str(i),size=9)

        canvas = FigureCanvasTkAgg(figure,
                                   master=self.window)
        canvas.draw()

        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().grid()

        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas,
                                       self.window)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().grid()


    def id_to_color(self,id_):

        rng = random.Random()
        rng.seed(id_)


        color =[ rng.randint(0,255) for i in range(3)]

        return "#" + "".join(f'{i:02X}' for i in color)



if __name__ == "__main__":

    am = AlgorithmModel()






