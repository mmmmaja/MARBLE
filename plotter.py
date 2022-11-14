from matplotlib.widgets import Button
import numpy as np
import matplotlib.pyplot as plt
from hardcoded_data import triangulation_back_bottom, triangulation_front, triangulation_back_top
from data_manager import Data

# time step in ms with which the plot is updated. Must be multiple of the recording time step (50ms)
TIME_STEP = 500

# Choose which patch will be visible
patch_display = [True, False, True]

FORCE_LIMIT = 2

colorTable = [
    [0, 0, 255], [0, 128, 255], [0, 255, 255], [0, 255, 128], [0, 255, 0],
    [128, 255, 0], [255, 255, 0], [255, 128, 0], [255, 0, 0]
]


class Plot:

    def __init__(self, time_recording):
        # Window of visualization consisting of two subplots
        self.window = None
        # Subplot with the patch to be displayed
        self.patch = None
        # Subplot with the orthosis and arm angles to be displayed
        self.angle_subplot = None
        # Pointers updated on the angle subplot over time
        self.point_arm, self.point_orthosis = None, None

        self.level_labels = np.linspace(0, FORCE_LIMIT, 20)

        self.time_recording = time_recording

        # FIXME Not sure here
        self.real_time, self.real_angle_arm, self.real_angle_orthosis = [], [], []

        # Create window and subplots
        self.initiate()

    def initiate(self):
        # Switch on interactive plot
        plt.ion()
        plt.style.use('dark_background')
        
        # Create window for two subplots
        self.window = plt.figure(figsize=(18, 8))

        # Choose a path to be added as a subplot
        if patch_display[0]:
            self.patch = self.display_front_patch()
        elif patch_display[1]:
            self.patch = self.display_back_up_patch()
        else:
            self.patch = self.display_back_bottom_patch()
        # Add second subplot
        self.angle_subplot, self.point_arm, self.point_orthosis = self.initiate_angle_subplot()

        axes = plt.axes([0.81, 0.000001, 0.1, 0.075])
        button = Button(axes, 'Add', color="yellow")
        # button.on_clicked(add)
        plt.show()

    def update_pressure_plot(self, _pressure_front, _pressure_back_top, _pressure_back_bottom):
        """
        :param _pressure_front: one batch of front pressure data from recording real time
        :param _pressure_back_top: one batch of back top pressure data from recording real time
        :param _pressure_back_bottom: one batch of back bottom pressure data from recording real time
        """
        # _pressure_front[_pressure_front > FORCE_LIMIT] = FORCE_LIMIT
        # _pressure_back_top[_pressure_back_top > FORCE_LIMIT] = FORCE_LIMIT
        # _pressure_back_bottom[_pressure_back_bottom > FORCE_LIMIT] = FORCE_LIMIT
        if patch_display[0]:
            self.patch.tricontourf(
                triangulation_front,
                _pressure_front,
                self.level_labels,
                cmap='jet')

        elif patch_display[1]:
            self.patch.tricontourf(
                triangulation_back_top,
                _pressure_back_top,
                self.level_labels,
                cmap='jet')

        else:
            self.patch.tricontourf(
                triangulation_back_bottom,
                _pressure_back_bottom,
                self.level_labels,
                cmap='jet')

        plt.draw()

    def update_angle_plot(self, _time, _angle_arm, _angle_orthosis):
        """
        :param _time: second of the simulation
        :param _angle_arm: current angle of the arm (linearized)
        :param _angle_orthosis: current angle of the arm (linearized)
        """
        # TODO show all check it
        self.real_time.append(_time)
        self.real_angle_arm.append(_angle_arm)
        self.real_angle_orthosis.append(_angle_orthosis)
        self.angle_subplot.plot(self.real_time, self.real_angle_arm, linewidth='3', color='#76e6b0')
        self.angle_subplot.plot(self.real_time, self.real_angle_orthosis, linewidth='3', color='#7953e6')

        self.angle_subplot.plot([_time], [_angle_arm], linewidth='3', color='#76e6b0')
        self.angle_subplot.plot([_time], [_angle_orthosis], linewidth='3', color='#7953e6')

        # draw the next point indicating the arm angle
        self.point_arm.set_data(_time, _angle_arm)
        # draw the next point indicating the orthosis angle
        self.point_orthosis.set_data(_time, _angle_orthosis)

        plt.draw()

    def initiate_angle_subplot(self):
        """
        Create subplot for arm and orthosis angles
        :return: subplot, arm and orthosis indicators that will be updated over time
        """

        subplot = self.window.add_subplot(1, 2, 2)
        subplot.set_ylim([0, 100])
        subplot.set_xlim([0, self.time_recording])
        # mark arm angle at time zero
        point_arm, = subplot.plot(
            0, 0,
            linestyle='none',
            markerfacecolor='#83ffc3',
            marker="o",
            markeredgecolor='#ceffe0',
            markersize=10,
            zorder=9)
        # mark orthosis angle at time zero
        point_orthosis, = subplot.plot(
            0, 0,
            linestyle='none',
            markerfacecolor='#875bff',
            marker="o",
            markeredgecolor='#cec2ff',
            markersize=10,
            zorder=10)
        subplot.set_yticks(np.arange(0, 100, step=5))  # Set label locations.
        subplot.set_xlabel('Time in [secs]', fontsize=12)
        subplot.set_ylabel('Angle [degrees]', fontsize=12)
        subplot.tick_params(which='major', width=0.75, length=2.5, labelsize=12)
        subplot.xaxis.grid(True, which='major')
        subplot.yaxis.grid(True, which='major')

        return subplot, point_arm, point_orthosis

    def display_front_patch(self):

        # 2D front triangulation and heatmap of the front array
        front_figure = self.window.add_subplot(1, 2, 1)
        front_figure.set_title(
            'Pressure plot - Artificial Skin Array (Ventral). Timestep: %i ms' % TIME_STEP
        )
        front_figure.set_xlabel('Lateral-Medial [mm]', fontsize=12)
        front_figure.set_ylabel('Superior-Inferior [mm]', fontsize=12)
        front_figure.invert_yaxis()
        front_figure.triplot(triangulation_front, '-k')
        front_figure.tick_params(which='major', width=0.75, length=2.5, labelsize=12)

        _pressure_front = np.zeros(shape=48)
        tcf = front_figure.tricontourf(
            triangulation_front,
            _pressure_front,
            self.level_labels,
            # norm=plt.normalize(vmax=2, vmin=0),
            cmap='jet')

        # add color bar
        color_bar = plt.colorbar(tcf, ax=front_figure, ticks=self.level_labels)
        color_bar.ax.set_yticklabels(["%1.1f" % y for y in self.level_labels], fontsize=12)

        return front_figure

    def display_back_up_patch(self):
        # 2D back top triangulation and heatmap of the back top array
        ventral_top_figure = self.window.add_subplot(1, 1, 1)
        ventral_top_figure.set_title(
            'Pressure plot - Artificial Skin Array (Dorsal, Top). Timestep: %i ms' % TIME_STEP
        )
        ventral_top_figure.set_xlabel('Lateral-Medial [mm]', fontsize=12)
        ventral_top_figure.set_ylabel('Superior-Inferior [mm]', fontsize=12)
        ventral_top_figure.invert_yaxis()
        ventral_top_figure.triplot(triangulation_back_top, '-k')
        ventral_top_figure.tick_params(which='major', width=0.75, length=2.5, labelsize=12)
        _pressure_back_top = np.zeros(shape=22)
        tcf = ventral_top_figure.tricontourf(
            triangulation_back_top,
            _pressure_back_top,
            self.level_labels,
            cmap='jet')

        # add color bar
        color_bar = plt.colorbar(tcf, ax=ventral_top_figure, ticks=self.level_labels)
        color_bar.ax.set_yticklabels(["%1.1f" % y for y in self.level_labels], fontsize=12)

        return ventral_top_figure

    def display_back_bottom_patch(self):
        # 2D  back top triangulation and heatmap of the back top array
        ventral_bottom_figure = self.window.add_subplot(1, 1, 1)
        ventral_bottom_figure.set_title(
            'Pressure plot - Artificial Skin Array (Dorsal, Bottom). Timestep: %i ms' % TIME_STEP
        )
        ventral_bottom_figure.set_xlabel('Lateral-Medial [mm]', fontsize=12)
        ventral_bottom_figure.set_ylabel('Superior-Inferior [mm]', fontsize=12)
        ventral_bottom_figure.invert_yaxis()
        ventral_bottom_figure.triplot(triangulation_back_bottom, '-k')
        ventral_bottom_figure.tick_params(which='major', width=0.75, length=2.5, labelsize=12)
        _pressure_back_bottom = np.zeros(shape=10)
        tcf = ventral_bottom_figure.tricontourf(
            triangulation_back_bottom,
            _pressure_back_bottom,
            self.level_labels,
            cmap='jet')
        color_bar = plt.colorbar(tcf, ax=ventral_bottom_figure, ticks=self.level_labels)
        color_bar.ax.set_yticklabels(["%1.1f" % y for y in self.level_labels], fontsize=12)

        return ventral_bottom_figure

    def display_data(self, path):
        """
        Read data saved when recording has ended
        :param path: Path to the xls data file from recording
        """
        data = Data(path)
        self.angle_subplot.plot(data.time, data.angle_arm, linewidth='3', color='#76e6b0')
        self.angle_subplot.plot(data.time, data.angle_orthosis, linewidth='3', color='#7953e6')

        plot_step = int(TIME_STEP / data.time_step_recording)
        for k in range(1, data.num_measurements, plot_step):
            self.update_pressure_plot(data.pressure_front[:, k], data.pressure_back_top[:, k], data.pressure_back_bottom[:, k])
            self.point_arm.set_data(data.time[k], data.angle_arm[k])
            self.point_orthosis.set_data(data.time[k], data.angle_orthosis[k])
            plt.pause(0.5)

