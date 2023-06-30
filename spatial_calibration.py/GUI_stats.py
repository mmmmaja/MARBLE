import numpy as np
import matplotlib.pyplot as plt
from pressure_recording_manager import *
import scipy
from utilities import *
from activation_decider import *





## All plots analyzing pressure distributions; behavior of activationDeciders, behavior of pressure to distance relations


def distance_distribution(sensor_positions,time_frames,decider,MAX_SEP,include_pressure = False, normalize = False):
    """

    :param sensor_positions: position of each sensor
    :param time_frames: all the time frames of the sensor recordings
    :param decider: the specified decider - for example CountDecider
    :param MAX_SEP: maximum separation of jointly activated sensors (determined by the outside stimuli size)
    :param include_pressure: option to return pressure values
    :return: returns distances of the pairs of activated sensors, total count of the activated sensors, count of pairs sensors below the MAX_SEP distance (optionally, pressure values)
    """
    activated_distances = []

    total_activated_cnt = 0

    below_max_sep_cnt = 0

    pressure_values = []

    for frame in time_frames:

        if normalize: frame = normalize_pressures(frame)

        activated, pressures = decider.decide_activated(frame)

        total_activated_cnt += len(activated)
        for i in range(len(activated)):
            for j in range(len(activated)):
                if i == j: continue

                distance = np.linalg.norm(sensor_positions[activated[i]] - sensor_positions[activated[j]])
                if distance <= MAX_SEP: below_max_sep_cnt += 1
                activated_distances.append(distance)
                pressure_values.append((pressures[i],pressures[j]))

    if include_pressure:
        return activated_distances,pressure_values, total_activated_cnt, below_max_sep_cnt
    else:
        return activated_distances, total_activated_cnt, below_max_sep_cnt

def plot_pressure_expdistribution(activated_distances,percent_activated,percent_under_max_sep,MIN_SEP):
    """
    1.Plots the histogram of distances of the pairs of activated sensors
    2. Fits (using mle) exponential distribution to the distance hist
    :param activated_distances: distances of activated pairs
    :param percent_activated: percent of sensors activated
    :param percent_under_max_sep: percent of pairs that were activated, and their distances were under the MAX_SEP distance
    :param MIN_SEP: minimum separation of sensors
    :return:
    """

    mle_lambda = 1/(np.mean(activated_distances) - MIN_SEP)
    x_ = np.linspace(min(activated_distances) - MIN_SEP + 10e-3, max(activated_distances) - MIN_SEP, 100)

    y_ = scipy.stats.expon(scale=1 / mle_lambda).pdf(x_)
    for i in range(len(x_)): x_[i] += MIN_SEP

    plt.figure()
    plt.title(f"Pairwise Distance Hist for Threshold {t}-times the Mean; {percent_activated}% Activated")

    plt.hist(activated_distances, density=True, bins=20)
    plt.plot(x_, y_)
    plt.xlabel(f"Under Max separation: {round(percent_under_max_sep, 2)}%")
    plt.axvline(MIN_SEP, color='k', linestyle='dashed')
    plt.axvline(MAX_SEP, color='r', linestyle='dashed')

    plt.show()


def plot_pressure_normdistribution(activated_distances,percent_activated,percent_under_max_sep,MIN_SEP,MAX_SEP):
    """
    1.Plots the histogram of distances of the pairs of activated sensors
    2. Fits (using mle) exponential distribution to the distance hist
    :param activated_distances: distances of activated pairs
    :param percent_activated: percent of sensors activated
    :param percent_under_max_sep: percent of pairs that were activated, and their distances were under the MAX_SEP distance
    :param MIN_SEP: minimum separation of sensors
    :return:
    """
    tmp = [MIN_SEP,MAX_SEP]
    mean = np.mean(activated_distances)
    std = np.std(activated_distances)
    x_ = np.linspace(min(activated_distances) , max(activated_distances), 100)

    y_ = scipy.stats.norm(loc = mean,scale=std).pdf(x_)
    y_minmax = scipy.stats.norm(loc = np.mean(tmp),scale=np.std(tmp)).pdf(x_)
    plt.figure()
    plt.title(f"Pairwise Distance Hist; {percent_activated}% Activated")

    plt.hist(activated_distances, density=True, bins=3)
    plt.plot(x_, y_,label = "mle normal distribution")
    plt.plot(x_, y_minmax, label = "min max inferred normal distribution")
    plt.xlabel(f"Distance [Under Max separation: {round(percent_under_max_sep, 2)}%]")
    plt.ylabel(f"Probability")
    plt.axvline(MIN_SEP, color='k', linestyle='dashed')
    plt.axvline(MAX_SEP, color='r', linestyle='dashed')
    plt.legend(loc = "upper right")

    plt.show()

def plot_pressure_logndistribution(activated_distances,percent_activated,percent_under_max_sep,MIN_SEP):
    """
    1.Plots the histogram of distances of the pairs of activated sensors
    2. Fits (using mle) lognormal distribution to the distance hist
    :param activated_distances: distances of activated pairs
    :param percent_activated: percent of sensors activated
    :param percent_under_max_sep: percent of pairs that were activated, and their distances were under the MAX_SEP distance
    :param MIN_SEP: minimum separation of sensors
    :return:
    """

    mle_mean = sum(np.log(activated_distances))/len(activated_distances)

    mle_std = 0
    for dist in activated_distances:
        mle_std += np.power(np.log(dist) - mle_mean,2)
    mle_std = np.sqrt(mle_std/len(activated_distances))

    print(f"mle mean: {mle_mean}, mle std: {mle_std}")

    x_ = np.linspace(10e-5, max(activated_distances), 100)

    y_ = scipy.stats.lognorm.pdf(x_,loc = mle_mean,s = mle_std)
    plt.figure()
    plt.title(f"Pairwise Distance Hist for Threshold {t}-times the Mean; {round(percent_activated,2)}% Activated")

    plt.hist(activated_distances, density=True, bins=20)
    plt.plot(x_, y_)
    plt.xlabel(f"Under Max separation: {round(percent_under_max_sep, 2)}%")
    plt.axvline(MIN_SEP, color='k', linestyle='dashed')
    plt.axvline(MAX_SEP, color='r', linestyle='dashed')

    plt.show()

def plot_3d_pressure_hist(activated_distances,pressure_values,percent_activated,percent_under_max_sep):
    """
    plot histogram of distances of activated sensors with respect to the pressure value of the pairs
    the distances of will be split into multiple bin sections according to the pressure at which they were activated
    :param activated_distances: distances of activated pairs
    :param pressure_values: pressure values for each pair
    :param percent_activated: percent of sensors activated
    :param percent_under_max_sep:percent of pairs that were activated, and their distances were under the MAX_SEP distance
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(pressure_values)): pressure_values[i] = sum(pressure_values[i])

    hist, xedges, yedges = np.histogram2d(activated_distances, pressure_values, bins=[12,5])

    # for i in range(hist.shape[1]):
    #     hist[:,i] = hist[:,i]/sum(hist[:,i])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 64 bars.
    dx = dy = 0.2 * np.ones_like(zpos)
    dz = hist.ravel()

    plt.title(f"Pairwise Distance Hist; {round(percent_activated,2)}% Activated")
    plt.xlabel(f"Distance [Under Max separation : {round(percent_under_max_sep, 2)}]%")
    plt.ylabel(f"Pressure value")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    plt.show()


def plot_2d_pressure_distance_scatter(activated_distances,pressure_values):


    pressure_values = [sum(pair) for pair in pressure_values]
    plt.scatter(pressure_values,activated_distances)
    plt.title("Scatter of activated pair pressure and distance relation")
    plt.xlabel("Sum of activated pair pressure")
    plt.ylabel("Distance")

    plt.show()


def plot_3d_pressure_distance_scatter(activated_distances,pressure_values):


    pressure_values = np.array(pressure_values)
    fig = plt.figure()
    ax = fig.add_subplot(projection=  "3d")


    ax.scatter(pressure_values[:,0],pressure_values[:,1],activated_distances)

    ax.set_xlabel("sensor X pressure")
    ax.set_ylabel("sensor Y pressure")
    ax.set_zlabel("distance")

    plt.show()


def boxplot_pressure_dist_map(distances, pressure_values, bins= 10,var = False, raw= False):

    pts = []
    for i in range(len(distances)):
        pts.append(np.array([pressure_values[i][0],pressure_values[i][1],distances[i]]))


    x_bins = []
    x_edges = np.linspace(0,max(np.array(pts)[:,0]),bins+1)
    y_edges = np.linspace(0,max(np.array(pts)[:,1]),bins+1)

    pts = sorted(pts, key = lambda x:x[0])


    j = 0
    for i in range(1,len(x_edges)):
        arr = []
        while j < len(pts) and pts[j][0] <= x_edges[i]:
            arr.append(pts[j])
            j += 1
            if j >= len(pts): break
        x_bins.append(arr)


    xy_bins = []
    for x_pts in x_bins:

        x_pts = sorted(x_pts, key= lambda x:x[1])

        j = 0
        x_bin = []
        for i in range(1,len(y_edges)):
            arr = []


            while j< len(x_pts) and x_pts[j][1] <= y_edges[i]:
                arr.append(x_pts[j][2])
                j+= 1


            x_bin.append(arr)

        xy_bins.append(x_bin)


    xy_var = np.zeros(shape=(bins,bins))
    xy_mean = np.zeros(shape= (bins,bins))

    for i in range(bins):
        for j in range(bins):

            if not raw:
                m, v = lognorm_mle(xy_bins[i][j])
                xy_mean[i,j] = m
                xy_var[i,j] = v
            else:
                xy_mean[i,j] = np.mean(xy_bins[i][j])
                xy_var[i,j] = np.var(xy_bins[i][j])


    ## -----------------------------

    scat_pts = []
    for i in range(1,len(x_edges)):
        for j in range(1,len(y_edges)):

            x = (x_edges[i-1] + x_edges[i])/2
            y = (y_edges[j-1] + y_edges[j])/2
            z = xy_var[i-1,j-1] if var else xy_mean[i-1,j-1]
            print(z)
            scat_pts.append(np.array([x,y,z]))

    scat_pts = np.array(scat_pts)


    x = np.linspace(0,1,20)
    y = np.linspace(0,1,20)
    x,y = np.meshgrid(x,y)
    print(y)
    z = []
    for x_ in x[0]:
        k = []
        for y_ in np.linspace(0,1,20):
            print()
            if not var:
                k.append((2-1)*(2-(x_ + y_))/2 + 1)
            else:
                k.append((0.5625 - 0.0625)*(2-(x_ + y_))/2 + 0.0625)
        z.append(k)
    z = np.array(z)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(scat_pts[:, 0], scat_pts[:, 1], scat_pts[:,2],s = 100)
    ax.plot_surface(x,y,z,cmap="autumn")
    ax.set_xlabel("sensor X pressure")
    ax.set_ylabel("sensor Y pressure")
    ax.set_zlabel("distance variance" if var else "mean distance")

    plt.show()



    return xy_mean, xy_var, x_edges, y_edges




if __name__ == "__main__":

    MIN_SEP = 0.5
    MAX_SEP = 2

    print(np.std([MIN_SEP,MAX_SEP]))

    sensor_positions, time_frames = read_labeled_recording("../pygame_model/random_min0,5_max2.csv")

    # We can plot the distance distributions of pairs of activated sensors for different ActivationDeciders
    # From this we can see, which decider is the best for determining relative positions, and additionally, we can use the mle distributions
    # to probabilistically estimate distance of two sensors when we have a new recording
    for t in range(2,20):

        t = t/2

        #decider = MeanThresholdDecider(threshold=t)
        decider = CountDecider(4)

        activated_distances,pressure_values, total_activated_cnt, below_max_sep_cnt = distance_distribution(sensor_positions,time_frames,decider,MAX_SEP,include_pressure=True,normalize = True)
        #print(pressure_values)

        percent_activated = 100*total_activated_cnt/(len(time_frames)*len(time_frames[0]))
        percent_below_max_sep = 100*below_max_sep_cnt/(len(activated_distances))

        #print(max(pressure_values))
        #plot_pressure_expdistribution(activated_distances,percent_activated,percent_below_max_sep,MIN_SEP)
        #plot_pressure_normdistribution(activated_distances,percent_activated,percent_below_max_sep,MIN_SEP,MAX_SEP)
        #plot_3d_pressure_hist(activated_distances,pressure_values,percent_activated,percent_below_max_sep)
        #plot_pressure_logndistribution(activated_distances,percent_activated,percent_below_max_sep,MIN_SEP)
        #plot_3d_pressure_distance_scatter(activated_distances,pressure_values)
        boxplot_pressure_dist_map(activated_distances,pressure_values,bins=10,var=True,raw = True)

        break