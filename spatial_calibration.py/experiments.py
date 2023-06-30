import numpy as np
from pressure_recording_manager import *
from spacial_algorithm_belief import BeliefSpacialAlgo
from activation_decider import CountDecider
import matplotlib.pyplot as plt
from GUI_models import GeneralModel
from utilities import matrix_least_squares
from bayes_spacial_algorithm import MultilatSpacialAlgo
class Metrics:

    def __init__(self,linear_transform_independent = False):

        self.lti = linear_transform_independent

    def transform_locs(self,est_locs,true_locs):

        if self.lti:
            X = matrix_least_squares(est_locs,true_locs)
            est_locs = [X.dot(loc) for loc in est_locs]

        return est_locs

    def n_neighborhood(self,loc, all_locs,n):

        dists = []

        for i,loc_ in enumerate(all_locs):

            dists.append((i,np.linalg.norm(loc_ - loc)))

        dists = sorted(dists,key=lambda x:x[1], reverse=False)[:n]
        ids = [i[0] for i in dists]
        return ids



    def avg_absolute_location_error(self,est_locs,true_locs):

        est_locs = self.transform_locs(est_locs,true_locs)


        all_dif = []
        for i in range(len(est_locs)):
            dif = np.linalg.norm(est_locs[i] - true_locs[i])
            all_dif.append(dif)
        return np.mean(all_dif), np.var(all_dif), all_dif

    def relative_location_error(self,est_locs,true_locs):

        est_locs = self.transform_locs(est_locs,true_locs)

        all_dif = []

        for i in range(len(est_locs)):
            for j in range(i+1,len(est_locs)):

                pair_dist_est = np.linalg.norm(est_locs[i] - est_locs[j])
                pair_dist_true = np.linalg.norm(true_locs[i] - true_locs[j])

                all_dif.append(abs(pair_dist_est - pair_dist_true))

        return np.mean(all_dif), np.var(all_dif), all_dif

    def unitarized_relative_location_error(self,est_locs,true_locs):

        est_locs = self.transform_locs(est_locs,true_locs)

        true_unit = np.linalg.norm(true_locs[0] - true_locs[1])
        est_unit = np.linalg.norm(est_locs[0] - est_locs[1])

        all_dif = []

        for i in range(len(est_locs)):
            for j in range(i + 1, len(est_locs)):
                pair_dist_est = np.linalg.norm(est_locs[i] - est_locs[j])/est_unit
                pair_dist_true = np.linalg.norm(true_locs[i] - true_locs[j])/true_unit

                all_dif.append(abs(pair_dist_est - pair_dist_true))

        return np.mean(all_dif), np.var(all_dif), all_dif

    def neighborhood_error(self,est_locs,true_locs, n = 4):

        est_locs = self.transform_locs(est_locs,true_locs)

        all_err = []

        for i in range(len(est_locs)):

            est_neigh = set(self.n_neighborhood(est_locs[i],est_locs,n))
            true_neigh = set(self.n_neighborhood(true_locs[i],true_locs,n))

            all_err.append(len(est_neigh.difference(true_neigh))/n)

        return np.mean(all_err), np.var(all_err), all_err




    def location_dif_stats(self, est_locs_list, true_locs_list, f):
        """

        :param est_locs_list: list of lists of estimated locations (each entry is a list of all sensors estimated locations)
        :param true_locs_list: list of lists of true locations (each entry is a list of all sensors true locations)
        :param f: function that measures error in between estimated and true locations
        :return: mean of error metric; variance of error metric
        """

        all_dif = []

        for i in range(len(est_locs_list)):
            avg_dif, var, all_err = f(est_locs_list[i], true_locs_list[i])
            all_dif.append(avg_dif)

        return np.mean(all_dif), np.var(all_dif)



    def find_relative_dist_outliers(self,est_locs,true_locs):

        errs = [0]*len(est_locs)

        for i in range(len(est_locs)):
            for j in range(i + 1, len(est_locs)):
                pair_dist_est = np.linalg.norm(est_locs[i] - est_locs[j])
                pair_dist_true = np.linalg.norm(true_locs[i] - true_locs[j])

                err = abs(pair_dist_est - pair_dist_true)

                errs[i] += err
                errs[j] += err

        errs = [(i,err) for i,err in enumerate(errs)]


        return sorted(errs,key = lambda x:x[1], reverse=True)



class SettingGenerator:


    def get_known_sensors(self,locations):
        known_pts_brd = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90,
                     91,
                     92, 93, 94, 95, 96, 97, 98, 99]

        known_pts_crn = [0, 9, 90, 99]
        known_pts_one_crn = [0]

        known_pts_one_ctr = [55]
        known_pts_corn_three = [0,1,10]


        names = ["Border","Corners","One Corner","One Center","Three Corner"]
        arr = [known_pts_brd,known_pts_crn,known_pts_one_crn,known_pts_one_ctr,known_pts_corn_three]
        locs = [get_positions_of_sensors(locations,known) for known in arr]

        return names, arr, locs


    def get_deciders(self,mean_param = 3,count_param = 4):
        decider_mean = MeanThresholdDecider(threshold=mean_param)
        decider_count = CountDecider(count_param)

        names = ["Mean Decider","Count Decider"]

        return names, [decider_mean, decider_count]

    def parse_recording(self,file_str):
        sensor_positions, time_frames = read_recording(file_str)

        return sensor_positions, time_frames

    def train_bayes_algo(self,min_sep,max_sep,decider,known_sensors,known_locs,time_frames,seed = 1):
        algo = BayesSpacalAlgo(decider, min_sep, max_sep, len(sensor_positions), dim=2,
                               seed=seed)

        for frame in time_frames:
            algo.update_sensor_graph(frame)


        algo.set_roots(known_sensors, known_locs)

        return algo

def get_corner_ids(locations):

    ar = [(i,loc) for i,loc in enumerate(locations)]

    btm_left = min(ar,key=lambda x:sum(x[1]))[0]
    top_right = max(ar,key=lambda x:sum(x[1]))[0]
    btm_right = max(ar,key=lambda x:-x[1][1] + x[1][0])[0]
    top_left = max(ar,key=lambda x:x[1][1] - x[1][0])[0]

    return [btm_left,top_left,top_right,btm_right]

def get_btm_left_ids(locations,n= 4):
    ar = [(i, loc) for i, loc in enumerate(locations)]
    ar = sorted(ar,key=lambda x:sum(x[1]))
    k = [ ar[i][0] for i in range(n)]

    return k

def get_border_ids(locations,n):

    ar = [(i, loc) for i, loc in enumerate(locations)]

    ar_ = sorted(ar,key=lambda x:x[1][0])
    left = np.copy(ar_[:int(n/4)])
    right = np.copy(ar_[-int(n/4):len(ar)])

    ar_ = sorted(ar,key=lambda x:x[1][1])
    bottom = np.copy(ar_[:int(n/4)])
    top = np.copy(ar_[-int(n/4):len(ar)])

    all_ = []
    for lst in [left,right,top,bottom]:
        for tpl in lst:
            all_.append(tpl[0])

    return list(set(all_))


def test_belief_model(min_sep,max_sep,area,recording,known_pts,conv_rate,n,corr_prob,metric_f, seed = 1):
    MIN_SEP = min_sep
    MAX_SEP = max_sep

    sensor_positions, time_frames = read_labeled_recording(recording)


    known_positions = get_positions_of_sensors(sensor_positions, known_pts)

    #model = BeliefModel(time_frames=time_frames, positions=sensor_positions, static_points=known_pts)

    decider = CountDecider(4)
    algo = BeliefSpacialAlgo(decider, MIN_SEP, MAX_SEP, len(sensor_positions), conv_rate=conv_rate,
                            area_range=area, seed=seed)

    print("Known Locs:",known_positions)
    algo.set_known_sensors(known_pts, known_positions)


    err_ev = []
    var_err_ev = []
    x = []


    j = 0
    for epoch in range(n):

        for frame in time_frames:
            algo.update_sensor_locations(frame)
            if np.random.random() <= corr_prob:
                algo.correct_sensor_locations(optimize=True)

            mean, var, all_ = metric_f(algo.get_locations(),sensor_positions)
            err_ev.append(mean)
            var_err_ev.append(var)
            x.append(j)
            j += 1

    x = np.array(x)
    err_ev = np.array(err_ev)
    var_err_ev = np.array(var_err_ev)

    fig, ax = plt.subplots()
    ax.plot(x, err_ev)
    ax.fill_between(x, (err_ev - var_err_ev), (err_ev + var_err_ev), color='b', alpha=.1)

    plt.xlabel("iteration")
    plt.ylabel("mean error of a sensor location distance")
    plt.title("Mean Absolute Error (of sensor) Evolution ")
    plt.show()


def conv_rate_test_belief_model(min_sep,max_sep,area,recordings :list,known_pts_f,conv_rate_range : list,n,corr_prob,metric_f, seed = 1):
    MIN_SEP = min_sep
    MAX_SEP = max_sep





    err_ev = []
    var_err_ev = []
    x = []


    j = 0

    for  conv_rate in conv_rate_range:

        errs = []

        for rec in recordings:
            sensor_positions, time_frames = read_labeled_recording(rec)
            known_pts = known_pts_f(sensor_positions)
            known_positions = get_positions_of_sensors(sensor_positions, known_pts)


            decider = CountDecider(4)
            algo = BeliefSpacialAlgo(decider, MIN_SEP, MAX_SEP, len(sensor_positions), conv_rate=conv_rate,
                                    area_range=area, seed=seed)
            algo.set_known_sensors(known_pts, known_positions)


            for epoch in range(n):

                for frame in time_frames:
                    algo.update_sensor_locations(frame)
                    if np.random.random() <= corr_prob:
                        algo.correct_sensor_locations(optimize=True)

            mean, var, all_ = metric_f(algo.get_locations(),sensor_positions)
            errs.append(mean)

        err_ev.append(np.mean(errs))
        var_err_ev.append(np.var(errs))
        x.append(j)
        j += 1


    err_ev = np.array(err_ev)
    var_err_ev = np.array(var_err_ev)

    fig, ax = plt.subplots()
    ax.plot(conv_rate_range, err_ev, label="Final Error",color="blue")
    ax.fill_between(conv_rate_range, (err_ev - var_err_ev), (err_ev + var_err_ev), color='blue', alpha=.1)
    plt.legend(loc="upper right")
    plt.xlabel("iteration")
    plt.ylabel("mean error of a sensor location distance")
    plt.title("Mean Absolute Error (of sensor) Evolution ")
    plt.show()


def eval_belief_model(decider,min_sep,max_sep,area,recordings,known_pts_f,conv_rate,n,corr_prob,metric_f, seed = 1):
    MIN_SEP = min_sep
    MAX_SEP = max_sep

    init_errs = []
    errs= []
    for rec in recordings:
        sensor_positions, time_frames = read_labeled_recording(rec)

        known_pts = known_pts_f(sensor_positions)

        known_positions = get_positions_of_sensors(sensor_positions, known_pts)

        algo = BeliefSpacialAlgo(decider, MIN_SEP, MAX_SEP, len(sensor_positions), conv_rate=conv_rate,
                                area_range=area, seed=seed)

        #print("Known Locs:", known_positions)
        algo.set_known_sensors(known_pts, known_positions)
        mean, var, all_ = metric_f(algo.get_locations(),sensor_positions)
        init_errs.append(mean)

        for epoch in range(n):

            for frame in time_frames:
                algo.update_sensor_locations(frame)
                if np.random.random() <= corr_prob:
                    algo.correct_sensor_locations(optimize=True)

        mean, var, all_ = metric_f(algo.get_locations(), sensor_positions)
        errs.append(mean)

    print("Initial AVG err: ",np.mean(init_errs))
    print("Initial VAR err: ",np.var(init_errs))

    print("AVG err: ",np.mean(errs))
    print("VAR err: ",np.var(errs))

def eval_multilat_model(decider,min_sep,max_sep,discount,recordings,known_pts_f,metric_f, seed = 1):
    MIN_SEP = min_sep
    MAX_SEP = max_sep

    init_errs = []
    errs= []
    for rec in recordings:
        sensor_positions, time_frames = read_labeled_recording(rec)

        known_pts = known_pts_f(sensor_positions)

        known_positions = get_positions_of_sensors(sensor_positions, known_pts)

        algo = MultilatSpacialAlgo(decider, MIN_SEP, MAX_SEP, len(sensor_positions),dim = 2,branch_factor=3,discount=discount
                                ,seed=seed)

        #print("Known Locs:", known_positions)
        algo.set_roots(known_pts,known_positions)

        mean, var, all_ = metric_f(algo.get_locations_snapshot(),sensor_positions)
        init_errs.append(mean)


        for frame in time_frames:
            algo.update_sensor_graph(frame)

        algo.build_map(iter_=len(sensor_positions),find_all=False)

        mean, var, all_ = metric_f(algo.get_locations_snapshot(), sensor_positions)
        errs.append(mean)

    print("Initial AVG err: ",np.mean(init_errs))
    print("Initial VAR err: ",np.var(init_errs))

    print("AVG err: ",np.mean(errs))
    print("VAR err: ",np.var(errs))



def discount_test_multilat_model(decider,min_sep,max_sep,discounts : list,recordings,known_pts_f,metric_f, seed = 1):
    MIN_SEP = min_sep
    MAX_SEP = max_sep

    err_ev = []
    var_err_ev = []

    init_err_ev = []
    init_var_err_ev = []


    for discount in discounts:
        init_errs = []
        errs= []
        for rec in recordings:
            sensor_positions, time_frames = read_labeled_recording(rec)

            known_pts = known_pts_f(sensor_positions)

            known_positions = get_positions_of_sensors(sensor_positions, known_pts)

            algo = MultilatSpacialAlgo(decider, MIN_SEP, MAX_SEP, len(sensor_positions),dim = 2,branch_factor=3,discount=discount
                                    ,seed=seed)

            #print("Known Locs:", known_positions)
            algo.set_roots(known_pts,known_positions)

            mean, var, all_ = metric_f(algo.get_locations_snapshot(),sensor_positions)
            init_errs.append(mean)


            for frame in time_frames:
                algo.update_sensor_graph(frame)

            algo.build_map(iter_=len(sensor_positions),find_all=False)

            mean, var, all_ = metric_f(algo.get_locations_snapshot(), sensor_positions)
            errs.append(mean)

        err_ev.append(np.mean(errs))
        var_err_ev.append(np.var(err_ev))

        init_err_ev.append(np.mean(init_errs))
        init_var_err_ev.append(np.var(init_errs))


    err_ev = np.array(err_ev)
    var_err_ev = np.array(var_err_ev)

    init_err_ev = np.array(init_err_ev)
    init_var_err_ev = np.array(init_var_err_ev)

    fig, ax = plt.subplots()
    ax.plot(discounts, err_ev, label="Final Error", color="blue")
    ax.fill_between(discounts, (err_ev - var_err_ev), (err_ev + var_err_ev), color='blue', alpha=.1)
    ax.plot(discounts, init_err_ev, label="Initial Error", color="green")
    ax.fill_between(discounts, (init_err_ev - init_var_err_ev), (init_err_ev + init_var_err_ev), color='green', alpha=.1)
    plt.legend(loc="upper right")
    plt.xlabel("iteration")
    plt.ylabel("mean error of a sensor location distance")
    plt.title("Mean Absolute Error (of sensor) Evolution ")
    plt.show()






if __name__ == "__main__":

    metrics = Metrics(linear_transform_independent=True)

    rec = "../pygame_model/rand2.csv"

    recs = [f"../pygame_model/rand{i}.csv" for i in range(1,6)]


    #test_belief_model(min_sep=0.5,max_sep=2,area=((0,10),(0,10),(0,0)),recording=rec,known_pts=known_pts,conv_rate=0.0005,n=5,corr_prob=0.1,metric_f=metrics.avg_absolute_location_error,seed = 1)
    #conv_rate_test_belief_model(min_sep=0.5,max_sep=2,area=((0,10),(0,10),(0,0)),recordings=recs,known_pts_f = lambda x: get_border_ids(x,n=36),conv_rate_range=[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05],n=10,corr_prob=0.1,metric_f=metrics.avg_absolute_location_error,seed = 1)
    eval_belief_model(CountDecider(12),min_sep=0.5,max_sep=2,area=((0,10),(0,10),(0,0)),recordings = recs,known_pts_f=lambda x: get_btm_left_ids(x,n=1),conv_rate=0.0005,n=15,corr_prob=0.1,metric_f=metrics.avg_absolute_location_error,seed = 1)


## In a randomized sensor patch, it can happen that stimulus is barely pressing on anything since at that time there are no sensors that are near each other.
## However since we normalize pressures at each time frame, it will be inferred that sensor is pressing strongly on some sensors anyway. In the end sensors will get shifted
## together eventough they are far away. [THIS cannot happen in grid like sensor array, because there are no sensor sparse regions, thus this accident cannot happen]
## Partial mitigation is that, if the maximum exhibited pressure at each time frame is too low, the frame is completely ignored, as it is possible that stimulus was not directly
## pressing on any sensor.


## TODO:
# 1. one border sensor known
# 2. three neighbor border sensor known
# 3. all borders known
# 4. all 4 corner sensors known


#TODO Task 1
# compare performance on randomized grid with different amount of sensors at play
# metrics: relative dists, linear indep - abs dists
# algos: both

## TODO Task 2
# compare different sized stimulis on randomized vs grid like sensor patch (only gridc line)
# metrics: linear indep metric - abs dists
# algos: both

## TODO Task 3
# compare cuboid and sphere stimulis on both random and grid like sensor arrays
# metrics: relativ dists, linear indep metric - abs dists
# algos: multilat (due to pressure to distance mapping)

## TODO Task 4
# compare Known sensor configurations in 100 sensor 2 max sep, random vs grid like sensor arrays
# matrics: relative dists, linear indep - abs dists, abs dists, neighborgoods
# algos: both (although multilat will have problems)

## TODO task 5
# examine how the error for belief model converges over time
# metrics: (one sensor known, all border sensors known), rel dist, lin indep - abs dist, abs dist
# algo: belief

## TODO task 6
# examine how does conv rate affect performance for the belief mode for 100 sensor 2 max sep vs 3 max sep (hypothetized that 3 max sep will need smaller conv rate, since more belief is compounded - more sensor activated at the same time)
# metrics: (one sensor known, all border sensors known), rel dist, lin indep - abs dist, abs dist
# algo: belief

## TODO task 7
# examine how does the discount param affect perf of Multilat model for 100 sensors 1 sensor known VS 4 corner sensors known (hypothesis that small discount will help fit into the corner point that are known)
# metrics: (one sensor known, corner known, all border known), rel dist, lin indep - abs dist, abs dist, neighborhood
# algo: multilat

## TODO task 8
# pathological sensor array, where min sep is very small, but in general 2 closest sensors are distant around 0.5 -


MIN sep-   0.1; MaxSep - 2, 20^2 ## DESCRIBE in algo


