from data_classification.noise_filter import *
import matplotlib.pyplot as plt
from scipy.stats import norm


## TODO filtering sensor pressures when the arm is rotated should be done with different distribution than normal, since the data is skewed

PATH = "C:/University/Marble/Data/incorrect_orthosis_down_1cm_90/"
dir = os.listdir(PATH)

vals_ = np.array([])
for file in dir:
    if not "RAW" in file: continue
    a = SampleDataAnalysis(1,file_path=PATH + file)

    p, r, stamp = a.sensor_values_at_arm_degrees(90)

    vals = p

    vals.sort()

    vals = vals[1:-1]

    mu = np.mean(vals)
    std = np.std(vals)
    x = np.linspace(-30, 130, 100)
    y = np.zeros(len(x))

    for i, x_ in enumerate(x): y[i] = norm.pdf(x_, mu, std)

    plt.hist(vals, 100, density=True)
    plt.plot(x, y)

    filtered = []

    bot = norm.ppf(0.01, mu, std)
    top = norm.ppf(0.99, mu, std)

    for v in vals:

        if bot <= v <= top:
            filtered.append(v)

    plt.hist(filtered, 100, density=True)
    plt.show()

    vals_ = np.concatenate([vals_,p],axis=0)


