
import tensorflow as tf
from noise_filter import *
import matplotlib.pyplot as plt
from scipy.stats import norm


## TODO filtering sensor pressures when the arm is rotated should be done with different distribution than normal, since the data is skewed

PATH = "C:/University/Marble/Data/incorrect_orthosis_down_1cm_90/"
dir = os.listdir(PATH)





