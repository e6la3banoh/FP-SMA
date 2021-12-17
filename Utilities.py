import math
import matplotlib.pyplot as plt
import numpy as np
import datetime

def arctanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def arctanh_scaled_offset(x, beta, offset):
    y = beta * (x - offset)
    return arctanh(y)

def plot_clustering(pop_pos):
    x = [pop_pos[i][0] for i in range(len(pop_pos))]
    y = [pop_pos[i][1] for i in range(len(pop_pos))]
    plt.scatter(x, y, marker='+')
    #plt.show()
    dt_date = datetime.datetime.now()
    file_name = "{}.png".format(dt_date.strftime("%d_%m_%y_%I_%M_%S_%p"))
    plt.savefig("{}.png".format(file_name))

