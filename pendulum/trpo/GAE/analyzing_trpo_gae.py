import numpy as np
import matplotlib.pyplot as plt
import os

plot_dir = './plots'

(_, dirs, _) = next(os.walk('./'))
dirs = dirs[1:] # drop plot dir
dirs = ['./' + d + '/' for d in dirs]


def get_params(folder):

    folder = folder[2:-1] # drop slashes
    kl = folder[3:7]
    print(kl)
    gamma = folder[13:16]
    lamda = folder[21:]
    print(gamma)
    print(lamda)

    return kl, gamma, lamda



get_params(dirs[0])
    
    