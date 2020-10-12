import numpy as np
import matplotlib.pyplot as plt
import os


file_npg = './pendulum/npg/GAE/lr_0.00100/all_results.npz'
file_trpo = './pendulum/trpo/GAE/kl_0.00100/all_results.npz'

data_npg = np.load(file_npg)
data_trpo = np.load(file_trpo)

for key in data_npg.keys():
    
# get mean and std of stepsizes
npg_stepsizes = []
for i in range(len(npg_files)):    
    # load 
    results = np.load(dir_npg+npg_files[i])
    npg_stepsizes[i].append(results['step_size'])

np.mean(npg_stepsizes, axis=0)
np.std(npg_stepsizes, axis=0)
