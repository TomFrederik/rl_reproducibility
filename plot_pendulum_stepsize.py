import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = './kl/logs/Pendulum-v0/'
plot_dir = './kl/plots/Pendulum-v0/'

trpo_files = [data_dir + 'TRPO/' + str(seed) + '_log.npz' for seed in range(10,15)]
npg_files = [data_dir + 'NPG/' + str(seed) + '_log.npz' for seed in range(10,15)]

max_kl = 0.01

# load step_size data
trpo_step_sizes = []

for i, f in enumerate(trpo_files):
    data = np.load(f)['step_size']
    trpo_step_sizes.append(data)

trpo_step_sizes = np.array(trpo_step_sizes)

mean_stepsize = np.mean(trpo_step_sizes, axis=0)
std_stepsize = np.std(trpo_step_sizes, axis=0)

plt.figure()
plt.plot(mean_stepsize, label='mean')
plt.fill_between(np.arange(len(mean_stepsize)), mean_stepsize - std_stepsize, mean_stepsize + std_stepsize, alpha=.4, label=r'$\pm 1 \sigma$')
plt.axhline(0.1, linestyle='--', color='r', label='NPG lr = 0.1')
plt.ylabel('Step Size')
plt.xlabel('Iteration')
plt.title('Max KL = 0.01')
plt.legend()
plt.savefig(plot_dir + 'trpo_stepsize_{}.pdf'.format(max_kl))