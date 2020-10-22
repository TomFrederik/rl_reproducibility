import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = './kl/logs/Acrobot-v1/'
plot_dir = './kl/plots/Acrobot-1/'

trpo_files = [data_dir + 'TRPO/' + str(seed) + '_log.npz' for seed in range(10,15)]
npg_files = [data_dir + 'NPG/' + str(seed) + '_log.npz' for seed in range(10,15)]

max_kl = 0.01

# load step_size data
trpo_step_sizes = []
trpo_returns = []
npg_returns = []

for i, f in enumerate(trpo_files):
    data_step_size = np.load(f)['step_size']
    trpo_step_sizes.append(data_step_size)
    data_returns = np.load(f)['returns']
    trpo_returns.append(data_returns)

for i, f in enumerate(npg_files):
    data_returns = np.load(f)['returns']
    npg_returns.append(data_returns)

trpo_step_sizes = np.array(trpo_step_sizes)
trpo_returns = np.array(trpo_returns)
npg_returns = np.array(npg_returns)

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
plt.show()
plt.savefig(plot_dir + 'trpo_stepsize_{}.pdf'.format(max_kl))

mean_trpo_return = np.mean(trpo_returns, axis=0)
std_trpo_return = np.std(trpo_returns, axis=0)
mean_npg_return = np.mean(npg_returns, axis=0)
std_npg_return = np.std(npg_returns, axis=0)

plt.figure()
plt.plot(mean_trpo_return, label="TRPO")
plt.plot(mean_npg_return, label="NPG")
plt.fill_between(np.arange(len(mean_trpo_return)), mean_trpo_return - std_trpo_return, 
                    mean_trpo_return + std_trpo_return, alpha=.4, label=r'$\pm 1 \sigma$')
plt.fill_between(np.arange(len(mean_npg_return)), mean_npg_return - std_npg_return, 
                    mean_npg_return + std_npg_return, alpha=.4, label=r'$\pm 1 \sigma$')
plt.ylabel('Return')
plt.xlabel('Iteration')
plt.title('Average Return')
plt.legend()
plt.show()
plt.savefig(plot_dir + 'avg_return.pdf')