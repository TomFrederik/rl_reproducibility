import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = './kl/logs/Pendulum-v0/long/'
plot_dir = './kl/plots/Pendulum-v0/long/'

files = [data_dir + 'TRPO/' + str(seed) + '_log.npz' for seed in range(10,20)]

max_kl = 0.01

# load step_size data
step_sizes = []
returns = []

for i, f in enumerate(files):
    data = np.load(f)
    returns.append(data['returns'])
    step_sizes.append(data['step_size'])

step_sizes = np.array(step_sizes)
returns = np.array(returns)

diff_returns = np.diff(returns,axis=1)
diff_step_sizes = step_sizes[:,:-1] # drop last step size

x_ret = np.concatenate((step_sizes.reshape(1, np.prod(step_sizes.shape)), returns.reshape(1, np.prod(returns.shape))), axis=0)
x_diff = np.concatenate((diff_step_sizes.reshape(1, np.prod(diff_step_sizes.shape)), diff_returns.reshape(1, np.prod(diff_returns.shape))), axis=0)


corr_ret = np.corrcoef(x_ret)[0,1]
corr_diff = np.corrcoef(x_diff)[0,1]
print('Correlation Return <-> step size: {0:1.3f}'.format(corr_ret))
print('Correlation Diff Return <-> step size: {0:1.3f}'.format(corr_diff))

