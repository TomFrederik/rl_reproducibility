import numpy as np
import matplotlib.pyplot as plt
import os


def get_params(folder):

    folder = folder[2:-1] # drop slashes
    lr = folder[3:7]
    gamma = folder[14:18]
    lamda = folder[23:]

    return lr, gamma, lamda

def get_returns(folder):
    '''
    returns mean and std of returns, averaged over all seeds in folder
    '''

    # get seed files
    (_, _, seed_files) = next(os.walk(folder))
    seed_files = [folder + f for f in seed_files]

    # load first file to determine shape
    first_data = np.load(seed_files[0])
    length = len(first_data['returns'])
    num_seeds = len(seed_files)

    # set up return array
    returns = np.zeros((num_seeds, length))
    returns[0] = first_data['returns']
    
    for i, f in enumerate(seed_files[1:]):
        data = np.load(f)
        returns[i] = data['returns']
    
    mean_return = np.mean(returns, axis=0)
    std_return = np.std(returns, axis=0)

    return mean_return, std_return



# define path for saving plots
plot_dir = './plots/'

# get folders for different parameter settings
(_, dirs, _) = next(os.walk('./'))
dirs = dirs[1:] # drop plot dir
dirs = ['./' + d + '/' for d in dirs]

# set up plot
plt.figure(1) # gamma = 0.9


for i, param_dir in enumerate(dirs):
    # get params of this run
    lr, gamma, lamda = get_params(param_dir)
    # get mean and std of return, averaged over seeds
    mean_return, std_return = get_returns(param_dir)

    plt.plot(np.arange(len(mean_return)), mean_return, label='lr='+lr+r'; $\gamma$='+gamma)

    #plt.fill_between(np.arange(len(mean_return)), mean_return-std_return, mean_return+std_return, alpha=.3)


plt.title(r'mean return; $\lambda$=0.97')
plt.xlabel('Iteration')
plt.ylabel('Return')
plt.legend()
plt.savefig(plot_dir+'comparison_lr_return.pdf')



'''
plt.figure(2) # lambda = 1

for i, param_dir in enumerate(dirs):
    # get params of this run
    kl, gamma, lamda = get_params(param_dir)
    # get mean and std of return, averaged over seeds
    mean_return, std_return = get_returns(param_dir)

    if lamda == '1.00':
        plt.plot(np.arange(len(mean_return)), mean_return, label=r'$\gamma$ = '+gamma)


plt.figure(2)
plt.title(r'kl 0.01 ; mean return ; $\lambda$=1')
plt.xlabel('Iteration')
plt.ylabel('Return')
plt.legend()
plt.savefig(plot_dir+'comparison_gamma_return.pdf')
'''

    