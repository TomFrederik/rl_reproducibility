import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from actor import Actor
from algorithms import BaselineCriticMC, ActorOnlyMC, NPG, TRPO, get_returns
from utils import sample_memory
import gym
from experiment_class import Experiment, mult_seed_exp
from time import time
import os

class Critic(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)
        
        # init weights
        gains = [np.sqrt(2), np.sqrt(2), 1]
        layers = [self.fc1, self.fc2, self.fc3]
        for i in range(len(layers)):
            nn.init.xavier_uniform_(layers[i].weight, gain=gains[i])
            layers[i].bias.data.fill_(0.01)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        v = self.fc3(x)
        return v

class ContActor(Actor):
    '''
    Actor for continous action space
    '''

    def __init__(self, num_input, num_hidden, **kwargs):
            super(ContActor, self).__init__(dist_type=torch.distributions.normal.Normal)
            self.fc1 = nn.Linear(num_input, num_hidden)
            self.fc2 = nn.Linear(num_hidden, num_hidden)
            self.fc3 = nn.Linear(num_hidden, 2)
            
            # init weights
            gains = [np.sqrt(2), np.sqrt(2), 1]
            layers = [self.fc1, self.fc2, self.fc3]
            for i in range(len(layers)):
                nn.init.xavier_uniform_(layers[i].weight, gain=gains[i])
                layers[i].bias.data.fill_(0.01)

    # override the parent class' forward method
    def forward(self, states):
        '''
        Computes parameters of gaussian distributions over actions given a tensor of states
        Input:
            states - pytorch tensor of shape (batch, *|S|)
        Output:
            params - pytorch tensor of shape (batch, *|params|)
        '''

        x = nn.functional.relu(self.fc1(states))
        x = nn.functional.relu(self.fc2(x))
        params = self.fc3(x)

        params[:,1] = torch.exp(params[:,1]) # make sure variance is positive
        
        return params

# run a single experiment

####
# dont change these params:
ac_kwargs = {'num_hidden':20}
critic_kwargs = {'num_hidden':20}
critic_optim_kwargs = {'lr':3e-4}
target_alg_kwargs = {'batch_size':16, 'epochs':5}
num_iters = 20
seeds = [42,11] # subject to change
log_dir = './mc_cont/NPG/BaselineCriticMC/'
try:
    os.mkdir(log_dir)
except:
    pass

####

####
# do change these params:
target_alg_kwargs['gamma'] = 0.98
ep_per_iter = 5
ac_alg_kwargs = {'max_kl':0.01}
####

experiment_parameters =   {'seed':42, 
                           'env_str':'MountainCarContinuous-v0',
                           'ac':ContActor,
                           'ac_kwargs':ac_kwargs,
                           'ac_alg':NPG,
                           'ac_alg_kwargs':ac_alg_kwargs,
                           'target_alg':BaselineCriticMC,
                           'target_alg_kwargs':target_alg_kwargs,
                           'critic':Critic,
                           'critic_kwargs':critic_kwargs,
                           'critic_optim':torch.optim.Adam,
                           'critic_optim_kwargs':critic_optim_kwargs,
                           'num_iters':num_iters,
                           'ep_per_iter':ep_per_iter,
                           'log_file':'./mc_cont/NPG/BaselineCriticMC/single.npz'}


###
# set up search space for max_kl
# we are setting up a log-normal distribution
###
low = -4
high = 2
num_trials = 3 # how many draws we are taking


search_time = time()
max_return = -np.inf
kl_return_list = []
kl_return_std_list = []
kl_list = []
for n in range(num_trials):

    trial_time = time()

    # draw a sample from the loguniform distribution
    max_kl = 10 ** (np.random.uniform(low, high))

    kl_list.append(max_kl)

    print('\nNow running wiht max_kl = {}'.format(max_kl))
    
    # set up experiment
    ac_alg_kwargs['max_kl'] = max_kl
    experiment_parameters['ac_alg_kwargs'] = ac_alg_kwargs
    trial_log_dir = log_dir + 'kl_{0:1.4f}/'.format(max_kl)
    try:
        os.mkdir(trial_log_dir)
        os.mkdir(trial_log_dir+'plots/')
    except:
        pass

    experiment = Experiment(**experiment_parameters)

    # run mutliple experiments with different seeds
    mult_exp = mult_seed_exp(experiment_parameters, seeds, trial_log_dir)
    mult_exp.run()

    # get results
    mean_returns, std_returns, _, _ = mult_exp.get_mean_results()

    print('Mean return at termination with max_kl {0:1.5f} was {1:1.2f}'.format(max_kl, mean_returns[-1]))

    kl_return_list.append(mean_returns[-1])
    kl_return_std_list.append(std_returns[-1])

    if mean_returns[-1]>max_return:
        print('Updating max return run..')
        max_return = mean_returns[-1]

        mult_exp.plot(trial_log_dir+ 'plots/best_run_')
    #mult_exp.plot(plot_path=log_dir+'plots/')

    print('This trial took {0:1.2f} seconds'.format((time()-trial_time)))

print('Hyperparameter search finished. Time: {0:1.0f} minutes and {1:1.2f} seconds'.format((time()-search_time)//60, (time()-search_time)))

print('Plotting Return - KL diagram')

plt.figure()
plt.errorbar(kl_list, kl_return_list, yerr=kl_return_std_list, fmt='o', capsize=6)
plt.xlabel('max KL')
plt.ylabel('Final Return')
plt.savefig(log_dir + 'kl_return_plot.pdf')

# save results
np.savez_compressed(log_dir + 'kl_return_summary.npz', kl_list=kl_list, kl_return_list=kl_return_list, kl_return_std_list=kl_return_std_list)

#def run_loguniform_hp_search(low, high, num_trials, keyword, params):
#    '''
#    Runs a loguniform hyperparameter search of the param 'keyword'
#    '''






