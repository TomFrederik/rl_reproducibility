import torch
import torch.nn as nn
import numpy as np
from actor import DummyDiscrete
from algorithms import BaselineCriticMC, ActorOnlyMC, NPG, TRPO, get_returns
from utils import sample_memory
import gym
from experiment_class import Experiment, mult_seed_exp


class Critic(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v

# run a single experiment

ac_kwargs = {'num_hidden':10}
ac_alg_kwargs = {'max_kl':0.01}

target_alg_kwargs = {'gamma':0.98, 'batch_size':16, 'epochs':5}

critic_kwargs = {'num_hidden':10}
critic_optim_kwargs = {'lr':3e-4}


# note that you only need to specify 'seed' in a single experiment run, 
# otherwise provide it in the constructor of mult_seed_exp
experiment_parameters =   {'seed':42, 
                           'env_str':'CartPole-v1',
                           'ac':DummyDiscrete,
                           'ac_kwargs':ac_kwargs,
                           'ac_alg':TRPO,
                           'ac_alg_kwargs':ac_alg_kwargs,
                           'target_alg':BaselineCriticMC,
                           'target_alg_kwargs':target_alg_kwargs,
                           'critic':Critic,
                           'critic_kwargs':critic_kwargs,
                           'critic_optim':torch.optim.Adam,
                           'critic_optim_kwargs':critic_optim_kwargs,
                           'num_iters':20,
                           'ep_per_iter':5,
                           'log_file':'./demo/single/log.npz'}

'''
experiment = Experiment(**experiment_parameters)

experiment.run()

experiment.plot('./demo/single/plots/')
'''

# run mutliple experiments with different seeds
seeds = [42,11,23,58]
log_dir = './demo/multi/'
mult_exp = mult_seed_exp(experiment_parameters, seeds, log_dir)

mult_exp.run()

mult_exp.plot(plot_path=log_dir+'plots/')