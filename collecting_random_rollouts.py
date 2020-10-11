import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from actor import Actor
from algorithms import get_episode_returns
import utils
import gym
from time import time
import os

class Uniform(Actor):
    '''
    Actor for continous action space
    '''

    def __init__(self, **kwargs):
            super(Uniform, self).__init__(dist_type=torch.distributions.uniform.Uniform)
            

    # override the parent class' forward method
    def forward(self, states):
        '''
        Computes parameters of uniform over actions given a tensor of states
        Input:
            states - pytorch tensor of shape (batch, *|S|)
        Output:
            params - pytorch tensor of shape (batch, *|params|)
        '''
        
        params = torch.ones((states.shape[0], 2))
        params[:,0] *= LOW
        params[:,1] *= HIGH

        return params

    # overwrite get_dist
    def get_dist(self, params):
        '''
        Computes a prob. distribution over actions for all given parameters

        Input:
            params - torch tensor of shape (batch, *|params|)
            Warning! This assumes all params are 1D!
        Output:
            dist - a distribution with batch_shape (batch), and corresponding parameters for each distribution
        '''
        assert len(params.shape) == 2, 'Expected params to have 2 dimensions (batch, num_params), but got {}'.format(params.shape)

        # generate distribution from parameters
        dist = self.dist_type(low=params[:,0], high=params[:,1])

        return dist
env = gym.make('MountainCarContinuous-v0')

rollouts = {'states': [], 'rewards': [], 'masks': []}

num_rollouts = 10000
ep_per_iter = 10
LOW = env.action_space.low[0]
HIGH = env.action_space.high[0]

actor = Uniform()

num_pos_rew = 0

for i in range(num_rollouts//ep_per_iter):

    # sample trajectories
    (states, actions, rewards, masks) = utils.sample_memory(env, actor, ep_per_iter)

    num_pos_rew += len(rewards[rewards>0])

    if len(rewards[rewards>0]) > 0:
        rollouts['states'].append(states)
        rollouts['rewards'].append(rewards)
        rollouts['masks'].append(masks)

    #returns = get_episode_returns(rewards, masks)
    
    print('Sampled {0} episodes. Num_pos_rew = {1}'.format((i+1)*ep_per_iter, num_pos_rew))


# save
np.savez_compressed('./random_rollouts/{}.npz'.format(int(time())), **rollouts)
