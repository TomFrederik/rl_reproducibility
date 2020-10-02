import numpy as np
import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Parent class for policies.
    Child classes need to specify the forward module and the type of distribution
    """

    def __init__(self, dist_type):
        '''
        Input:
            dist_type - torch distribution class
        '''
        super().__init__()
        if (dist_type != torch.distributions.normal.Normal) & (dist_type != torch.distributions.categorical.Categorical):
            print('WARNING! Expected Normal or Categorical distribution, but got {}! Might lead to unintended behavior.'.format(dist_type))
        
        self.dist_type = dist_type
    
    def forward(self, states):
        '''
        Computes parameters of the distribution given a tensor of states
        Input:
            states - pytorch tensor of shape (batch, *|S|)
        Output:
            params - pytorch tensor of shape (batch, *|params|)
        '''
        #implement forward pass in each child class
        raise NotImplementedError('You should implement the forward method in a child class')


    def get_dist(self, params):
        #TODO: Make this work with params of different dimensions
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
        if self.dist_type == torch.distributions.normal.Normal:
            dist = self.dist_type(*torch.transpose(params, 0, 1))
        elif self.dist_type == torch.distributions.categorical.Categorical:
            dist = self.dist_type(params)
        else:
            print('WARNING: Expected categorical or normal dist_type, but {}. Behavior might be unintended.'.format(self.dist_type))
        
        return dist
    
    def sample_action(self, state):
        '''
        Samples an action from the policy for the given state
        Input:
            state - torch tensor of shape (1, *|S|)
        Output:
            action - torch tensor of state (1, *|A|)
        '''

        assert state.shape[0] == 1, 'Expected state to have 0th dim 1, but got {}!'.format(state.shape[0])
        assert len(state.shape) == 2, 'Expected state to have 2 dimensions (batch, *|S|), but got {}'.format(state.shape)

        # compute parameters
        params = self.forward(state)

        # get distribution
        dist = self.get_dist(params)
        
        # sample from the distribution
        action = dist.sample((1,))[0]

        return action


    def get_log_probs(self, states, actions):
        '''
        expects the same number of states and actions and returns logprobabilities
        Input:
            states - torch tensor of shape (batch, *|S|)
            actions - torch tensor of shape (batch, *|A|)
        '''
        assert states.shape[0] == actions.shape[0], 'states and actions dont match along 0th dimension: {} and {}!'.format(states.shape, actions.shape)
        
        # compute params
        params = self.forward(states)

        # get dists
        dist = self.get_dist(params)
        
        # get log_probs
        log_probs = dist.log_prob(actions.squeeze())

        return log_probs

    def get_kl(self, states, old_actor=None):
        '''
        Computes the KL divergence between the policy and itself, averaged over states.
        This function will be used to compute the Fisher matrix via torch.autograd

        Input:
            states - torch tensor of shape (batch, *|S|)
        Output:
            kl - torch tensor of shape (1,)
        '''
        # compute parameters for all states
        params = self.forward(states)
        if old_actor is None:
            params_detach = params.detach().clone() # detached parameters for second policy
        else:
            params_detach = old_actor.forward(states).detach()

        # get dists for all states
        dists = self.get_dist(params)
        dists_detached = self.get_dist(params_detach)

        # compute mean kl over all states
        kl = torch.distributions.kl_divergence(dists, dists_detached)      
        kl = kl.mean()

        return kl


class DummyCont(Actor):
    '''
    dummy child class to showcase how to use the actor parent class for continuous control
    '''

    def __init__(self, env, num_hidden):
        '''
        init child class, set up forward parameters, such as number of hidden units or size of the state and action space.
        '''
        # init parent class and set dist type, e.g. univariate normal distribution
        super().__init__(torch.distributions.normal.Normal)

        #####
        # some more initialization, depending on the specific method 
        # e.g.
        self.network = torch.nn.Sequential(torch.nn.Linear(env.observation_space.shape[0], num_hidden), torch.nn.ReLU(), torch.nn.Linear(num_hidden, 2))        
        # ...
        #####

    # override the parent class' forward method
    def forward(self, states):
        '''
        Computes parameters of the distribution given a tensor of states
        Input:
            states - pytorch tensor of shape (batch, *|S|)
        Output:
            params - pytorch tensor of shape (batch, *|params|)
        '''

        params = self.network(states)
        params[:,1] = torch.exp(params[:,1]) # make sure variance is positive
        
        return params


class DummyDiscrete(Actor):
    '''
    dummy child class to showcase how to use the actor parent class for discrete control.

    This model outputs probabilities for each discrete action.
    '''

    def __init__(self, env, num_hidden):
        '''
        init child class, set up forward parameters, such as number of hidden units or size of the state and action space.
        '''
        # init parent class and set dist type, e.g. univariate normal distribution
        super().__init__(torch.distributions.categorical.Categorical)

        #####
        # some more initialization, depending on the specific method 
        # e.g.
        self.network = torch.nn.Sequential(torch.nn.Linear(env.observation_space.shape[0], num_hidden), torch.nn.ReLU(), torch.nn.Linear(num_hidden, env.action_space.n), torch.nn.Softmax(dim=1))        
        # ...
        #####

    # override the parent class' forward method
    def forward(self, states):
        '''
        Computes parameters of the distribution given a tensor of states
        Input:
            states - pytorch tensor of shape (batch, *|S|)
        Output:
            params - pytorch tensor of shape (batch, *|params|)
        '''

        params = self.network(states)
        
        return params

