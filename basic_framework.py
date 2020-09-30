import numpy as np
import torch

class actor:
    """
    Parent class for policies.
    Child classes need to specify the forward module and the type of distribution
    """

    def __init__(self, dist_type):
        '''
        Input:
            dist_type - torch distribution class
        '''

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
        '''
        Computes a prob. distribution over actions for all given parameters

        Input:
            params - torch tensor of shape (batch, *|params|)
        Output:
            dist - a distribution with batch_shape (batch), and corresponding parameters for each distribution
        '''

        # generate distribution from parameters
        dist = self.dist_type(params)

        return dist
    
    def sample_action(self, state):
        '''
        Samples an action from the policy for the given state
        Input:
            state - torch tensor of shape (1, *|S|)
        Output:
            action - torch tensor of state (1, *|A|)
        '''

        assert state.shape[0] == 1, 'Expected state to have 0th dim 1, but got {}'.format(state.shape[0])
        assert len(state.shape) > 1, 'Expected state to have at least 2 dimensions, but got {}'.format(len(state))

        # compute parameters
        params = self.forward(state)

        # get distribution
        dist = self.get_dist(params)[0]

        # sample from the distribution
        action = dist.sample_n(1)

        return action


    def get_log_probs(self, states, actions):
        '''
        expects the same number of states and actions and returns logprobabilities
        Input:
            states - torch tensor of shape (batch, *|S|)
            actions - torch tensor of shape (batch, *|A|)
        '''
        assert states.shape[0] == actions.shape[0], 'states and actions dont match along 0th dimension: {} and {}'.format(states.shape[0], actions.shape[0])
        
        # compute params
        params = self.forward(states)

        # get dists
        dist = self.get_dist(params)

        # get log_probs
        log_probs = dist.log_prob(actions)

        return log_probs

    def get_kl(self, states):
        '''
        Computes the KL divergence between the policy and itself, averaged over states.
        This function will be used to compute the Fisher matrix via torch.autograd
        '''
        # compute parameters for all states
        params = self.forward(states)
        params_detach = params.detach().clone() # detached parameters for second policy

        # get dists for all states
        dists = self.get_dist(params)
        dists_detached = self.get_dist(params_detach)

        # compute mean kl over all states
        kl = torch.distributions.kl_divergence(dists, dists_detached)      
        kl = kl.mean()

        return kl


class dummy_child(actor):
    '''
    dummy child class to showcase how to use the actor parent class
    '''

    def __init__(self, env, num_hidden):
        '''
        init child class, set up forward parameters, such as number of hidden units or size of the state and action space.
        '''
        # init parent class and set dist type, e.g. univariate normal distribution
        super(dummy_child, self).__init__(torch.distributions.normal.Normal)

        #####
        # some more initialization, depending on the specific method 
        # e.g.
        self.network = torch.nn.Sequential(torch.nn.Linear(env.nS, num_hidden), torch.nn.ReLU(), torch.nn.Linear(num_hidden, 2))        
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


