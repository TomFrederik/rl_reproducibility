import torch
import actor as act

class experiment:

    def __init__(self, ac, ac_kwargs, train_alg, alg_kwargs, v=None, v_kwargs=None):
        '''
        Input:
            ac - an actor object
            ac_kwargs - dict, kwargs for ac
            train_alg - training algorithm class, should be NPG or TRPO
            alg_kwargs - dict, kwargs for the training algorithm, contains hyperparams
            v - a critic object
            v_kwargs - dict, kwargs for v
        '''

        self.actor = ac(**ac_kwargs)
        if v:
            self.critic = v(**v_kwargs)
        
        self.algorithm = train_alg(**alg_kwargs)


        # for result logging
        self.results = {}

    def run(self, seed):
        '''
        Run experiment with a the given random seed
        '''
        #TODO: flesh this out further
        self.results[seed] = self.algorithm.train()
    
    def plot(self):
        '''
        Plot all experiments run so far, i.e. for different seeds
        '''
        #TODO: for seed in self.results.keys(): plot results[seed]
        