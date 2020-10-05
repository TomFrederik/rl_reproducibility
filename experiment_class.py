import torch
import actor as act
import gym
import utils
import algorithms

class Experiment:

    def __init__(self, seed, env_str, ac, ac_kwargs, ac_alg, ac_alg_kwargs, target_alg, target_alg_kwargs, v=None, v_kwargs=None, v_alg=None, v_alg_kwargs=None, num_iters=100, ep_per_iter=5):
        '''
        Input:
            seed - random seed for the experiment
            env_str - string to describe a gym environment
            ac - an Actor subclass
            ac_kwargs - dict, kwargs for ac
            ac_alg - ActorAlgorithm subclass, should be NPG or TRPO
            alg_kwargs - dict, kwargs for the training algorithm, contains hyperparams
            target_alg - TargetAlgorithm subclass
            v - a Critic class
            v_kwargs - dict, kwargs for v
            num_iters - int, number of iterations / updates applied to the policy
            ep_per_iter - int, number of episodes to use for updating per iteration
        '''

        # save values for later
        self.seed = seed
        self.num_iters = num_iters
        self.ep_per_iter = ep_per_iter

        # instantiate environment and neural networks
        self.env = gym.make(env_str)
        self.env.seed(self.seed)

        self.actor = ac(env=self.env, **ac_kwargs)
        self.target_alg = target_alg(**target_alg_kwargs)
        self.ac_alg = ac_alg(self.actor, self.target_alg, **ac_alg_kwargs)

        if not (v is None):
            self.critic = v(**v_kwargs)
            self.v_alg = v_alg(**v_alg_kwargs)
            self.v.optim = torch.optim.Adam(self.critic.parameters(), lr=1e-3) # should this be in one of the kwargs, or passed via v_lr?
    

        # for result logging
        self.results = {}

    def run(self):
        '''
        Run experiment
        '''
        #TODO: flesh this out further

        # make sure torch is seeded correctly
        torch.manual_seed(seed)

        '''
        for i in range(100):
            # sample 5 episodes
            memory = sample_memory(env, actor, 5, render=True)
            returns = get_returns(memory[2], memory[3])
            print("Episode {}, Returns: {}".format(10 * (i + 1), returns.mean().item()))
            # train the critic on those episodes
            critic_alg.train(memory)
            # train the actor on those episodes
            actor_alg.train(memory)
        '''



    
    def plot(self):
        '''
        Plot all experiments run so far, i.e. for different seeds
        '''
        #TODO: for seed in self.results.keys(): plot results[seed]
        
