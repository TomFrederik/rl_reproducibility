import torch
import actor as act
import gym
import utils
import algorithms
import matplotlib.pyplot as plt
import numpy as np

class Experiment:

    def __init__(self, seed, env_str, ac, ac_kwargs, ac_alg, ac_alg_kwargs, target_alg, target_alg_kwargs, critic=None, critic_kwargs=None, critic_optim=None, critic_optim_kwargs=None, num_iters=100, ep_per_iter=5, log_file='./log.npz'):
        '''
        Input:
            seed - random seed for the experiment
            env_str - string to describe a gym environment
            ac - an Actor subclass
            ac_kwargs - dict, kwargs for ac
            ac_alg - ActorAlgorithm subclass, should be NPG or TRPO
            alg_kwargs - dict, kwargs for the training algorithm, contains hyperparams
            target_alg - TargetAlgorithm subclass, will compute the target, e.g. MC return or GAE
            critic - a Critic class
            critic_kwargs - dict, kwargs for critic
            critic_optim - a optimizer class from torch.optim
            critic_optim_kwargs - kwargs for the critic optimizer
            num_iters - int, number of iterations / updates applied to the policy
            ep_per_iter - int, number of episodes to use for updating per iteration
            log_file - str, file for saving the results of the experiment in
        '''

        # save values for later
        self.seed = seed
        self.num_iters = num_iters
        self.ep_per_iter = ep_per_iter
        self.log_file = log_file

        # instantiate environment and neural networks
        self.env = gym.make(env_str)
        self.env.seed(self.seed)

        # set up actor and target / critic
        self.actor = ac(num_input=self.env.observation_space.shape[0],
                        # continuous action spaces don't have n attribute,
                        # but in that case num_output isn't used anyways
                        num_output=getattr(self.env.action_space, 'n', 1),
                        **ac_kwargs)

        if critic is None:
            # setting up target alg
            self.target_alg = target_alg(**target_alg_kwargs)
        else:
            # setting up critic
            # giving the first argument like this might lead to problems
            self.critic = critic(self.env.observation_space.shape[0], **critic_kwargs)
            self.critic.optim = critic_optim(self.critic.parameters(), **critic_optim_kwargs) # should this be in one of the kwargs, or passed via v_lr?
            
            # setting up target alg
            self.target_alg = target_alg(critic=self.critic, critic_optim=self.critic.optim,**target_alg_kwargs)
        
        self.ac_alg = ac_alg(self.actor, self.target_alg, **ac_alg_kwargs)


        # for result logging
        self.results = {}

    def run(self):
        '''
        Run experiment
        '''
        #TODO: log more metrics?

        # make sure torch is seeded correctly
        torch.manual_seed(self.seed)

        # for logging
        self.results['returns'] = []
        self.results['step_size'] = []

        #
        for i in range(self.num_iters):
            # sample trajectories
            memory = utils.sample_memory(self.env, self.actor, self.ep_per_iter)

            # get returns for monitoring
            returns = algorithms.get_episode_returns(memory[2], memory[3])
            self.results['returns'].append(returns.mean().item())

            print("Episode {}, Mean Return: {}".format(self.ep_per_iter * (i + 1), returns.mean().item()))
                        
            # train actor 
            step_size = self.ac_alg.train(memory)
            self.results['step_size'].append(step_size)

            # train critic, will just pass if no training available
            if not (self.critic is None):
                self.target_alg.train(memory)
        
        # save results
        np.savez_compressed(self.log_file, returns=self.results['returns'], step_size=self.results['step_size'])



    def load_results(self, log_file='./log.npz'):
        '''
        Load results for a previously run experiment
        Use if you just want to plot stuff without re-running experiments.
        '''
        self.results = np.load(log_file)


    def plot(self, plot_path='./plots/'):
        '''
        Plot the results of the last / last loaded run.
        '''

        # plot step sizes
        plt.figure()
        plt.plot(np.arange(self.num_iters), self.results['step_size'], label='step size')
        plt.ylabel('Step Size')
        plt.xlabel('Iteration')
        plt.legend()
        plt.savefig(plot_path+'step_size.pdf')
        
        # plot returns
        plt.figure()
        plt.plot(np.arange(self.num_iters), self.results['returns'], label='returns')
        plt.xlabel('Iteration')
        plt.ylabel('Return')
        plt.legend()
        plt.savefig(plot_path+'returns.pdf')

        


class mult_seed_exp:

    def __init__(self, experiment_parameters, seeds, log_dir='./'):
        '''
        Args:
            experiment_parameters - dict, parameters for the experiment
            seeds - list or ndarray, seeds you want to run the experiment on
            log_dir - str, top-level dir for saving the results for this set of experiment
        '''
        self.experiment_parameters = experiment_parameters
        self.seeds = seeds
        self.log_dir = log_dir
    
        # for logging
        self.results = {}

    def run(self):
        ' Runs the experiment'
        for i in range(len(self.seeds)):
            
            print('Now running experiment with seed {}'.format(self.seeds[i]))
            # set seed
            self.experiment_parameters['seed'] = self.seeds[i]
            self.experiment_parameters['log_file'] = self.log_dir+str(self.seeds[i])+'_results.npz'

            # set up new experiment
            exp = Experiment(**self.experiment_parameters)
            
            # run experiment
            exp.run()

            # store results
            self.results[self.seeds[i]] = exp.results
        
        # save results
        result_kwargs = {}
        for seed in self.seeds:
            result_kwargs[str(seed)] = self.results[seed]
        np.savez_compressed(self.log_dir + 'all_results.npz', **result_kwargs)


    def get_mean_results(self):
        '''
        Returns the averaged results of all seed runs
        Use this method if you want to plot experiments with different parameters in one plot.
        '''
        # put everything in one array
        returns = np.zeros((len(self.seeds), self.experiment_parameters['num_iters']))
        step_size = np.zeros((len(self.seeds), self.experiment_parameters['num_iters']))
        
        for i in range(len(self.seeds)):
            returns[i,:] = self.results[self.seeds[i]]['returns']
            step_size[i,:] = self.results[self.seeds[i]]['step_size']

        # compute mean and std
        returns_means = np.mean(returns, axis=0)    
        returns_stds = np.std(returns, axis=0)    
        step_size_means = np.mean(step_size, axis=0)    
        step_size_stds = np.std(step_size, axis=0)

        return returns_means, returns_stds, step_size_means, step_size_stds

    def load_results(self, result_file='./all_results.npz'):
        '''
        Load results for a previously run experiment. 
        Use if you just want to plot stuff without re-running experiments.

        result_file - str, path to the result file
        '''
        self.results = np.load(result_file)


    def plot(self, plot_path='./plots/'):
        '''
        Plots the mean and std of return and step size, averaged over the different seeds.
        '''

        # get mean results
        returns_means, returns_stds, step_size_means, step_size_stds = self.get_mean_results()

        # plot returns
        plt.figure()
        plt.plot(np.arange(len(returns_means)), returns_means, label='mean returns')
        plt.fill_between(np.arange(len(returns_means)), returns_means-returns_stds, returns_means+returns_stds, alpha=.5, label=r'$\pm 1\sigma$')
        plt.xlabel('Iteration')
        plt.ylabel('Return')
        plt.legend()
        plt.savefig(plot_path+'avg_returns.pdf')
        
        # plot step sizes
        plt.figure()
        plt.plot(np.arange(len(step_size_means)), step_size_means, label='mean step size')
        plt.fill_between(np.arange(len(step_size_means)), step_size_means-step_size_stds, step_size_means+step_size_stds, alpha=.5, label=r'$\pm 1\sigma$')
        plt.xlabel('Iteration')
        plt.ylabel('Step Size')
        plt.legend()
        plt.savefig(plot_path+'avg_step_size.pdf')
        
        
        
        
        
