import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from actor import DummyDiscrete, DummyCont
from algorithms import GAE, NPG, TRPO
from actor import Actor
from utils import sample_memory
from experiment_class import Experiment, mult_seed_exp


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
            
            
            #self.log_var = nn.parameter.Parameter(data=torch.Tensor([0]), requires_grad=True)

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
        #var = torch.exp(self.log_var)
        #params = torch.cat((params, torch.ones_like(params) * var), dim=1)

        return params


target_alg_kwargs = {'batch_size':16, 'epochs':3, 'gamma':0.9, 'lamda':0.97}

ac_kwargs = {'num_hidden': 64}
critic_kwargs = {'num_hidden':64}
critic_optim_kwargs = {'lr':3e-4}

all_ac_alg_kwargs = {
    TRPO: {'max_kl': 0.01},
    NPG: {'lr': 0.1}
}

all_cvs = {}
means = {}

seeds = [10,11,12,13,14,15,16,17,18,19]

env_str = "Pendulum-v0"

for alg in [NPG, TRPO]:
    cvs = []
    os.makedirs("kl/logs/{}/long/{}".format(env_str, alg.__name__), exist_ok=True)
    os.makedirs("kl/plots/{}/long/".format(env_str), exist_ok=True)

    plt.figure()
    for i, seed in enumerate(seeds):
        print("===================================")
        print("{} [{}/{}]".format(alg.__name__, i+1, len(seeds)))
        print("===================================")
        ac_alg_kwargs = all_ac_alg_kwargs[alg]

        experiment_parameters = {'seed': seed,
                                 'env_str': env_str,
                                 'ac': ContActor,
                                 'ac_kwargs': ac_kwargs,
                                 'ac_alg': alg,
                                 'ac_alg_kwargs': ac_alg_kwargs,
                                 'target_alg': GAE,
                                 'target_alg_kwargs': target_alg_kwargs,
                                 'critic': Critic,
                                 'critic_kwargs': critic_kwargs,
                                 'critic_optim': torch.optim.Adam,
                                 'critic_optim_kwargs': critic_optim_kwargs,
                                 'num_iters': 400,
                                 'ep_per_iter': 5,
                                 'log_file': './kl/logs/{}/long/{}/{}_log.npz'.format(env_str, alg.__name__, seed)}

        experiment = Experiment(**experiment_parameters)

        experiment.run()

        kls = torch.tensor(experiment.results['kl'])
        mean = torch.mean(kls).item()
        std = torch.std(kls).item()
        cvs.append(std/mean)

        plt.plot(kls, alpha=0.2, color="blue")

        #experiment.plot('./kl/plots/cartpole/{}/{}_'.format(alg.__name__, seed))

        # show one example episode as a sanity check
        #sample_memory(experiment.env, experiment.actor, 1, True)
    print(cvs)
    print(np.mean(cvs))
    means[alg] = np.mean(cvs)
    all_cvs[alg] = cvs
    plt.xlabel("Iteration")
    plt.ylabel("KL update")
    plt.savefig("kl/plots/{}/long/{}.pdf".format(env_str, alg.__name__))

plt.figure()
# for i, alg in enumerate([NPG, TRPO]):
#     plt.scatter(
#         [i] * len(all_cvs[alg]),
#         all_cvs[alg],
#         color="blue",
#         s=4
#     )
#     plt.scatter(i, means[alg], color="blue", s=16)
plt.boxplot(all_cvs.values(), labels=[x.__name__ for x in all_cvs])
plt.ylabel("Coefficient of variation σ/μ")
plt.savefig("kl/plots/{}/long/cvs.pdf".format(env_str))
