import torch
import torch.nn as nn
import numpy as np
from actor import DummyDiscrete
from algorithms import BaselineCriticMC, ActorOnlyMC, NPG, TRPO, get_returns
from utils import sample_memory
import gym


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


seed = 42
num_hidden = 10

env = gym.make('CartPole-v1')
env.seed(seed)
actor = DummyDiscrete(env, num_hidden)

## Use this to use a critic baseline:
critic = Critic(env.observation_space.shape[0], num_hidden)
optimizer = torch.optim.SGD(critic.parameters(), lr=1e-3)
target_alg = BaselineCriticMC(critic, optimizer)
## Or this to just use MC returns (actor-only):
#critic_alg = ActorOnlyMC()

## Choose between NPG and TRPO
#actor_alg = NPG(actor, critic_alg, lr=0.5)
actor_alg = TRPO(actor, target_alg, max_kl=0.01)

for i in range(100):
    # sample 5 episodes
    memory = sample_memory(env, actor, num_episodes=5, render=True)
    returns = get_returns(memory[2], memory[3])
    print("Episode {}, Returns: {}".format(10 * (i + 1), returns.mean().item()))
    # train the critic on those episodes
    target_alg.train(memory)
    # train the actor on those episodes
    actor_alg.train(memory)
