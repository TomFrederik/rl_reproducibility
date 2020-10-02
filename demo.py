import torch
import torch.nn as nn
import numpy as np
from actor import DummyDiscrete
from algorithms import ActorOnlyMC, NPG, get_returns
from utils import sample_memory
import gym


# class Critic(nn.Module):
#     def __init__(self, num_inputs, num_hidden):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(num_inputs, num_hidden)
#         self.fc2 = nn.Linear(num_hidden, num_hidden)
#         self.fc3 = nn.Linear(num_hidden, 1)
#         self.fc3.weight.data.mul_(0.1)
#         self.fc3.bias.data.mul_(0.0)

#     def forward(self, x):
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         v = self.fc3(x)
#         return v


seed = 42
num_hidden = 10

env = gym.make('CartPole-v1')
env.seed(seed)
actor = DummyDiscrete(env, num_hidden)
#critic = Critic(env.observation_space.shape[0], num_hidden)
critic_alg = ActorOnlyMC()
actor_alg = NPG(actor, critic_alg)

for i in range(100):
    memory = sample_memory(env, actor, 10, render=True)
    returns = get_returns(memory[2], memory[3])
    print("Episode {}, Returns: {}".format(10*i, returns.mean().item()))
    critic_alg.train(memory)
    actor_alg.train(memory)
