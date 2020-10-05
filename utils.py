import torch
import numpy as np
from collections import deque


def sample_memory(env, actor, num_episodes, render=False):
    """Sample episodes from an environment using an Actor to select actions.

    Args:
        env: an OpenAI Gym environment instance
        actor: an Actor instance (needs a sample_action method)
        num_episodes: (int) number of episodes to sample
        render: (bool) whether to render the environment after every step
                (should be turned off during long training runs for performance reasons)

    Returns:
        A tuple (states, actions, rewards, masks) where all elements are Tensors and
          - states has shape (N, observation_space_dim)
          - actions has shape (N, )
          - rewards has shape (N, )
          - masks has shape (N, ); it is 0 if the episode is done and 1 otherwise

    Note: for now, only environments with scalar actions are supported, as can be seen from
    the signature"""

    actor.eval()
    memory = []

    for i in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # not entirely sure what this does, I think it normalizes and clips the state values?
            #state = running_state(state)
            action = actor.sample_action(torch.Tensor(state).unsqueeze(0)).item()
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()
            #next_state = running_state(next_state)

            if done:
                mask = 0
            else:
                mask = 1

            memory.append((state, action, reward, mask))

            state = next_state
    return [torch.tensor(xs) for xs in zip(*memory)]


# Some more utilities that are used in algorithms.py, probably not needed elsewhere
def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length
