import torch
import numpy as np
from collections import deque


def sample_memory(env, actor, num_episodes, render=False):
    """Sample episodes from an environment using an Actor to select actions.

    Returned memory has at least num_steps steps but usually has a few more because
    we sample until the episode is finished.

    If an episode takes more than 10000 steps, it is stopped at that point.
    For environments with really long episodes, we may need another method."""
    actor.eval()
    memory = []

    steps = 0
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
