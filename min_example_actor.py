import torch
from actor import Actor
from actor import DummyCont
from actor import DummyDiscrete
import gym


seed = 3141
torch.manual_seed(seed)
num_hidden = 10

print('Testing continuous actor on MountainCarContinuous-v0!')

# init env and policy
env = gym.make('MountainCarContinuous-v0')
env.seed(seed)
policy = DummyCont(env, num_hidden)
obs_list = []
action_list = []

# reset env
done = False
obs = env.reset()
obs_list.append(obs)
obs = torch.from_numpy(obs).reshape(1,-1).float() # make sure obs has batch dimension and is cast to float32 instead of float64


# check if we can batch process states
print('Computing dists for three sets of params')
params = policy.forward(torch.cat((obs,obs,obs), dim=0))
dist = policy.get_dist(params)
print('batch shape of dist is ',dist.batch_shape)

# sample trajectory
print('Sampling a trajectory')
while not done:    
    # sample action
    action = policy.sample_action(obs)
    
    # take step
    obs, rew, done, _ = env.step(action)
    action_list.append(action.detach().numpy())
    
    # don't include last state
    if not done:
        obs_list.append(obs)
        obs = torch.from_numpy(obs).reshape(1, -1).float()
    
print('\nTesting log_probs')
# convert lists to tensor
obs = torch.Tensor(obs_list)    
actions = torch.Tensor(action_list)    

# get logprobs
log_probs = policy.get_log_probs(obs, actions)

print('\nLogprobs are tensor of shape {}.'.format(log_probs.shape))
#print(log_probs)

print('\nTesting KL divergence')
kl = policy.get_kl(obs)
print('Expecting KL of 0, got {}'.format(kl))

print('\n\n####################\n####################\n\n')

print('Testing discrete control on MountainCar-v0')

#TODO: Implement this
# init env and policy
env = gym.make('MountainCar-v0')
env.seed(seed)
policy = DummyDiscrete(env, num_hidden)
obs_list = []
action_list = []



# reset env
done = False
obs = env.reset()
obs_list.append(obs)
obs = torch.from_numpy(obs).reshape(1,-1).float() # make sure obs has batch dimension and is cast to float32 instead of float64


# check if we can batch process states
print('Computing dists for two sets of params')
params = policy.forward(torch.cat((obs,obs), dim=0))
dist = policy.get_dist(params)
print('batch shape of dist is ',dist.batch_shape)

# sample trajectory
print('Sampling a trajectory')
while not done:    
    # sample action
    action = policy.sample_action(obs).item()
    
    # take step
    obs, rew, done, _ = env.step(action)
    action_list.append(action)
    
    # don't include last state
    if not done:
        obs_list.append(obs)
        obs = torch.from_numpy(obs).reshape(1, -1).float()
    
print('\nTesting log_probs')
# convert lists to tensor
obs = torch.Tensor(obs_list)    
actions = torch.Tensor(action_list)    

# get logprobs
log_probs = policy.get_log_probs(obs, actions)

print('Logprobs are tensor of shape {}.'.format(log_probs.shape))
#print(log_probs)

print('\nTesting KL divergence')
kl = policy.get_kl(obs)
print('Expecting KL of 0, got {}'.format(kl))
