import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

env = gym.make('MountainCarContinuous-v0')

model = TRPO(MlpPolicy, env, verbose=1)

print(env.action_space.shape)

'''
model.learn(total_timesteps=250000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
'''