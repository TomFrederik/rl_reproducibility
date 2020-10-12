import gym
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

env = gym.make('Pendulum-v0')

model = TRPO(MlpPolicy, env, verbose=1)

print(env.action_space.shape)


while True:
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    time.sleep(5)


