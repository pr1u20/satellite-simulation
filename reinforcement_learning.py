import gymnasium as gym
from utils import SatelliteEnv

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env

env = SatelliteEnv(render = False)
env.realTime = False #To speed up training
check_env(env)

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000000, log_interval=10, progress_bar=True)
env.close()

env = SatelliteEnv(render = True)
obs, info = env.reset()

done = False
while not done:

    action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)
