"""
copy paste from https://highway-env.farama.org/quickstart/#quickstart
see for exmample - https://colab.research.google.com/github/Farama-Foundation/HighwayEnv/blob/main/scripts/sb3_highway_dqn.ipynb
"""

import gymnasium
import highway_env
from stable_baselines3 import DQN
from tqdm.notebook import trange
# from utils import show_videos
from gymnasium.wrappers import RecordVideo
import IProgress
import matplotlib.pyplot as plt

"""

import gymnasium as gym

# Opens a window on your screen
env = gym.make("CartPole-v1", render_mode="human")

# Or: returns frames as numpy arrays (for recording video, no window)
env = gym.make("CartPole-v1", render_mode="rgb_array")


~~~~~save videos~~~~~

from gymnasium.wrappers import RecordVideo

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: ep % 10 == 0)


"""

env = gymnasium.make("highway-fast-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", episode_trigger = lambda ep: ep % 100 == 0)
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
            #   tensorboard_log="highway_dqn/") # tensorboard not installed
            )
# model.learn(int(2e4))
# model.save("highway_dqn/model")

# Load and test saved model
model = DQN.load("highway_dqn/model")
# while True:
for episode in range(3): # need to import IProgress
    env = RecordVideo(env, video_folder="videos", name_prefix=f"rl-{episode}-video")
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
env.close()
# show_videos()