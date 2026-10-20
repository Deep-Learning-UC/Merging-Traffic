import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo, FlattenObservation
import numpy as np


def create_env(scenario="training", render_mode="rgb_array", video_folder=None, seed=None, flatten_observation=False):
    """
    Factory function to create and configure the highway-merge environment.
    
    Args:
        config_type (str): See list of availble configs in "configs" folder
        render_mode (str): "human" for GUI, "rgb_array" for video recording, None for headless training.
        video_folder (str): Path to save MP4s (only if render_mode="rgb_array").
        seed (int): Random seed for reproducibility.
        flatten_observation (bool): Set to True if 1D array output is desired (may not be needed).
        
    Returns:
        gym.Env: The configured Gymnasium environment.
    """

    env = gym.make("merge-v1", render_mode=render_mode)

    # The Base Observation Template
    obs_config = {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
        "order": "sorted"
    }

    # Scenario Logic
    if scenario == "training":
        # Scenario A: The Control
        config = {
            "observation": obs_config,
            "vehicles_count": 15,
            "vehicles_density": 1.0,
            "collision_reward": -5,
            "reward_speed_range": [20, 30],
        }

    elif scenario == "rush_hour":
        # Scenario B: The Wall of Traffic
        # We increase the number of cars the agent sees so it can find gaps
        obs_rush = obs_config.copy()
        obs_rush["vehicles_count"] = 8 
        
        config = {
            "observation": obs_rush,
            "vehicles_count": 35,
            "vehicles_density": 2.0,
            "collision_reward": -10,
            "reward_speed_range": [10, 20], # Slow traffic
        }

    elif scenario == "road_rage":
        # Scenario C: The Noise Test
        obs_noise = obs_config.copy()
        obs_noise["noise"] = 0.05  # Adds 5% Gaussian noise to all features
        
        config = {
            "observation": obs_noise,
            "vehicles_count": 15,
            "vehicles_density": 1.2,
            "collision_reward": -5,
            "reward_speed_range": [20, 35],
        }

    env.unwrapped.configure(config)
    
    # If enabled, flatten the 5x5 Kinematics matrix into a 25-element vector
    if flatten_observation:
        env = FlattenObservation(env)
    
    # Automatically record videos every 50 episodes if a folder is provided
    if video_folder and render_mode == "rgb_array":
        env = RecordVideo(
            env, 
            video_folder=video_folder, 
            episode_trigger=lambda e: e % 50 == 0,
            name_prefix=f"highway-merge-{scenario}"
        )

    # Set Seed
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


if __name__ == "__main__":
    # MANUAL TEST SCRIPT
    # Run this file directly to see the environment in action.
    print("Starting Manual Environment Test...")
    
    # Use "human" mode to see the window (requires a monitor)
    # If running on a server, use render_mode=None
    test_env = create_env(scenario="training", render_mode="human")
    
    obs, info = test_env.reset()
    done = truncated = False
    
    print(f"Observation Space: {test_env.observation_space}")
    print(f"Action Space: {test_env.action_space}")
    
    try:
        while not (done or truncated):
            # Take a random action
            action = test_env.action_space.sample()
            obs, reward, done, truncated, info = test_env.step(action)
            test_env.render()
    except KeyboardInterrupt:
        print("Test stopped by user.")
    finally:
        test_env.close()
        print("Environment closed.")
