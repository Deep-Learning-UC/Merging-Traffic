import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import os

from environment import create_env

def main():
    video_path = "./recordings"
    os.makedirs(video_path, exist_ok=True)

    # We use render_mode="rgb_array" to allow the RecordVideo wrapper to work
    env = create_env(
        scenario="training", 
        render_mode="rgb_array", 
        video_folder=video_path
    )

    print(f"Recording started. Videos will be saved to: {os.path.abspath(video_path)}")

    # The wrapper in environment.py is set to record every 50 episodes, 
    # but since this is a test, let's just run 1 episode.
    for episode in range(1):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            # For now, we just use random actions
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")

    # Close the env to finalize the MP4 file
    env.close()
    print("Recording finished and file saved.")

    #===========#
    # Rush Hour #
    #===========#
    env = create_env(
            scenario="rush_hour", 
            render_mode="rgb_array", 
            video_folder=video_path
        )

    print(f"Recording started. Videos will be saved to: {os.path.abspath(video_path)}")
    
    # The wrapper in environment.py is set to record every 50 episodes, 
    # but since this is a test, let's just run 1 episode.
    for episode in range(1):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            # For now, we just use random actions
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
    
    # Close the env to finalize the MP4 file
    env.close()
    print("Recording finished and file saved.")

    #===========#
    # Road Rage #
    #===========#
    env = create_env(
                scenario="road_rage", 
                render_mode="rgb_array", 
                video_folder=video_path
            )
    
    print(f"Recording started. Videos will be saved to: {os.path.abspath(video_path)}")
    
    # The wrapper in environment.py is set to record every 50 episodes, 
    # but since this is a test, let's just run 1 episode.
    for episode in range(1):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            # For now, we just use random actions
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
    
    # Close the env to finalize the MP4 file
    env.close()
    print("Recording finished and file saved.")

    
if __name__ == "__main__":
    main()