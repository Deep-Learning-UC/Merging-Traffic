"""
Copied from Claude - model_research_two.md prompt 2
"""
"""
Benchmark template: train + evaluate SB3 algorithms on highway-env over several seeds.

Install:  pip install gymnasium highway-env stable-baselines3 torch numpy
Run:      python highway_sb3_benchmark.py --algo DQN --timesteps 50000 --seeds 0 1 2

This uses the built-in `merge-v0` as a stand-in. Swap ENV_ID for your own custom
merge-and-exit environment once it is registered.

The point of the template is the *experimental protocol*:
  - same environment + same reward for every algorithm
  - multiple seeds per algorithm
  - report task metrics (collision rate, speed, return), not just return
"""
import argparse

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.utils import set_random_seed

try:  # newer gymnasium versions need explicit registration
    gym.register_envs(highway_env)
except AttributeError:
    pass

ENV_ID = "merge-v0"

# Extra keys are harmless if a given env ignores them; check the docs for your version.
ENV_CONFIG = {
    "collision_reward": -1,
    "high_speed_reward": 0.2,
}

ALGOS = {
    "DQN": (DQN, dict(
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4, buffer_size=15_000, learning_starts=200,
        batch_size=32, gamma=0.8, train_freq=1, gradient_steps=1,
        target_update_interval=50, exploration_fraction=0.7,
    )),
    "PPO": (PPO, dict(
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4, n_steps=512, batch_size=64, gamma=0.9,
    )),
}


def make_env():
    return gym.make(ENV_ID, config=ENV_CONFIG)


def evaluate(model, seed: int, n_episodes: int = 50) -> dict:
    """Run the trained policy deterministically and collect task metrics."""
    env = make_env()
    crashes, returns, mean_speeds, lengths = 0, [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + seed * 1_000 + ep)
        done, ep_ret, speeds, info = False, 0.0, [], {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            speeds.append(info.get("speed", np.nan))
            done = terminated or truncated
        crashes += int(info.get("crashed", False))
        returns.append(ep_ret)
        mean_speeds.append(np.nanmean(speeds))
        lengths.append(len(speeds))
    env.close()
    return {
        "collision_rate": crashes / n_episodes,
        "mean_return": float(np.mean(returns)),
        "mean_speed": float(np.mean(mean_speeds)),
        "mean_ep_length": float(np.mean(lengths)),   # proxy for "time to finish"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS, default="DQN")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    algo_cls, algo_kwargs = ALGOS[args.algo]
    all_results = []

    for seed in args.seeds:
        set_random_seed(seed)
        env = make_env()
        model = algo_cls("MlpPolicy", env, seed=seed, verbose=0, **algo_kwargs)
        model.learn(total_timesteps=args.timesteps)
        model.save(f"{args.algo}_seed{seed}")
        res = evaluate(model, seed)
        all_results.append(res)
        print(f"[{args.algo} seed {seed}] {res}")
        env.close()

    print(f"\n=== {args.algo} summary over {len(args.seeds)} seeds (mean +/- std) ===")
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        print(f"{key:>16}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")


if __name__ == "__main__":
    main()
