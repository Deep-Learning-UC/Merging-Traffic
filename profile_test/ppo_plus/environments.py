"""Environment adapters, construction, horizons, and rollouts."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import numpy as np
from gymnasium.core import Env
from gymnasium.spaces import Box


# def _spec_to_box(specification, dtype=np.float32):
#     from dm_env import specs

#     bounds = []
#     for item in specification:
#         size = int(np.prod(item.shape))
#         if isinstance(item, specs.BoundedArray):
#             bounds.append(
#                 (
#                     np.broadcast_to(item.minimum, item.shape),
#                     np.broadcast_to(item.maximum, item.shape),
#                 )
#             )
#         else:
#             limit = np.full(size, np.inf)
#             bounds.append((-limit, limit))
#     low, high = zip(*bounds)
#     return Box(
#         np.concatenate([np.ravel(value) for value in low]).astype(dtype),
#         np.concatenate([np.ravel(value) for value in high]).astype(dtype),
#         dtype=dtype,
#     )


def _flatten_obs(obs, dtype=np.float32):
    pieces = [np.array([value]) if np.isscalar(value) else value.ravel() for value in obs.values()]
    return np.concatenate(pieces).astype(dtype)


# class DMCGym(Env):
#     """Adapt one DeepMind Control Suite task to the Gymnasium API."""

#     # def __init__(self, domain, task):
#     #     from dm_control import suite

#     #     self._env = suite.load(domain, task)
#     #     self.observation_space = _spec_to_box(self._env.observation_spec().values())
#     #     self.action_space = _spec_to_box([self._env.action_spec()])

#     def step(self, action):
#         timestep = self._env.step(np.asarray(action, dtype=np.float32))
#         return (
#             _flatten_obs(timestep.observation),
#             timestep.reward,
#             False,
#             timestep.last(),
#             {"discount": timestep.discount},
#         )

#     def reset(self, seed=None, options=None):
#         if seed is not None:
#             self._env.task._random = np.random.RandomState(seed)
#         return _flatten_obs(self._env.reset().observation), {}


# def create_environments(env_name: str) -> tuple[gym.Env, gym.Env]:
#     """Create independent training and evaluation environments."""
#     gym_envs = {
#         "InvertedDoublePendulum-v5",
#         "Walker2d-v5",
#         "HalfCheetah-v5",
#         "Hopper-v5",
#         "Ant-v5",
#         "Humanoid-v5",
#     }
#     if env_name in gym_envs:
#         return (
#             gym.wrappers.RecordEpisodeStatistics(gym.make(env_name, max_episode_steps=1_000)),
#             gym.wrappers.RecordEpisodeStatistics(gym.make(env_name, max_episode_steps=1_000)),
#         )

#     domain, task = env_name.split("-")
#     return DMCGym(domain, task), DMCGym(domain, task)


def get_max_steps_for_env(env_name: str) -> int:
    """Return the paper's training horizon for an environment."""
    if env_name in {
        "Humanoid-v5",
        "HumanoidStandup-v5",
        "dog-stand",
        "dog-walk",
        "dog-trot",
        "dog-run",
    }:
        return 5_000_000
    if env_name == "InvertedDoublePendulum-v5":
        return 200_000
    return 1_000_000


def rollout_policy(
    agent: Any,
    env: Any,
    exploration_rng: jax.Array,
    discount: float,
    max_steps: int | None = None,
    max_rollouts: int | None = None,
    eval: bool = False,
    reset_seed: int | None = None,
    replay_buffer: Any = None,
    actor_buffer: Any = None,
) -> tuple[float, float, int]:
    """Collect transitions until a step or completed-episode limit is reached."""
    n_steps, n_rollouts, discount_multiplier = 0, 0, 1.0
    discounted_returns, returns = [], []
    discounted_return, episode_return = 0.0, 0.0

    if actor_buffer is not None:
        actor_buffer.reset()

    env.seed(reset_seed) # env.reset does not have seed param in vectorized
    observation = env.reset()
    while (max_steps is None or n_steps < max_steps) and (
        max_rollouts is None or n_rollouts < max_rollouts
    ):
        if eval:
            # this should never run now
            action = agent.deterministic_action(observation)
            log_probability, pre_action = 0.0, action
        else:
            exploration_rng, action_rng = jax.random.split(exploration_rng)
            action, log_probability, pre_action = agent.sample_actions(observation, seed=action_rng)
            # this is batched

        action = np.asarray(action) # is this necessary?
        next_observation, reward, dones, _ = env.step(action)

        for i in range(env.num_envs):
            discounted_return += reward[i] * discount_multiplier
            episode_return += reward[i]
            mask = float(not dones[i])

            transition = {
                "observations": observation[i],
                "actions": action[i],
                "rewards": reward[i],
                "masks": mask,
                "truncateds": dones[i],
                "next_observations": next_observation[i],
                "log_probs": log_probability[i],
                "pre_actions": pre_action[i],
            }
            # print("TRANSITION", transition)
            if replay_buffer is not None:
                replay_buffer.add_transition(transition)
            if actor_buffer is not None:
                actor_buffer.add_transition(transition)

            # observation = next_observation
            discount_multiplier *= discount * mask
            n_steps += 1

            if dones[i]:
                # print("TERM/TRUNC")
                discounted_returns.append(discounted_return)
                returns.append(episode_return)
                discounted_return, episode_return = 0.0, 0.0
                # observation, _ = env.reset() called automatically? when vectorized
                discount_multiplier = 1.0
                n_rollouts += 1
        observation = next_observation # i think? 

    return np.mean(discounted_returns), np.mean(returns), n_steps


def rollout_policy_ppo(agent: Any, env: Any, num_rollouts: int = 5) -> float:
    """Evaluate a deterministic baseline policy for complete episodes."""
    returns = []
    observation, _ = env.reset()

    while len(returns) < num_rollouts:
        action = agent.deterministic_action(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if "episode" in info:
            returns.append(info["episode"]["r"])
        if terminated or truncated:
            observation, _ = env.reset()

    return np.mean(returns)
