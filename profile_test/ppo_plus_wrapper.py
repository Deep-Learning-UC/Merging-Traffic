"""
a lot of this was generated from claude
dude i don't understand like any of it


NOTES:
i was able to figure it out
it's easiest if you start with what the model already has and keep trying to run it and editing it
with each error message
the hardest part is that PPO+ expects gymnasium.Env - which is a singular environment
but we pass in a VecMonitor() instead - which is a series of environment running in parallel
so i had to edit some of the internals of PPO+ to use this VecMonitor() appropriately
"""

"""
SB3-style wrapper around PPO+ so train.py / main.py / evaluate.py can treat it like DQN, PPO and A2C.

Put this next to train.py. It expects the upstream package at profile_test/ppo_plus/ with its
original imports (from ppo_plus.xxx import ...).

Why the training loop is copied here instead of calling ppo_plus.agent.run_training():
run_training() never hands the agent back until it finishes, so there would be no checkpoints
and no early stop at goal_exit_rate. The loop below mirrors run_training() line for line, and
calls Progress (your existing callback) once per policy update, i.e. every `policy_steps` steps.
This replaces the Metrics class, so you don't need one.
"""
import pickle
import zipfile
from collections import deque
from dataclasses import replace
from pathlib import Path

import flax.serialization
import gymnasium as gym
import jax

from ppo_plus.agent import PPOPlusConfig, RunConfig, _create_run_state, create_learner, run_training
from ppo_plus.environments import rollout_policy
import numpy as np

from profile_test_environment import make_env
from train import Progress

# settings that belong to RunConfig; everything else in TRAIN["settings"]["PPOPlus"] goes to PPOPlusConfig
RUN_KEYS = {"buffer_size", "policy_steps", "num_epochs", "on_policy_data"}


# records every finished life the way SB3's VecMonitor does, so Progress can read exit/crash/timeout rates
class EpisodeLog(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = deque(maxlen=100)  # last 100 lives: {"r": return, "outcome": ...}
        self.finished = 0
        self._return = 0.0

    def reset(self, **kwargs):
        self._return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._return += reward
        if terminated or truncated:
            self.lives.append({"r": self._return, "outcome": info.get("outcome")})
            self.finished += 1
        return obs, reward, terminated, truncated, info


class PPOPlus:
    single_env = True  # train() builds ONE gymnasium env for this model, not a SubprocVecEnv

    # policy / policy_kwargs / verbose only exist so train() can call it exactly like an SB3 algorithm
    def __init__(self, policy, env, gamma=0.99, seed=0, policy_kwargs=None, verbose=0, **settings):
        # run_kw = {k: settings.pop(k) for k in list(settings) if k in RUN_KEYS}
        # self.run_kw = {"buffer_size": 50_000, "policy_steps": 5_000, "num_epochs": 10, "on_policy_data": False, **run_kw}
        self.learner_config = PPOPlusConfig(discount=gamma, seed=seed, **settings)  # typos in settings raise here
        # self.env = env # EpisodeLog(env)
        self.observation_space, self.action_space = env.observation_space, env.action_space
        # self.ep_info_buffer = self.env.lives  # Progress reads this
        # self.num_timesteps = 0
        self.agent = None
        # self.policy = None  # main.ask_model checks model.policy; None makes it fall through to predict()
        # self.logger = None
        self.env = env

    def get_env(self):  # BaseCallback.init_callback asks for this
        return self.env

    # ------------------------------------------------------------------ training
    def learn(self, total_timesteps, callback=None):
        print("PPO PLUS LEARN")
        state = run_training(
            learner_config=self.learner_config,
            # run_config=RunConfig(total_steps=300_000, buffer_size=50_000, policy_steps=5_000,
            #   num_epochs=10, on_policy_data=False, log_interval=10**9),
            run_config=RunConfig(total_steps=1_000, buffer_size=50_000, policy_steps=1_000,
                num_epochs=10, on_policy_data=False, log_interval=10**9), # log_interval is just supposed to be > total_steps, to make it work
            env=self.env,
            eval_env=None,
            metrics=Metrics()
        )
        # cfg, run = self.learner_config, RunConfig(total_steps=total_timesteps, **self.run_kw)
        # state = _create_run_state(cfg, run, self.env)
        # self.agent = state.agent
        # seen = 0
        # if callback is not None:
        #     callback.init_callback(self)
        #     callback.on_training_start({}, {})

        # while state.environment_steps < total_timesteps:
        #     next_rng, rng = jax.random.split(state.rollout_rng)
        #     replay = state.replay_buffer
        #     if run.on_policy_data:
        #         replay.reset()
        #     # 1) collect policy_steps steps with the current policy
        #     _, _, n = rollout_policy(
        #         state.agent, self.env, rng, discount=cfg.discount, max_steps=run.policy_steps,
        #         reset_seed=cfg.seed if state.environment_steps == 0 else None,
        #         replay_buffer=replay, actor_buffer=state.actor_buffer,
        #     )
        #     # 2) update critics on the replay buffer, actor on the fresh batch
        #     agent = state.agent
        #     critic_data, actor_data = replay.get_all(), state.actor_buffer.get_all()
        #     for _ in range(run.num_epochs):
        #         agent = agent.update_critics_seq(critic_data)
        #         agent, self.last_info = agent.update_actor_seq(actor_data)
        #     state = replace(state, agent=agent, replay_buffer=replay, rollout_rng=next_rng,
        #                     environment_steps=state.environment_steps + n)
        #     self.agent, self.num_timesteps = agent, state.environment_steps

        #     # 3) hand control to Progress: it logs, checkpoints (calls self.save) and can say stop
        #     if callback is not None:
        #         callback.update_locals({"dones": [1] * (self.env.finished - seen)})
        #         seen = self.env.finished
        #         if not callback.on_step():
        #             break
        self.agent = state.agent
        return self

    # ------------------------------------------------------------------ using the model
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """One unbatched observation in, (action, None) out, like SB3."""
        if deterministic:
            action = self.agent.deterministic_action(observation)
        else:
            self._rng = jax.random.split(getattr(self, "_rng", jax.random.PRNGKey(0)))[0]
            action = self.agent.sample_actions(observation, seed=self._rng)[0]
        return np.asarray(action, dtype=np.float32), None

    # ------------------------------------------------------------------ save / load
    # a real .zip (evaluate.py hard-codes trained/NAME.zip): actor weights + what's needed to rebuild the network
    def save(self, path):
        path = Path(path)
        path = path if path.suffix == ".zip" else Path(str(path) + ".zip")
        meta = {"learner_config": self.learner_config, "observation_space": self.observation_space,
                "action_space": self.action_space}
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("meta.pkl", pickle.dumps(meta))
            z.writestr("actor.msgpack", flax.serialization.to_bytes(self.agent.actor.params))

    @classmethod
    def load(cls, path, device=None, **_):
        with zipfile.ZipFile(path) as z:
            meta, weights = pickle.loads(z.read("meta.pkl")), z.read("actor.msgpack")
        self = cls.__new__(cls)  # skip __init__: there is no env when loading
        self.learner_config = meta["learner_config"]
        self.observation_space, self.action_space = meta["observation_space"], meta["action_space"]
        self.policy, self.logger, self.num_timesteps = None, None, 0
        agent = create_learner(self.learner_config, self.observation_space.sample()[None], self.action_space.sample()[None])
        params = flax.serialization.from_bytes(agent.actor.params, weights)
        self.agent = agent.replace(actor=agent.actor.replace(params=params))
        return self

# TODO what does wandb metrics do? which is what metrics are supposed to be
class Metrics:
    def log(self, dict, step, commit=True):
        print(f"\nMETRICS: {dict} step: {step}\n")
        return

    def finish(self):
        print("PPOPLUS METRICS FINISH")
        return

if __name__ == "__main__":

    # # smoke test 
    # # i think it's running
    # run_training(
    #     learner_config=PPOPlusConfig(),
    #     # run_config=RunConfig(total_steps=300_000, buffer_size=50_000, policy_steps=5_000,
    #     #   num_epochs=10, on_policy_data=False, log_interval=10**9),
    #     run_config=RunConfig(total_steps=10_000, buffer_size=50_000, policy_steps=1_000,
    #       num_epochs=10, on_policy_data=False, log_interval=10**9),
    #     env=make_env(),
    #     eval_env=None,
    #     metrics=None #Metrics()
    # )
    print("start")
    pp = PPOPlus(None, make_env())
    print("ppo")
    pp.learn(1_000, Progress("name", 100))