"""
a lot of this was generated from claude
dude i don't understand like any of it

i frankesteined it to work. i apologize in advance


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

class PPOPlus:
    # policy / policy_kwargs / verbose only exist so train() can call it exactly like an SB3 algorithm
    def __init__(self, policy, env, gamma=0.99, seed=0, policy_kwargs=None, verbose=0, **settings):
        """
        settings = {
            "hidden_dims"
            "actor_lr"
            "critic_lr"
            "temp_lr"
            "momentum"
            "clipping_ratio"
            "total_steps"
            "buffer_size"
            "policy_steps"
            "num_epochs"
            "on_policy_data"
        }
        """
        self.learner_config = PPOPlusConfig(
            discount=gamma, 
            seed=seed, 
            hidden_dims=settings["hidden_dims"],
            actor_lr=settings["actor_lr"],
            critic_lr=settings["critic_lr"],
            temp_lr=settings["temp_lr"],
            momentum=settings["momentum"],
            clipping_ratio=settings["clipping_ratio"],
        ) 
        self.run_config = RunConfig(
            total_steps=settings["total_steps"], 
            buffer_size=settings["buffer_size"], 
            policy_steps=settings["policy_steps"],
            num_epochs=settings["num_epochs"], 
            on_policy_data=settings["on_policy_data"], 
            log_interval=10**9 # just has to be higher than total_steps
        )
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.num_timesteps = 0
        self.agent = None
        self.policy = None  # main.ask_model checks model.policy; None makes it fall through to predict()
        self.logger = None
        self.env = env

    def get_env(self):  # BaseCallback.init_callback asks for this
        return self.env

    # ------------------------------------------------------------------ training
    def learn(self, total_timesteps, callback=None):
        # run_training builds the agent and trains all in one. it's provided by the PPO+ repo
        state = run_training(
            learner_config=self.learner_config,
            run_config=self.run_config,
            env=self.env,
            eval_env=None,
            metrics=Metrics()
        )
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
    # def __init__(self, progress: Progress):
    #     self.p: Progress = progress # train.py Progress() class
    # integrating with progress doesn't work :(

    def log(self, dict, step, commit=True):
        # self.p._on_step()
        print(f"METRICS: {dict} step: {step}\n")
        return

    def finish(self):
        print("PPOPLUS METRICS FINISH")
        return

if __name__ == "__main__":
    # smoke test 

    # without wrapper
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

    # with wrapper
    pp = PPOPlus(None, make_env())
    pp.learn(1_000, Progress("name", 100))