"""PPO+ learner, update rules, and experiment configuration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
import tqdm
from flax.training.train_state import TrainState

from ppo_plus.environments import create_environments, get_max_steps_for_env, rollout_policy
from ppo_plus.kernels import (
    actor_minibatch_indices,
    compute_gae,
    critic_minibatch_indices,
    critic_mse,
    critic_target,
    policy_objective,
    tanh_squash_log_prob,
    temperature_objective,
)
from ppo_plus.networks import OriginalCritic, Policy, ensemblize
from ppo_plus.replay import ReplayBuffer
from ppo_plus.tracking import setup_wandb

Batch = dict[str, Any]


def _select_batch(index, batches):
    return jax.tree.map(lambda array: array[index], batches)


def _update_critic_body(index, state):
    agent, batches = state
    return agent.update_critics(_select_batch(index, batches)), batches


class Temperature(nn.Module):
    """Scalar temperature represented in log space."""

    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


class PPOPlusAgent(flax.struct.PyTreeNode):
    """Immutable PPO+ learner state and update operations."""

    rng: jax.Array
    critic: TrainState
    actor: TrainState
    temp: TrainState
    config: PPOPlusConfig = flax.struct.field(pytree_node=False)
    target_entropy: float = flax.struct.field(pytree_node=False)

    def update_critics(self, batch: Batch) -> PPOPlusAgent:
        """Apply one critic gradient update."""
        new_rng, _, next_key = jax.random.split(self.rng, 3)

        def critic_loss_fn(critic_params):
            next_actions, next_log_probs, _ = self.sample_actions(
                batch["next_observations"], seed=next_key
            )
            next_q = self.critic.apply_fn(
                {"params": critic_params}, batch["next_observations"], next_actions
            )
            target_q = critic_target(
                batch["rewards"],
                batch["masks"],
                next_q,
                next_log_probs,
                self.temp.apply_fn({"params": self.temp.params}),
                self.config.discount,
                min_target=self.config.min_target,
            )
            q = self.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            return critic_mse(q, target_q)

        grads = jax.grad(critic_loss_fn)(self.critic.params)
        new_critic = self.critic.apply_gradients(grads=grads)
        return self.replace(rng=new_rng, critic=new_critic)

    @partial(jax.jit, static_argnames=("sample_pool_batches",))
    def update_critics_seq(
        self,
        transitions: Batch,
        sample_pool_batches: int | None = None,
    ) -> PPOPlusAgent:
        """Update critics over the current replay-buffer batch schedule."""
        batch_size = 250
        indices, num_batches = critic_minibatch_indices(
            self.rng,
            transitions["observations"].shape[0],
            sample_pool_batches,
            batch_size,
        )

        batches = jax.vmap(lambda index: jax.tree.map(lambda array: array[index], transitions))(
            indices
        )
        agent, _ = jax.lax.fori_loop(
            0,
            num_batches,
            _update_critic_body,
            (self, batches),
        )

        return agent

    @jax.jit
    def update_actor_seq(self, batch: Batch) -> tuple[PPOPlusAgent, dict[str, jax.Array]]:
        """Update the actor and temperature over one rollout batch."""

        def actor_loss_fn(
            actor_params,
            advantages,
            batch,
            indices,
        ):
            batch = jax.tree.map(lambda array: array[indices], batch)
            advantages = advantages[indices]

            masks = batch["masks"]
            old_log_probs = batch["log_probs"]
            distribution = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            pre_actions = batch["pre_actions"]
            pre_log_probs = distribution.log_prob(pre_actions)
            return policy_objective(
                pre_log_probs,
                pre_actions,
                old_log_probs,
                masks,
                advantages,
                self.config.clipping_ratio,
                tanh_squash_actions=self.config.tanh_squash_actions,
            )

        def temp_loss_fn(temp_params, entropy, target_entropy):
            temperature = self.temp.apply_fn({"params": temp_params})
            return temperature_objective(temperature, entropy, target_entropy)

        new_rng, current_key, _ = jax.random.split(self.rng, 3)

        observations, next_observations = batch["observations"], batch["next_observations"]
        observations = jnp.concatenate([observations, next_observations[-1][None]], axis=0)

        def evaluate(observations, key):
            actions, log_probs, _ = self.sample_actions(observations, seed=key)
            q_values = self.critic.apply_fn({"params": self.critic.params}, observations, actions)
            return jnp.mean(q_values, axis=0), log_probs

        if self.config.gae_lambda > 0.0:
            value_samples, log_prob_samples = jax.vmap(evaluate, in_axes=(None, 0))(
                observations, jax.random.split(current_key, 10)
            )

            values = jnp.mean(value_samples, axis=0)
            sampled_log_probs = jnp.mean(log_prob_samples, axis=0)
            values -= self.temp.apply_fn({"params": self.temp.params}) * sampled_log_probs
            current_values, next_values = values[:-1], values[1:]

            rewards = (
                batch["rewards"]
                - self.temp.apply_fn({"params": self.temp.params}) * batch["log_probs"]
            )
            dones = jnp.bool(1 - batch["masks"])
            truncations = jnp.bool(batch["truncateds"])

            advantages = compute_gae(
                rewards.squeeze(),
                current_values.squeeze(),
                next_values.squeeze(),
                dones.squeeze(),
                truncations.squeeze(),
                gamma=self.config.discount,
                lam=self.config.gae_lambda,
            )
            advantages = advantages.reshape(-1)

        else:
            value_samples, log_prob_samples = jax.vmap(evaluate, in_axes=(None, 0))(
                batch["observations"], jax.random.split(current_key, 10)
            )
            values = jnp.mean(value_samples, axis=0)
            sampled_log_probs = jnp.mean(log_prob_samples, axis=0)
            q_values = self.critic.apply_fn(
                {"params": self.critic.params}, batch["observations"], batch["actions"]
            ).mean(axis=0)
            temperature = self.temp.apply_fn({"params": self.temp.params})
            advantages = (q_values - temperature * batch["log_probs"]) - (
                values - temperature * sampled_log_probs
            )
            advantages = advantages.reshape(-1)

        index_batches = actor_minibatch_indices(new_rng, advantages.shape[0])
        agent = self

        for indices in index_batches:
            actor_grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(
                agent.actor.params,
                advantages,
                batch,
                indices,
            )
            new_actor = agent.actor.apply_gradients(grads=actor_grads)

            temp_grads, temp_info = jax.grad(temp_loss_fn, has_aux=True)(
                agent.temp.params,
                actor_info["entropy"],
                self.target_entropy,
            )
            new_temp = agent.temp.apply_gradients(grads=temp_grads)

            agent = agent.replace(rng=new_rng, actor=new_actor, temp=new_temp)

        info = {**actor_info, **temp_info}
        return agent, info

    @jax.jit
    def sample_actions(
        self,
        observations: np.ndarray,
        seed: jax.Array,
        temperature: float = 1.0,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample actions and return their log probabilities and pre-actions."""
        distribution = self.actor.apply_fn(
            {"params": self.actor.params}, observations, temperature=temperature
        )
        pre_actions, pre_log_probs = distribution.sample_and_log_prob(seed=seed)

        if self.config.tanh_squash_actions:
            actions = jax.nn.tanh(pre_actions)
            log_probs = tanh_squash_log_prob(pre_log_probs, pre_actions)
        else:
            log_probs = pre_log_probs
            actions = pre_actions

        return actions, log_probs, pre_actions

    @jax.jit
    def deterministic_action(
        self,
        observations: np.ndarray,
    ) -> jnp.ndarray:
        """Return the deterministic action for an observation batch."""
        seed = jax.random.PRNGKey(0)
        distribution = self.actor.apply_fn(
            {"params": self.actor.params}, observations, temperature=0.0
        )
        pre_actions, _ = distribution.sample_and_log_prob(seed=seed)
        if self.config.tanh_squash_actions:
            actions = jax.nn.tanh(pre_actions)
        else:
            actions = pre_actions

        return actions


@dataclass(frozen=True)
class PPOPlusConfig:
    """Learner configuration for PPO+."""

    hidden_dims: int = 256
    use_layer_norm: bool = True
    final_fc_init_scale: float = 1e-2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    momentum: float = 0.9
    b2: float = 0.999
    clip_grad_norm: float = 0.5
    clipping_ratio: float = 0.25
    gae_lambda: float = 0.5
    entropy_coeff: float = 0.5
    temperature: float = 1.0
    seed: int = 42
    discount: float = 0.99
    num_critics: int = 2
    min_target: bool = False
    tanh_squash_actions: bool = True

    @classmethod
    def from_args(cls, args: Any) -> PPOPlusConfig:
        """Build the learner configuration from parsed command-line arguments."""
        return cls(
            hidden_dims=args.hidden_dims,
            use_layer_norm=args.use_layer_norm,
            clipping_ratio=args.clipping_ratio,
            gae_lambda=args.gae_lambda,
            entropy_coeff=args.entropy_coeff,
            temperature=args.temperature,
            seed=args.seed,
            discount=args.gamma,
            num_critics=args.num_critics,
            min_target=args.min_target,
            tanh_squash_actions=args.bound_actions,
        )


def create_learner(
    config: PPOPlusConfig,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
) -> PPOPlusAgent:
    """Create a PPO+ learner from example observations and actions.

    Args:
        config: Resolved learner configuration.
        observations: Example observations for network initialization.
        actions: Example actions for network initialization.
    """
    rng = jax.random.PRNGKey(config.seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    hidden_dims = (config.hidden_dims, config.hidden_dims)

    action_dim = actions.shape[-1]
    actor_def = Policy(
        hidden_dims,
        action_dim=action_dim,
        activations=nn.tanh,
        final_fc_init_scale=config.final_fc_init_scale,
        tanh_squash_distribution=False,
        use_layer_norm=config.use_layer_norm,
    )

    critic_def = ensemblize(OriginalCritic, config.num_critics)(
        hidden_dims=hidden_dims,
        use_layer_norm=config.use_layer_norm,
        activations=nn.tanh,
    )
    critic_params = critic_def.init(critic_key, observations, actions)["params"]
    critic = TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=config.critic_lr),
    )

    actor_params = actor_def.init(actor_key, observations)["params"]
    temp_def = Temperature(config.temperature)
    temp_params = temp_def.init(rng)["params"]

    tx = optax.chain(
        optax.clip_by_global_norm(config.clip_grad_norm),
        optax.adam(
            learning_rate=config.actor_lr,
            b1=config.momentum,
            b2=config.b2,
        ),
    )
    actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=tx)
    temp = TrainState.create(
        apply_fn=temp_def.apply,
        params=temp_params,
        tx=optax.adam(
            learning_rate=config.temp_lr,
            b1=config.momentum,
            b2=config.b2,
        ),
    )

    return PPOPlusAgent(
        rng,
        critic=critic,
        actor=actor,
        temp=temp,
        config=config,
        target_entropy=-config.entropy_coeff * action_dim,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the PPO+ command-line interface."""
    parser = argparse.ArgumentParser()
    parser.set_defaults(algo_name="ppo_plus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="PPO-Plus")
    parser.add_argument("--env_name", type=str, default="Walker2d-v5")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_coeff", type=float, default=0.5)
    parser.add_argument("--num_critics", type=int, default=2)
    parser.add_argument("--hidden_dims", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--on_policy_data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min_target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_layer_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clipping_ratio", type=float, default=0.25)
    parser.add_argument("--gae_lambda", type=float, default=0.5)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--policy_steps", type=int, default=5_000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--bound_actions", action=argparse.BooleanOptionalAction, default=True)
    return parser


@dataclass(frozen=True)
class RunConfig:
    """Environment, storage, and update schedule for one run."""

    total_steps: int
    buffer_size: int
    policy_steps: int
    num_epochs: int
    on_policy_data: bool
    log_interval: int = 20_000

    @classmethod
    def from_args(cls, args: Any) -> RunConfig:
        return cls(
            total_steps=(
                args.max_steps
                if args.max_steps is not None
                else get_max_steps_for_env(args.env_name)
            ),
            buffer_size=args.buffer_size,
            policy_steps=args.policy_steps,
            num_epochs=args.num_epochs,
            on_policy_data=args.on_policy_data,
        )

    @classmethod
    def from_dict(cls, d: Any) -> RunConfig:
        return cls(
            total_steps=(
                d["max_steps"]
                if d["max_steps"] is not None
                else get_max_steps_for_env(d["env_name"])
            ),
            buffer_size=d["buffer_size"],
            policy_steps=d["policy_steps"],
            num_epochs=d["num_epochs"],
            on_policy_data=d["on_policy_data"],
        )

    @property
    def replay_capacity(self) -> int:
        return self.policy_steps if self.on_policy_data else self.buffer_size


@dataclass(frozen=True)
class RunState:
    agent: PPOPlusAgent
    replay_buffer: ReplayBuffer
    actor_buffer: ReplayBuffer
    rollout_rng: jax.Array
    environment_steps: int = 0
    steps_since_evaluation: int = 0


def _create_run_state(
    learner_config: PPOPlusConfig,
    run_config: RunConfig,
    env: Any,
) -> RunState:
    example_transition = {
        "observations": env.observation_space.sample(),
        "actions": env.action_space.sample(),
        "rewards": 0.0,
        "masks": 1.0,
        "truncateds": 0.0,
        "next_observations": env.observation_space.sample(),
        "pre_actions": env.action_space.sample(),
        "log_probs": 0.0,
    }
    replay_buffer = ReplayBuffer.create(example_transition, size=run_config.replay_capacity)
    actor_buffer = ReplayBuffer.create(example_transition, size=run_config.policy_steps)
    agent = create_learner(
        config=learner_config,
        observations=example_transition["observations"][None],
        actions=example_transition["actions"][None],
    )
    return RunState(
        agent=agent,
        replay_buffer=replay_buffer,
        actor_buffer=actor_buffer,
        rollout_rng=jax.random.fold_in(jax.random.PRNGKey(learner_config.seed), 0),
    )


def run_training(
    learner_config: PPOPlusConfig,
    run_config: RunConfig,
    *,
    env: Any,
    eval_env: Any,
    metrics: Any,
) -> RunState:
    """Train PPO+ and return the state at the final update boundary."""
    state = _create_run_state(learner_config, run_config, env)

    with tqdm.tqdm(total=run_config.total_steps) as progress:
        while state.environment_steps < run_config.total_steps:
            next_rollout_rng, rollout_rng = jax.random.split(state.rollout_rng)
            replay_buffer = state.replay_buffer
            if run_config.on_policy_data:
                replay_buffer.reset()

            policy_return, undiscounted_return, num_steps = rollout_policy(
                state.agent,
                env,
                rollout_rng,
                discount=learner_config.discount,
                max_steps=run_config.policy_steps,
                reset_seed=learner_config.seed if state.environment_steps == 0 else None,
                replay_buffer=replay_buffer,
                actor_buffer=state.actor_buffer,
            )
            state = replace(
                state,
                replay_buffer=replay_buffer,
                rollout_rng=next_rollout_rng,
                environment_steps=state.environment_steps + num_steps,
                steps_since_evaluation=state.steps_since_evaluation + num_steps,
            )
            progress.update(num_steps)

            agent = state.agent
            critic_transitions = state.replay_buffer.get_all()
            actor_transitions = state.actor_buffer.get_all()
            for _ in range(run_config.num_epochs):
                agent = agent.update_critics_seq(critic_transitions)
                agent, actor_info = agent.update_actor_seq(actor_transitions)
            state = replace(state, agent=agent)

            train_metrics = {f"training/{key}": value for key, value in actor_info.items()}
            train_metrics["training/undisc_return"] = undiscounted_return
            metrics.log(train_metrics, step=state.environment_steps, commit=False)
            metrics.log(
                {"exploration/disc_return": policy_return},
                step=state.environment_steps,
                commit=False,
            )

            if state.steps_since_evaluation >= run_config.log_interval:
                policy_return, undiscounted_return, _ = rollout_policy(
                    state.agent,
                    eval_env,
                    state.rollout_rng,
                    discount=learner_config.discount,
                    max_rollouts=10,
                    reset_seed=learner_config.seed + 1,
                    eval=True,
                )
                evaluation = {
                    "evaluation/policy_return": policy_return,
                    "evaluation/undisc_policy_return": undiscounted_return,
                }
                print(evaluation)
                metrics.log(evaluation, step=state.environment_steps, commit=True)
                state = replace(state, steps_since_evaluation=0)

    metrics.finish()
    return state


def train(args: Any) -> PPOPlusAgent:
    env, eval_env = create_environments(args.env_name)
    state = run_training(
        PPOPlusConfig.from_args(args),
        RunConfig.from_args(args),
        env=env,
        eval_env=eval_env,
        # metrics=setup_wandb(args.project_name, vars(args)),
    )
    env.close()
    eval_env.close()
    return state.agent




def main(argv: Sequence[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    np.random.seed(args.seed)
    train(args)

    # disregard all this

    # run_c = {
    #     "max_steps": 300_000,
    #     "buffer_size": 50_000,
    #     "policy_steps": 50_000, #?
    #     "num_epochs": 10, #?
    #     "on_policy_data": True,
    #     "log_interval": 300_000_000, # just make it above max_steps, i don't want eval_env being called
    # }

    # import gymnasium  
    # import highway_env

    # run_training(
    #     learner_config=PPOPlusConfig(),
    #     run_config=RunConfig.from_dict(run_c),
    #     env=gymnasium.make('highway-v0'),
    #     eval_env=None,
    #     metrics=None,
    # )

if __name__ == "__main__":
    main()


