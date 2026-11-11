"""Numerical kernels defining the PPO+ updates."""

import jax
import jax.numpy as jnp

Array = jax.Array
Metrics = dict[str, Array]


def critic_minibatch_indices(
    rng: Array,
    num_transitions: int,
    sample_pool_batches: int | None = None,
    batch_size: int = 250,
) -> tuple[Array, int]:
    """Select one buffer-sized set of critic minibatches."""
    n_batches = num_transitions // batch_size
    if n_batches < 100:
        sample_pool_batches = 100

    if sample_pool_batches:
        indices = jax.random.choice(
            rng,
            a=num_transitions,
            shape=(sample_pool_batches, batch_size),
            replace=True,
        )
    else:
        permutation = jax.random.permutation(rng, jnp.arange(num_transitions))
        indices = permutation[: batch_size * n_batches].reshape((n_batches, batch_size))
    return indices, n_batches


def actor_minibatch_indices(
    rng: Array,
    num_transitions: int,
    batch_size: int = 250,
) -> Array:
    """Shuffle complete actor minibatches and discard an incomplete remainder."""
    n_batches = num_transitions // batch_size
    permutation = jax.random.permutation(rng, jnp.arange(num_transitions))
    return permutation[: batch_size * n_batches].reshape((n_batches, batch_size))


def compute_gae(
    rewards: Array,
    values: Array,
    next_values: Array,
    dones: Array,
    truncations: Array,
    gamma: float,
    lam: float,
) -> Array:
    """Compute generalized advantage estimates for one ordered rollout."""
    deltas = rewards + gamma * next_values * (1 - dones) - values
    deltas = deltas * (1 - truncations)

    def update_advantage(cumulative_advantage, delta_and_mask):
        delta, truncation, done = delta_and_mask
        cumulative_advantage = (
            delta + gamma * lam * (1 - done) * (1 - truncation) * cumulative_advantage
        )
        return cumulative_advantage, cumulative_advantage

    _, advantages = jax.lax.scan(
        update_advantage,
        0.0,
        (deltas[::-1], truncations[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    advantages = advantages * (1 - truncations)
    return advantages


def tanh_squash_log_prob(pre_log_probs: Array, pre_actions: Array) -> Array:
    """Apply the stable change-of-variables term for tanh-squashed actions."""
    correction = jnp.sum(
        2 * (jnp.log(2) - pre_actions - jax.nn.softplus(-2 * pre_actions)),
        axis=-1,
    )
    return pre_log_probs - correction


def policy_objective(
    pre_log_probs: Array,
    pre_actions: Array,
    old_log_probs: Array,
    masks: Array,
    advantages: Array,
    clipping_ratio: float,
    *,
    tanh_squash_actions: bool,
) -> tuple[Array, Metrics]:
    """Compute the PPO policy objective and its reported metrics."""
    if tanh_squash_actions:
        new_log_probs = tanh_squash_log_prob(pre_log_probs, pre_actions)
    else:
        new_log_probs = pre_log_probs

    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()

    actor_loss_unclipped = masks * advantages * ratio
    actor_loss_clipped = (
        masks
        * advantages
        * jnp.clip(
            ratio,
            1 - clipping_ratio,
            1 + clipping_ratio,
        )
    )
    actor_loss = -jnp.minimum(actor_loss_unclipped, actor_loss_clipped).mean()

    masked_log_probs = masks * new_log_probs
    entropy = -1 * (masks * masked_log_probs).sum() / masks.sum()
    return actor_loss, {
        "actor_loss": actor_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
    }


def temperature_objective(
    temperature: Array,
    entropy: Array,
    target_entropy: Array,
) -> tuple[Array, Metrics]:
    """Compute the adaptive entropy-temperature objective and metrics."""
    temp_loss = (temperature * (entropy - target_entropy)).mean()
    temp_loss = jax.lax.cond(
        jnp.logical_and(temperature <= 0.001, temp_loss > 0),
        lambda _: 0.0,
        lambda _: temp_loss,
        operand=None,
    )
    return temp_loss, {
        "temp_loss": temp_loss,
        "temperature": temperature,
    }


def critic_target(
    rewards: Array,
    masks: Array,
    next_q: Array,
    next_log_probs: Array,
    temperature: Array,
    discount: float,
    *,
    min_target: bool,
) -> Array:
    """Build the entropy-regularized critic target."""
    target_q = rewards + discount * masks * next_q
    target_q = target_q - discount * masks * next_log_probs * temperature
    target_q = jax.lax.stop_gradient(target_q)

    if min_target:
        target_q = jnp.min(target_q, axis=0)
        target_q = jnp.repeat(target_q.reshape(1, -1), 2, axis=0)
    return target_q


def critic_mse(q: Array, target_q: Array) -> Array:
    """Compute the critic mean-squared error."""
    return ((q - target_q) ** 2).mean()
