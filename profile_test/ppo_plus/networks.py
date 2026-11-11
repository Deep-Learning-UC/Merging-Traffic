"""Flax networks shared by PPO+ and SAC."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def layer_init(std: float = jnp.sqrt(2.0)):
    """Return orthogonal kernel and constant bias initializers."""
    kernel_init = nn.initializers.orthogonal(std)
    bias_init = nn.initializers.zeros_init()
    return kernel_init, bias_init


class MLP(nn.Module):
    """Fully connected network with optional LayerNorm before activations."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray]
    use_layer_norm: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for size in self.hidden_dims:
            kernel_init, bias_init = layer_init()
            x = nn.Dense(size, kernel_init=kernel_init, bias_init=bias_init)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activations(x)
        return x


class OriginalCritic(nn.Module):
    """State-action value network used by PPO+ and SAC."""

    hidden_dims: Sequence[int]
    use_layer_norm: bool
    activations: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        intermediate = MLP(
            self.hidden_dims,
            use_layer_norm=self.use_layer_norm,
            activations=self.activations,
        )(inputs)

        kernel_init, bias_init = layer_init()
        q = nn.Dense(1, kernel_init=kernel_init, bias_init=bias_init)(intermediate)

        return jnp.squeeze(q, -1)


def ensemblize(cls, num_qs):
    """Create an independently initialized ensemble of Flax modules."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=num_qs,
    )


class Policy(nn.Module):
    """Diagonal-Gaussian continuous-action policy."""

    hidden_dims: Sequence[int]
    action_dim: int

    use_layer_norm: bool
    activations: Callable[[jnp.ndarray], jnp.ndarray]
    tanh_squash_distribution: bool
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            use_layer_norm=self.use_layer_norm,
            activations=self.activations,
        )(observations)

        kernel_init, bias_init = layer_init(self.final_fc_init_scale)

        means = nn.Dense(
            self.action_dim, kernel_init=kernel_init, bias_init=bias_init, name="means"
        )(outputs)

        log_stds = nn.Dense(
            self.action_dim, kernel_init=kernel_init, bias_init=bias_init, name="log_stds"
        )(outputs)

        log_stds = jnp.clip(log_stds, -10, 2)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            distribution = distrax.Transformed(
                distribution,
                distrax.Block(distrax.Tanh(), ndims=1),
            )

        return distribution
