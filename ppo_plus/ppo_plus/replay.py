"""Replay storage shared by PPO+ and SAC."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict


class ReplayBuffer(FrozenDict):
    """Fixed-capacity transition dataset with a circular write cursor."""

    @classmethod
    def create(cls, transition: Any, size: int) -> ReplayBuffer:
        """Allocate a buffer from one example transition."""

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree.map(create_buffer, transition)
        return cls(buffer_dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = len(jax.tree.leaves(self._dict)[0])
        self.size = 0
        self.pointer = 0

    def sample(self, batch_size: int) -> Any:
        """Return a uniformly sampled batch."""
        indices = np.random.randint(self.size, size=batch_size)
        return jax.tree.map(lambda array: array[indices], self._dict)

    def add_transition(self, transition: Any) -> None:
        """Write one transition at the current cursor."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree.map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_all(self) -> Any:
        """Return all transitions exposed by the current cursor state."""
        batch = jax.tree.map(lambda x: x[: self.size], self._dict)
        return jax.tree.map(lambda x: jax.device_put(x), batch)

    def reset(self) -> ReplayBuffer:
        """Reset the exposed size and write cursor."""
        self.size = 0
        self.pointer = 0
        return self
