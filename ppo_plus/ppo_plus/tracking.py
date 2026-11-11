"""Weights & Biases setup shared by the experiment runners."""

import os

import wandb


def setup_wandb(project, hyperparameters):
    """Initialize and return a W&B run for one experiment."""
    return wandb.init(
        config=hyperparameters,
        project=project,
        mode=os.environ.get("WANDB_MODE", "online"),
        save_code=True,
    )
