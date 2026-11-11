REPO LINK: https://github.com/kallel-mahdi/PPO-Plus

# PPO+: Enhancing proximal policy optimization

This repository contains the research implementation accompanying **“PPO+:
Enhancing proximal policy optimization,”** accepted at the Reinforcement
Learning Conference (RLC) 2026 for publication in the Reinforcement Learning
Journal (RLJ).

PPO+ keeps the actor update on-policy while training a Q-function critic from a
replay buffer. The implementation also uses bounded continuous actions and an
entropy regularization term. Experiments in the paper cover six MuJoCo-v5 tasks
and the `dog-{stand,walk,trot,run}` tasks from the DeepMind Control Suite.

## Installation

Python 3.10–3.12 is supported by the package metadata. A fresh environment is
strongly recommended.

### CPU / portable installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[paper]'
```

The base installation deliberately installs the CPU-safe JAX package. MuJoCo
experiments additionally require the platform libraries needed by MuJoCo and
OpenGL. DeepMind Control Suite experiments may require EGL or OSMesa on a
headless machine.

### NVIDIA GPU installation

Install the project first, then install JAX using the command appropriate for
the machine and CUDA driver from the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The convenience extra for the CUDA 12 wheel is:

```bash
python -m pip install -e '.[paper,cuda12]'
```

PPO and TRPO use PyTorch. To install those baselines without the complete
paper plotting and DM-Control stack, use:

```bash
python -m pip install -e '.[baselines]'
```

For GPU work, using separate environments for the JAX PPO+ implementation and
the PyTorch PPO/TRPO baselines is recommended. Their binary wheels can request
different CUDA/cuDNN combinations; do not assume that a single unconstrained
environment is reproducible merely because dependency resolution succeeds.

Plotting tools are optional:

```bash
python -m pip install -e '.[plots]'
```

Supply each figure's W&B source and output path explicitly, for example:

```bash
python -m plots.performance benchmark \
  --project-path your-entity/your-project \
  --output plots/benchmarks/benchmark
```

The remaining paper figures are exposed by `plots.combined`, `plots.diagnostics`,
and `plots.distributions`; each module documents its required W&B project
arguments through `--help`.

## Running experiments

The launchers are repository-level scripts. They currently initialize Weights
& Biases logging, so authenticate outside the repository:

```bash
wandb login
```

or provide `WANDB_API_KEY` through the process environment or the cluster's
secret manager. Never place an API key in source code, notebooks, job scripts,
or committed `.env` files.

For local smoke tests without network logging:

```bash
WANDB_MODE=disabled python run.py ppo_plus --env_name Hopper-v5 --max_steps 5000
```

A single PPO+ MuJoCo run can be launched with:

```bash
python run.py ppo_plus \
  --env_name Hopper-v5 \
  --seed 0 \
  --max_steps 1000000 \
  --project_name PPO-Plus
```

A DeepMind Control Suite task uses the `domain-task` form:

```bash
python run.py ppo_plus \
  --env_name dog-walk \
  --seed 0 \
  --max_steps 5000000 \
  --project_name PPO-Plus
```

The comparison launchers are:

```bash
python run.py ppo --env_name Hopper-v5 --seed 0 --max_steps 1000000
python run.py sac --env_name Hopper-v5 --seed 0 --max_steps 1000000
python run.py trpo --env_name Hopper-v5 --seed 0 --max_steps 1000000
```

Use `python run.py sweep --help` to configure multi-seed local runs.

The paper reports ten seeds for the principal benchmark curves and five seeds
for the replay-buffer ablation. A single run is only a smoke test, not a paper
reproduction.

## Paper configuration

The accepted manuscript and its original experiment runner together define the
following PPO+ configuration:

| Parameter | Value | Source |
| --- | --- | --- |
| Optimizer / learning rate | Adam / `3e-4` | Manuscript Table 3 |
| Discount | `0.99` | Manuscript Table 3 |
| Critic replay capacity | `50,000` | Manuscript Table 3 |
| Critics | 2, trained independently without a minimum target | Manuscript Table 3 and implementation |
| Networks | 2 × 256, tanh, LayerNorm | Manuscript Table 3 |
| Minibatch size | 250 | Manuscript Table 3 |
| Actor update interval | 5,000 environment steps | Manuscript Table 3 |
| Update epochs | 10 | Original experiment runner |
| GAE λ | `0.5` | Manuscript Table 3 |

The scripts contain default values and ablation switches. Do not treat an
unreviewed default command as a substitute for the accepted protocol. Preserve
the resolved command, commit, package versions, seeds, and W&B run identifiers
when producing new results.

## Repository map

- `ppo_plus/`: PPO+, SAC, environments, and diagnostics.
- `paper_baselines/`: PPO and TRPO baselines.
- `plots/`: paper figure scripts.
- `run.py`: experiment launcher.

## Citation

```bibtex
@inproceedings{kallel2026ppoplus,
  title     = {PPO+: Enhancing proximal policy optimization},
  author    = {Kallel, Mahdi and Holgado-Alvarez, Jose-Luis and
               Tosatto, Samuele and D'Eramo, Carlo},
  booktitle = {Reinforcement Learning Conference},
  year      = {2026}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License and attribution

The PPO+ release is available under the [MIT License](LICENSE). Parts of the
code build on JAXRL-M, CleanRL, and dmc2gym; their notices are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
