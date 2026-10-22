"""
from Claude - model_research_two prompt 2
"""
"""
Minimal DQN from scratch (PyTorch + Gymnasium).

Purpose: show every moving part of a deep RL loop in ~100 lines.
Install:  pip install torch gymnasium numpy
Run:      python dqn_from_scratch.py        (CartPole should reach ~200-500 return)

Big picture of the loop:
  1. Agent looks at the state and picks an action (mostly greedy, sometimes random).
  2. Environment returns reward + next state.
  3. Store the transition in a replay buffer.
  4. Sample a random mini-batch from the buffer and nudge the network so that
        Q(s, a)  ->  r + gamma * max_a' Q_target(s', a')
  5. Every so often, copy the network into a slowly-changing "target" network.
"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- hyperparameters
ENV_ID = "CartPole-v1"
GAMMA = 0.99                 # discount factor: how much future reward matters
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
LEARNING_STARTS = 1_000      # collect random experience before learning
TARGET_SYNC_EVERY = 500      # steps between target-network updates
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 10_000   # exploration schedule
TOTAL_STEPS = 50_000


# ---------------------------------------------------------------- 1. the network
class QNetwork(nn.Module):
    """Maps a state to one Q-value per action."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- 2. replay buffer
class ReplayBuffer:
    """Stores past transitions; random sampling breaks the correlation between
    consecutive steps, which is what makes neural-network training stable."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------- 3. training loop
def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = QNetwork(obs_dim, n_actions)
    q_target = QNetwork(obs_dim, n_actions)
    q_target.load_state_dict(q.state_dict())          # start identical
    optimizer = torch.optim.Adam(q.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    obs, _ = env.reset(seed=0)
    ep_return, recent_returns = 0.0, []

    for step in range(1, TOTAL_STEPS + 1):
        # --- a) choose an action: epsilon-greedy exploration
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY_STEPS)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q(torch.tensor(obs, dtype=torch.float32)).argmax().item()

        # --- b) act in the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # --- c) remember it. Only `terminated` (real end, e.g. crash) cuts the
        #        bootstrap; `truncated` (time limit) does not.
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_return += reward
        if terminated or truncated:
            recent_returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()

        # --- d) learn from a random mini-batch
        if len(buffer) >= LEARNING_STARTS:
            s, a, r, s2, d = buffer.sample(BATCH_SIZE)

            q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)       # Q(s, a) we predicted
            with torch.no_grad():                                   # the "label" (TD target)
                target = r + GAMMA * (1.0 - d) * q_target(s2).max(dim=1).values

            loss = nn.functional.smooth_l1_loss(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- e) periodically refresh the target network
        if step % TARGET_SYNC_EVERY == 0:
            q_target.load_state_dict(q.state_dict())

        if step % 5_000 == 0 and recent_returns:
            print(f"step {step:6d} | eps {eps:.2f} | "
                  f"mean return (last 20 eps) {np.mean(recent_returns[-20:]):.1f}")

    env.close()


if __name__ == "__main__":
    main()
