I wanted to move these out of the README, but I didn't want to get rid of them:
  
`python -m venv .venv` -> create virtual environment named `.venv`  
`. ./.venv/Scripts/activate` -> activate virtual environment  
`pip freeze > requirements.txt` -> output dependencies into static `requirements.txt` file  
`pip install -r requirements.txt` -> install requirements  

--- 


**Goal:** Find two deep reinforcement learning models that can integrate with highway env, and how to integrate them with highway env

first let's test with a non DRL model

# Model List
- A2C (x2)
- ARS
- CrossQ
- DDPG (x2)
- DQN (x3)
- DroQ
- HER (x2)
- Maskable PPO
- PPO
- QR-DQN
- RecurrentPPO
- SAC
- SAC-N
- SimBa
- TD3
- TQC
- TRPO (x2)
- VI
- CEM
- MCTS
- RVI
- DROP
- IRP
- FTQ
- BFTQ
- ACER
- ACKTR
- GAIL
- PPO1 (obsolete)
- PPO2


# Models in Stable Baseline3
https://github.com/DLR-RM/stable-baselines3/blob/master/README.md?plain=1

YUP = mentioned by Claude

- A2C - YUP
- ARS
- CrossQ
- DDPG - YUP
- DQN - YUP
- DroQ
- HER   
- Maskable PPO
- PPO - YUP
- QR-DQN
- RecurrentPPO
- SAC - YUP
- SAC-N
- SimBa
- TD3 - YUP
- TQC
- TRPO - YUP


# models in eluerent/rl-agents
https://github.com/eleurent/rl-agents/tree/master
 - VI
 - CEM
 - MCTS
 - RVI
 - DROP
 - IRP
 - DQN
 - FTQ
 - BFTQ




 # models in openai/baselines
 https://github.com/openai/baselines
  - A2C
  - ACER
  - ACKTR
  - DDPG
  - DQN
  - GAIL
  - HER
  - PPO1 (obsolete)
  - PPO2
  - TRPO


# Prompts

## Prompt 1 
Claude Sonnet 5.5 Medium Effort  
hi, can you give me a crash course in deep reinforcement learning? with a focus on the types of models and examples of each?

## Response 1

### Deep RL Crash Course

#### The core idea

An **agent** interacts with an **environment**. At each step it sees a state, takes an action, and receives a reward. The goal is to learn a **policy** (state → action) that maximizes cumulative reward. "Deep" means neural networks approximate the policy, the value function, or a model of the world.

Key vocabulary:
- **Value function V(s) / Q(s,a):** expected future reward from a state, or from taking an action in a state.
- **On-policy vs. off-policy:** whether you learn only from data produced by the current policy, or can reuse older data (replay buffers).
- **Exploration vs. exploitation:** trying new things versus using what already works.
- **Model-free vs. model-based:** whether the agent learns a model of the environment's dynamics.

#### 1. Value-based methods

Learn Q(s,a) with a network, then act greedily. Best for **discrete action spaces**.

- **DQN (2013/2015):** Q-learning with a CNN, a replay buffer, and a target network. Learned Atari games from raw pixels.
- **Double DQN, Dueling DQN, Prioritized Replay:** fixes for overestimation, better architecture, and smarter sampling.
- **Rainbow:** combines six of these improvements. It is still a strong Atari baseline.

#### 2. Policy gradient methods

Directly optimize the policy by nudging probabilities toward actions that led to high reward.

- **REINFORCE:** the simplest version, with high variance.
- **TRPO / PPO:** constrain how far each update can move the policy, which keeps training stable. **PPO** is the workhorse of modern RL. OpenAI Five (Dota 2) used it, as did many robotics and RLHF pipelines.

#### 3. Actor-critic methods

Combine both: an *actor* (policy) and a *critic* (value function that reduces variance).

- **A2C / A3C:** synchronous and asynchronous advantage actor-critic.
- **DDPG:** off-policy actor-critic for **continuous actions** (e.g., robot joint torques).
- **TD3:** DDPG with fixes for overestimation and instability.
- **SAC (Soft Actor-Critic):** adds an entropy bonus to encourage exploration. It is sample-efficient, robust, and a default choice for continuous control and real-robot learning.

#### 4. Model-based methods

Learn (or are given) a model of the environment and use it to plan or generate imagined experience. They are more sample-efficient but harder to get right.

- **AlphaGo / AlphaZero:** Monte Carlo Tree Search guided by neural networks, with the rules of the game given. They mastered Go, chess, and shogi.
- **MuZero:** learns its own latent model, with no rules provided, and still matches AlphaZero.
- **World Models / Dreamer (V1–V3):** learn a latent world model and train the policy entirely inside "imagination." DreamerV3 collected diamonds in Minecraft without human data.
- **MBPO, PETS:** learned dynamics models for continuous control.

#### 5. Offline (batch) RL

Learn from a fixed dataset with no environment interaction. This is essential when exploring is costly or dangerous (healthcare, robotics, recommendation systems).

- **CQL (Conservative Q-Learning):** penalizes overestimated Q-values for actions not seen in the data.
- **IQL, BCQ, TD3+BC:** other ways to avoid the out-of-distribution action problem.
- **Decision Transformer:** treats RL as sequence modeling. You condition a transformer on a desired return and it generates actions.

#### 6. Imitation and inverse RL

Learn from demonstrations instead of (or alongside) a reward signal.

- **Behavior cloning:** supervised learning on expert state-action pairs. Simple, but errors compound.
- **DAgger:** queries an expert to fix the states the learner actually visits.
- **GAIL:** a GAN-style setup where a discriminator tells expert behavior from the agent's.
- **Inverse RL:** infers the reward function from demonstrations.

#### 7. RL for language models

This is now one of the most important applications.

- **RLHF:** train a reward model on human preferences, then optimize the LLM with PPO. This is how ChatGPT-style assistants were aligned.
- **DPO:** skips the explicit reward model and optimizes directly on preference pairs.
- **GRPO and RLVR (RL with verifiable rewards):** use rewards that can be checked automatically, such as math answers or passing unit tests. GRPO drops the critic and compares samples within a group. This approach underlies much of today's reasoning-model training (e.g., DeepSeek-R1).

#### 8. Other important families

- **Exploration methods:** curiosity (ICM) and Random Network Distillation help in sparse-reward games like Montezuma's Revenge. Go-Explore built on this idea.
- **Multi-agent RL:** MADDPG, QMIX, and self-play. AlphaStar (StarCraft II) and OpenAI Five are famous examples.
- **Hierarchical RL:** options and feudal networks that learn high-level goals plus low-level skills.
- **Distributional RL:** C51 and QR-DQN learn the full distribution of returns, not just the mean.

#### Which should you use?

| Situation | Good starting point |
|---|---|
| Discrete actions, lots of cheap simulation | DQN/Rainbow or PPO |
| Continuous control, want sample efficiency | SAC or TD3 |
| Need a robust general-purpose default | PPO |
| Simulation is expensive, data is scarce | Model-based (Dreamer) |
| Only have logged data | Offline RL (CQL, IQL) |
| Have expert demos | Behavior cloning, then fine-tune with RL |
| Aligning or training an LLM | DPO, GRPO, or PPO |

#### Learning path

1. Learn the basics from Sutton & Barto's *Reinforcement Learning: An Introduction* (free online).
2. Implement DQN on CartPole, then PPO on a continuous task, using Gymnasium.
3. Use clean reference code from **CleanRL** or **Stable-Baselines3**.
4. Watch OpenAI's *Spinning Up in Deep RL* material, which is excellent for the theory-to-code bridge.
5. Be ready for practical pain: RL is sensitive to hyperparameters, seeds, and reward design.

If you tell me your background and goals (research, games, robotics, LLMs), I can go deeper on a specific branch or sketch out a hands-on project.


## Summary
- **DQN (2013/2015):** Q-learning with a CNN, a replay buffer, and a target network. Learned Atari games from raw pixels.
- **Double DQN, Dueling DQN, Prioritized Replay:** fixes for overestimation, better architecture, and smarter sampling.
- **Rainbow:** combines six of these improvements. It is still a strong Atari baseline.
- **REINFORCE:** the simplest version, with high variance.
- **TRPO / PPO:** constrain how far each update can move the policy, which keeps training stable. **PPO** is the workhorse of modern RL. OpenAI Five (Dota 2) used it, as did many robotics and RLHF pipelines.
- **A2C / A3C:** synchronous and asynchronous advantage actor-critic.
- **DDPG:** off-policy actor-critic for **continuous actions** (e.g., robot joint torques).
- **TD3:** DDPG with fixes for overestimation and instability.
- **SAC (Soft Actor-Critic):** adds an entropy bonus to encourage exploration. It is sample-efficient, robust, and a default choice for continuous control and real-robot learning.
- **AlphaGo / AlphaZero:** Monte Carlo Tree Search guided by neural networks, with the rules of the game given. They mastered Go, chess, and shogi.
- **MuZero:** learns its own latent model, with no rules provided, and still matches AlphaZero.
- **World Models / Dreamer (V1–V3):** learn a latent world model and train the policy entirely inside "imagination." DreamerV3 collected diamonds in Minecraft without human data.
- **MBPO, PETS:** learned dynamics models for continuous control.
- **CQL (Conservative Q-Learning):** penalizes overestimated Q-values for actions not seen in the data.
- **IQL, BCQ, TD3+BC:** other ways to avoid the out-of-distribution action problem.
- **Decision Transformer:** treats RL as sequence modeling. You condition a transformer on a desired return and it generates actions.
- **Behavior cloning:** supervised learning on expert state-action pairs. Simple, but errors compound.
- **DAgger:** queries an expert to fix the states the learner actually visits.
- **GAIL:** a GAN-style setup where a discriminator tells expert behavior from the agent's.
- **Inverse RL:** infers the reward function from demonstrations.
- **RLHF:** train a reward model on human preferences, then optimize the LLM with PPO. This is how ChatGPT-style assistants were aligned.
- **DPO:** skips the explicit reward model and optimizes directly on preference pairs.
- **GRPO and RLVR (RL with verifiable rewards):** use rewards that can be checked automatically, such as math answers or passing unit tests. GRPO drops the critic and compares samples within a group. This approach underlies much of today's reasoning-model training (e.g., DeepSeek-R1).
- **Exploration methods:** curiosity (ICM) and Random Network Distillation help in sparse-reward games like Montezuma's Revenge. Go-Explore built on this idea.
- **Multi-agent RL:** MADDPG, QMIX, and self-play. AlphaStar (StarCraft II) and OpenAI Five are famous examples.
- **Hierarchical RL:** options and feudal networks that learn high-level goals plus low-level skills.
- **Distributional RL:** C51 and QR-DQN learn the full distribution of returns, not just the mean.
