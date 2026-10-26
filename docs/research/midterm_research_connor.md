on 10/20/22 after the meeting, she popped the midterm report that introduced requirements that have not been mentioned since the first meeting

 - [x] analyze requirements of the midterm report
 - [x] construct new TDL to begin research for that report
 - [ ] research 
   - [ ] read Jace's papers
   - [ ] find recent paper on DRL
   - [ ] analyze model - name, authors, key techniques, performance, strengths/weaknesses, how to implement in python if it doesn't already exist
   - [ ] repeat 5 more times
 - [ ] bullet background information from the prior research
 - [ ] how do you implement LaTeX?
 - [ ] extend TDL to begin to finalize report



new TDL 
- [ ] acquire baseline understand of DRL
  - [x] what are the metrics - Prompt 8
  - [ ] understand the core models in the various family of functions
    - [x] DQN - Q learning - Prompt 7
    - [x] PPO - policy gradient - Prompt 12
    - [ ] SAC - off-policy actor-critic
- [ ] research
  - [ ] Starting points: 
    - [ ] PPO+, 
    - [ ] ExO-PPO, 
    - [ ] Off-policy Distributional Q(lambda), 
    - [ ] Reversal Q-learning, and 
    - [ ] SimBa.
    - [ ] Do I need more? 
    - NOTE: But first check if they do not contain public repos or have an implementation somewhere. We cannot implement them if they do not contain public repos. The five models we need to consider should have implementations somewhere.
      - NOTE: possible implementations include: Stable-Baselines3, SB3-Contrib, and SBX. But recent models are likely not be included in libraries.
- [ ] finish assignment



# midterm report
  - one submission per group
  - due 11/3/26
  - cited - any citation but IEEE is preferred
  - background
    - overview of problem / challenge problem addresses
    - motivation behind topic selection
    - relevance to field
    - specific issues that project wants to fix
    - definitions
  - related studies
    - 5 state-of-art (2025-2026) models
    - model name, author, paper cited, key technique/algorithms, performance on standard metrics, notable strengths/weaknesses
    - how do we build or deviate from these studies
  - LaTeX or Word
  - in class she said that the project should either address a small research problem or benchmark many models (probably preferred option, but let's research)


# Research

## 1 Deep Reinforcement Learning for Autonomous Driving
 - from Jace
 - https://arxiv.org/abs/1811.11329, and https://arxiv.org/pdf/1811.11329


## 2 LMDrive: Closed-Loop End-to-End Driving with Large Language Models
 - from Jace
 - https://arxiv.org/abs/2312.07488

## 3 Fully charmed tetraquarks from LHC to FCC: Natural stability from fragmentation
 - from Jace
 - https://arxiv.org/abs/2405.14773

## 4 GAIA-1: A Generative World Model for Autonomous Driving
 - from Jace
 - https://arxiv.org/abs/2309.17080 



# AI prompting

claude sonnet 5.5 medium effort

## Prompt 1 Initial Problem

Context: 
Hello, our team project for our Deep Learning university class has encountered a problem. 

The project is open-ended, and originally our project was to utilize Deep Reinforcement Learning models on the highway-env python library (https://highway-env.farama.org/) on a specific scenario (merging onto a highway) and determing which model could merge onto a highway and take an exit in the fastest time without causing any accients. Admittedly, I'm not sure if our problem that we're trying to solve was clear from the start. 

As we were beginning to research more heavily into which models we wanted to specifically compare, our instructor provided clearer expectations from the project, which may make our initial project invalid. We are intended to make a contribution to an existing small research gap OR to benchmark existing models. We are also to build off of models that were published in 2026, potentially 2025. I'm not familiar with many of the models that are compatible with highway-env, but I understood that they seemed to be 'classic' DRL models. The models that we found that were explicitly compatible were (from a variety of repos): A2C, ARS, CrossQ, DDPG, DQN, DroQ, HER, Maskable PPO, PPO, QR-DQN, RecurrentPPO, SAC, SAC-N, SimBa, TD3, TQC, TRPO, VI, CEM, MCTS, RVI, DROP, IRP, FTQ, BFTQ, ACER, ACKTR, GAIL, PPO1 (obsolete?), and PPO2. 

Before we completely change ideas, we might try to continue to explore in the same field and modify what we already have to fit the project requirements, so as not to nullify the work we've already done. 

Questions:
1) What are some recently-developed DRL models, and the papers that published them?
2) Are these previous models compatible with our original scenario, or will we have to adjust the problem that we're trying to solve?
3) If I wish to find more recent papers on DRL, where would I go to search? Are there specific journals? A field survey or a review paper?  Keywords beyond 'deep reinforcement learning' for our university database?
4) Do you have any general advice as we continue with our project?


## Response 1 Initial Problem

I'll check what's been published recently so the paper list is current.

The first two answers come from papers I found while searching and from my own recall. Check the details yourself before citing anything.

### 1) Recent DRL models and papers

Your list is mostly 2015-2023 algorithms, so your instructor is likely asking you to go beyond it. Here are recent ones, grouped by how easily they'd fit your setup.

**2026 papers I confirmed**
- **PPO+** (Kallel, Holgado-Alvarez, Tosatto, D'Eramo), *Reinforcement Learning Journal* vol. 7, presented at RLC 2026. It is a principled PPO modification that improves asymptotic performance and narrows the gap with off-policy methods on continuous control. It's a natural "PPO vs. PPO+" comparison.
- **ExO-PPO** (Wang & Yang), arXiv 2602.09726, Feb 2026. It is a PPO variant that adds off-policy data reuse while keeping PPO-style stability.
- **Off-policy Distributional Q(λ)** (Tang, Rowland, Munos, Pires, Dabney), AISTATS 2026. It combines Q(λ) with the C51 distributional agent. It's relevant to your discrete-action setup and to the QR-DQN-style models you already found.
- **Reversal Q-Learning** (arXiv 2606.17551, June 2026). It trains flow-based policies off-policy. It's cutting-edge but probably too heavy to reimplement.

**2024-2025 efficiency-focused methods (from my memory, verify before citing)**
- **SimBa** (Lee et al., ICLR 2025), a simple architecture that scales network parameters. It's already on your list.
- **SimBa-V2** and **BRO** (Nauman et al., NeurIPS 2024), which are scaled-up network designs for off-policy RL.
- **PQN** (Gallici et al., 2024-25), a parallelized Q-learning method without replay buffers.
- **FastTD3** and related 2025 work on fast, parallel off-policy training.
- **CrossQ** (Bhatt et al., ICLR 2024) and **DroQ**, which are sample-efficiency methods already on your list.

**Highway merging papers from 2026**
- *Partial Attention in Deep RL for Safe Multi-Agent Control* (arXiv 2603.21810, March 2026) uses QMIX plus attention on a highway merging task.
- *Make Sufficient Probing* (Wang et al., 2026) is an RL motion planner for dense merging.
- *A Systematic Study of Multi-Agent DRL for Safe and Robust Autonomous Highway Ramp Entry* (arXiv 2411.14593) is a merging benchmark-style paper.

### 2) Compatibility with your scenario

Your scenario stays valid. Every algorithm above is general-purpose, and highway-env is a standard Gymnasium environment. The adjustments are mostly about framing:

- **Action space.** Highway-env supports both discrete (`DiscreteMetaAction`: lane change, faster, slower) and continuous (`ContinuousAction`: steering and throttle). DQN-family methods (QR-DQN, PQN, distributional Q(λ)) need the discrete version. SAC, TD3, CrossQ, SimBa and PPO+ lean toward continuous.
- **Your scenario.** The built-in `merge-v0` environment is different from "merge, then take an exit." You have probably made a custom environment, or you could combine `merge-v0` and `exit-v0`. This is a good place to claim a contribution: you can document your custom scenario and reward design as a reusable benchmark.
- **Metrics.** "Fastest time without any accidents" is two objectives (speed and safety), so define it precisely. You could use success rate, collision rate, time-to-exit and episode return, with multiple seeds. Safe RL methods (constrained RL, shielding) also fit this framing.
- **Implementation.** Newer methods may not ship in Stable-Baselines3. SB3-Contrib and SBX (the JAX version) cover CrossQ, TQC, DroQ and others. For 2025-26 papers you may have to use the authors' code or reimplement, which is a risk for your timeline.

### 3) Where to search

**Venues**
- **Conferences:** NeurIPS, ICML, ICLR, AAAI, AISTATS, and the **Reinforcement Learning Conference (RLC)**, whose papers appear in the *Reinforcement Learning Journal*.
- **Journals:** JMLR, TMLR, IEEE TNNLS and Artificial Intelligence.
- **Domain venues for driving:** IEEE Transactions on Intelligent Transportation Systems, IEEE Transactions on Intelligent Vehicles, Transportation Research Part C, IEEE ITSC and IEEE IV.
- **Preprints:** arXiv cs.LG, cs.RO and cs.AI. Semantic Scholar, Google Scholar and Connected Papers help with citation tracing.

**Surveys**
- Search "survey" together with "reinforcement learning autonomous driving decision making," and read their "open problems" sections. That is where research gaps are stated explicitly.
- Search "reproducibility" and "benchmark" in RL (Henderson et al.'s "Deep RL That Matters" is the classic). Benchmark-style contributions are what your instructor said is acceptable.

**Keywords**
- Algorithms: off-policy, sample efficiency, distributional RL, scaling networks, UTD ratio, plasticity loss, flow/diffusion policy, offline-to-online.
- Safety: safe RL, constrained MDP, shielding, risk-sensitive, CVaR.
- Driving: on-ramp merging, cooperative merging, interaction-aware, ramp entry, decision-making, motion planning.
- Research-gap terms: benchmark, comparative study, reproducibility, sim-to-real gap, generalization across traffic density.

### 4) Advice

1. **Reframe rather than restart.** A benchmark fits your work well: "Do recent (2025-26) sample-efficient DRL methods outperform classic baselines on a merge-and-exit task, in both safety and speed?" Your existing environment and baselines (PPO, DQN, SAC) become the control group.
2. **Pick a small number of models, around 4-6.** For example, one classic per family (PPO, DQN, SAC) plus two or three recent ones (PPO+, CrossQ or SimBa, and a distributional method). Depth beats breadth.
3. **Ask the instructor to confirm "2026" before you commit.** For example, does a method published at ICLR 2025 count? Does "build off models published in 2026" mean the algorithm itself, or simply a recent paper you extend?
4. **Report results over multiple seeds** (5 or more) with confidence intervals, and keep reward shaping identical across models. Poor seed discipline is the most common way student RL benchmarks lose credibility.
5. **Check compute and code availability** for each candidate before choosing it. A method with no public repo can cost you weeks.
6. **Write up the reward design.** The balance between speed and collision penalties will heavily shape which model "wins," so an ablation on it can be a contribution in itself.

If you tell me your timeline, team size and whether your merge-and-exit environment is custom, I can help you narrow this to a concrete model list and a research question.

## Summary 1 Initial Problem and follow-up questions
 - Current models to investigate
   - PPO+
   - ExO-PPo
   - Off-policy Distributional Q(lambda)
   - Reversal Q-Learning
   - SimBa
   - FastTD3
 - Also look at Highway merging papers
 - I need a crash course on how existing DRL works. It's history, what are its most popular models, how do they work, what does implementation look like. There's a lot that I don't understand - before researching
 - Need to refine our metrics
 - Authors can have code that we can reimplement for highway-env
 - Resources:
   - Reinforcement Learning Journal
 - Before we solidify whether we want to make a contribution or benchmark, we should conduct our research first
 - I should try to find a survey on DRL highway merging see if I can find outstanding questions
 - When analyzing models - do they have a public repo?
 - Are there frameworks for building newer models? I've used TensorFlow to build non-DRL models before (e.g. VGG) - is there a similar framework for DRL? 

 - Things to tell Claude in follow-ups context:
   - We are assigned to research 5 state-of-the-art models in a week - if we choose to benchmark we will benchmark those 5 models
   - She explicitly set 2026 models, 2025 at the earliest (framed as a last-resort)
   - The environment is custom
   - Timeline is a final demo 6 weeks from today, in the week of 11/30-12/4 at the earliest. This time is not cemented. We will need a poster. The project started unofficially in August - it took several weeks for each person to find a group. 
   - There are 5 people on our team 

 - Next steps:
   - Crash course / history
   - is there a DRL framework
   - Research those models / highway merging papers


## Prompt 2 Additional context and foundational DRL

Additional Context:
  - We are assigned to research 5 state-of-the-art models in a week - if we choose to benchmark we will benchmark those 5 models. This is the midterm progress report. 
  - She explicitly said 2026 models, 2025 at the earliest, perhaps as a last-resort
  - Yes, the environment will be custom
  - Timeline is a final demo 6 weeks from today, in the week of 11/30-12/4 at the earliest. This time is not cemented - even the instructor isn't sure. We will need a poster. The project started unofficially in August - it took several weeks for each person to find a group. We thought we should have it finished / presentable a week or two before that week.
  - There are 5 people on our team.

On a separate topic I am mostly unfamiliar with DRL, and also with RL besides a quick introduction in an AI class a year ago.

Questions: 
1) Can you give a quick history of DRL?
2) What are the most popular models used in DRL? How do they work? What year were they published? What are the names of the papers they were published in? 
3) How does a DRL implementation generally work, at a high-level? A code example would be greatly appreciated.  


## Response 2 Additional context and foundational DRL

I'll write two code files you can read through. They're untested, so expect small version fixes.

### The core idea

An **agent** observes a **state**, picks an **action**, and receives a **reward** and a new state from the **environment**. It repeats this over many episodes. The agent's goal is to learn a **policy** (state → action) that maximizes the expected sum of discounted future rewards, called the **return**. Two functions appear everywhere:

- **V(s)** is how good it is to be in state s.
- **Q(s, a)** is how good it is to take action a in state s and then follow the policy.

"Deep" RL means a neural network approximates these functions or the policy. That's necessary because a driving scene has too many possible states to store in a table.

### 1) Short history

- **1950s-60s:** Bellman's dynamic programming gives the math (the Bellman equation).
- **1988-92:** Sutton's temporal-difference learning, Watkins' Q-learning (1989) and Williams' REINFORCE (1992) arrive. The last is the first policy-gradient method. TD-Gammon (Tesauro, 1992-95) plays backgammon with a neural network but doesn't transfer to other games.
- **1990s-2000s:** RL is mostly tabular or uses linear function approximation. Combining it with neural networks is known to be unstable.
- **2013-15: the DQN breakthrough.** DeepMind's DQN plays Atari from raw pixels. Replay buffers and target networks are what made it stable.
- **2015-17: the policy-gradient wave.** TRPO, DDPG, A3C and then PPO appear. AlphaGo beats Lee Sedol in 2016. AlphaZero follows in 2017.
- **2018-19:** TD3 and SAC make continuous control robust. Rainbow combines DQN improvements. OpenAI Five and AlphaStar beat human pros at Dota 2 and StarCraft II.
- **2019-23:** Model-based and world-model methods (MuZero, Dreamer) mature. RL with human feedback (RLHF) is used to train language models like InstructGPT.
- **2023-26:** The research focus moves to sample efficiency, scaling network sizes, offline RL, flow and diffusion policies, safe RL, and RL for reasoning models.

### 2) The most popular models

**Value-based (learn Q, act greedily; discrete actions)**

| Model | Year | Paper | Idea |
|---|---|---|---|
| DQN | 2013 / 2015 | *Playing Atari with Deep Reinforcement Learning* (Mnih et al., arXiv); *Human-level control through deep reinforcement learning* (Nature) | A network predicts Q for each action. A replay buffer and a slow target network keep training stable. |
| Double DQN | 2015-16 | *Deep Reinforcement Learning with Double Q-learning* (van Hasselt et al.) | Reduces Q-value overestimation. |
| Dueling DQN | 2016 | *Dueling Network Architectures for Deep RL* (Wang et al.) | Separates state value from action advantage. |
| Prioritized replay | 2016 | *Prioritized Experience Replay* (Schaul et al.) | Replays surprising transitions more often. |
| C51 / QR-DQN | 2017 / 2018 | *A Distributional Perspective on RL* (Bellemare et al.); *Distributional RL with Quantile Regression* (Dabney et al.) | Learns the whole distribution of returns, not just the mean. |
| Rainbow | 2017-18 | *Rainbow: Combining Improvements in Deep RL* (Hessel et al.) | Bundles six DQN improvements. |

**Policy gradient (learn the policy directly)**

| Model | Year | Paper | Idea |
|---|---|---|---|
| REINFORCE | 1992 | *Simple Statistical Gradient-Following Algorithms for Connectionist RL* (Williams) | Increases the probability of actions that led to high return. |
| TRPO | 2015 | *Trust Region Policy Optimization* (Schulman et al.) | Limits how far each update can move the policy. |
| A3C / A2C | 2016 | *Asynchronous Methods for Deep RL* (Mnih et al.) | An actor-critic with parallel workers. A2C is the synchronous version. |
| **PPO** | 2017 | *Proximal Policy Optimization Algorithms* (Schulman et al.) | Clips the update ratio as a simple trust region. It is the most widely used default. |

**Off-policy actor-critic (continuous actions, sample-efficient)**

| Model | Year | Paper | Idea |
|---|---|---|---|
| DDPG | 2015-16 | *Continuous Control with Deep RL* (Lillicrap et al.) | A deterministic actor plus a Q-critic. It is brittle. |
| TD3 | 2018 | *Addressing Function Approximation Error in Actor-Critic Methods* (Fujimoto et al.) | Fixes DDPG with twin critics, delayed actor updates and target smoothing. |
| **SAC** | 2018 | *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor* (Haarnoja et al.) | Adds an entropy bonus so the agent keeps exploring. It is robust and widely used. |
| TQC | 2020 | *Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics* (Kuznetsov et al.) | SAC with distributional critics. |
| CrossQ | 2024 | *CrossQ: Batch Normalization in Deep RL for Greater Sample Efficiency and Simplicity* (Bhatt et al.) | A simplified, very sample-efficient SAC variant. |

**Planning and model-based**

| Model | Year | Paper | Idea |
|---|---|---|---|
| AlphaZero | 2017 | *Mastering Chess and Shogi by Self-Play with a General RL Algorithm* (Silver et al.) | Tree search guided by a neural network. |
| MuZero | 2019-20 | *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (Schrittwieser et al.) | Learns the model it plans with. |
| DreamerV3 | 2023 | *Mastering Diverse Domains through World Models* (Hafner et al.) | Learns a world model and trains the policy inside it. |

If you want to understand only a few, learn **DQN, PPO and SAC**. Almost every newer method is a modification of one of those three, and the 2026 papers I mentioned (PPO+, ExO-PPO, distributional Q(λ)) are variants of PPO and DQN.

### 3) How a DRL implementation works

A DRL implementation has two phases: interacting and learning. The loop is:

```
for each step:
    action     = policy(state) + exploration noise
    next_state, reward, done = env.step(action)
    store (state, action, reward, next_state, done)
    sample a batch of stored experience
    update the network(s) toward better predictions / better actions
```

The differences between algorithms mostly come down to what the network predicts, what the update target is, and whether the data is reused (off-policy: DQN, SAC, TD3) or thrown away after each update (on-policy: PPO, A2C). The core of DQN's update is this:

```python
q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)              # what we predicted
target = r + GAMMA * (1 - d) * q_target(s2).max(1).values     # what Bellman says it should be
loss = smooth_l1_loss(q_sa, target)                           # shrink the gap
```

I wrote two files for you, plus a short guide to which to use:

- **`dqn_from_scratch.py`** is a complete DQN in about 100 lines on CartPole. Read it first, because it contains every moving part: the Q-network, replay buffer, epsilon-greedy exploration, target network and TD loss.
- **`highway_sb3_benchmark.py`** is what you'd use for your project. It trains DQN or PPO from Stable-Baselines3 on highway-env across several seeds and reports collision rate, mean speed, return and episode length. Swap `merge-v0` for your custom environment.

In practice you won't implement algorithms from scratch. You use a library such as Stable-Baselines3, SBX or CleanRL and spend your effort on the environment, reward design and evaluation. For the 2026 methods, you'll mostly use the authors' code.

### Planning note

With 5 people, one week to select five 2026 models, and a demo roughly 6 weeks out, I'd suggest:

1. **This week:** Each person reads the DQN, PPO and SAC papers or a good tutorial. OpenAI's Spinning Up and the SB3 docs are the best starting points. Meanwhile, shortlist 2026 methods that have public code.
2. **Next 2 weeks:** Finish the custom environment and a shared evaluation script, and run the classic baselines to validate the setup.
3. **Weeks 4-5:** Train and compare the five models, then write the poster.
4. **Week 6:** Buffer time, since the date may shift.

I can help you choose the five 2026 models with a quick check of which have code. I can also help you define the metrics for the midterm report.

## Summary 2 Additional context and foundational DRL

 - I think I understand the core idea, and what makes RL DRL, but I don't know how the actions, policies, values, and states are represented, so it's difficult to picture it
 - the history kinda went over my head, but I just need to know DQN is the big start in 2013-2015
 - I want to know what the types of models are at a high level
 - DQN, PPO, SAC - those are the big ones that 
 - what is the Bellman equation
 - libraries - Stable-Baseline3, SBX, CleanRL
 - and for planning So heavy research this week
 - also i read the files and they don't make sense
 
 next steps:
  - get at what reinforcement learning is
  - what is the bellman equation - it seems important
  - try to understand DQN better
  - next try to understand the higher level of the other types of models - other value-based, policy gradient, off policy actor-critic, and i guess planning and model-based. they might be too high-level though idk
  - then specifically how PPO and SAC works
 

## Prompt 3 Clarify foundational DRL

Those questions might've been too loaded, a lot of it went over my head. I'll start with trying to understand RL/DRL.

To try to clarify, a state is what the agent is in, the action is what the agent can do in that state, and the reward is how good the new state is (e.g. V(s')). The agent wants to be in the state with the highest reward (the highest V). Q(s,a) is what the new state that the action will lead the agent to, and the value of that new state? And the policy is the pattern of actions to take on the state that lead to the highest value? 

Could you also give one or two simple examples?


## Response 3 Clarify foundational DRL

You have the state and action right. The reward, V and Q are where it gets slightly tangled.

## Fixing the definitions

**State** is the situation the agent observes right now, such as its position in a maze or the positions and speeds of nearby cars.

**Action** is a choice the agent can make in that state, such as move left, speed up or change lane.

**Reward** is not V(s'). It's a single number the environment hands back right after an action, like a score for that one step: +10 for reaching the goal, -1 for crashing, 0 for most ordinary steps. It's immediate and short-term.

**Value, V(s)**, is the total reward the agent expects to collect from now until the end, starting from state s and following its policy. It's long-term. A state can have a high value even if the reward there is zero, because good things are coming. Future rewards are discounted by a factor γ (say 0.9) so sooner is worth more than later.

So the agent isn't trying to reach "the state with the highest reward." It's trying to maximize the total reward over the whole episode. Value is how it measures how promising a state is.

**Q(s, a)** is the expected total reward if you take action a in state s and then keep playing well. Unlike V, it scores a state-action pair, so it lets you compare actions. It's not the new state itself. It's roughly the reward you get now, plus the discounted value of where you end up.

**Policy** is a rule that says what to do in each state ("in this state, do this"). It's a mapping from states to actions, not a sequence of actions. The best policy is the one that gets the highest value. If you know Q, the policy is simple: pick the action with the highest Q.

## Example 1: a tiny corridor

There are four cells in a row: **A, B, C, Goal**. The agent starts in A and can move left or right. Reaching the Goal gives +10 and ends the episode. Every other step gives 0. Let γ = 0.9.

- From C, moving right gives +10, so **V(C) = 10**.
- From B, one step to C gives 0 now and 10 later, discounted once: **V(B) = 0.9 × 10 = 9**.
- From A: **V(A) = 0.9 × 9 = 8.1**.

Notice that the reward is 0 in A, B and C, yet their values are high. Value captures that the goal is close.

Now Q. In B, moving right leads to C, so **Q(B, right) = 9**. Moving left leads back to A, so **Q(B, left) = 0.9 × 8.1 = 7.29**. Right has the higher Q, so the best policy in B is "go right." The best policy overall is "always go right."

At the start, the agent knows none of these numbers. It has to learn them by trying things and seeing what rewards come back.

## Example 2: your highway merge

- **State:** the ego car's position and speed, plus those of nearby cars.
- **Actions:** speed up, slow down, change lane, keep going.
- **Reward:** a small positive amount for driving fast, a large negative amount for a collision, and perhaps a bonus for taking the exit.
- **Value of a state:** how well things are likely to go from here. A state where a crash is imminent has very low value even though no reward has been lost yet.
- **Q(state, speed up):** how good it is to speed up right now. It might be low if there's a car in the gap you're trying to merge into, and high if the gap is open.
- **Policy:** the driver's decision rule. It looks at the traffic and picks the action with the best long-term outcome.

## Where "deep" comes in

In the corridor you could write V and Q in a small table. In highway driving, the number of possible situations is far too large for a table. Deep RL uses a neural network that takes the state as input and outputs Q-values (as in DQN) or action probabilities (as in PPO). Training adjusts the network's weights until its predictions match what the rewards actually turn out to be.

If you'd like, I can follow up with a short exercise like computing Q-values by hand on a slightly bigger grid, or explain how the agent learns these numbers by trial and error.

## Summary 3 and Prompt 4 Clarify foundational DRL and Check understanding foundational DRL
  
My Understanding:
 - Okay, reward is from the environment and its what we specify. It's tied to specific states? So if I'm in a car on a typical, non-problematic state, take an action (turn left) and crash (new state), because I crashed I get a reward of -10. We - the designers of the environment - determine what those rewards are
 - Value is the current reward of that state, or if you followed a series of optimal steps (an optimal policy) to get to that reward? In the corridor example, V(C) is 10 because the action there that has a reward of 10. It's the maximum of the values of the actions at those states, in addition to the decremented (by gamma) value of the optimal actions of future states. It seems quite difficult to calculate. 
 - Q(s,a) returns gamma * V(s'). Q(B, right) returns 0.9 * V(C) = 9, Q(B, left) returns 0.9 * V(A)

 Questions: 
 - What does a policy look like? 
 - What does the learning process look like from beginning to end? What does the environment look like? the model presumably starts from nothing, how does it learn a policy, V(s), etc.?

## Response 4 Check understanding foundational DRL

 Your reward and value ideas are mostly right, with two small corrections below.

### Checking your understanding

**Reward:** Yes. You, the designer, decide it, and you attach it to what just happened. Strictly, it's tied to the *transition* (state, action, new state), not just the state. Your crash example is right: turning left in a normal state led to a crash, and that step yields -10.

**Value:** V(s) is not the current reward. It's the total discounted reward you'd collect from here onward if you followed a good policy. Your corridor reasoning was correct: V(C) = 10 because the best action there pays 10 and the episode then ends. The formula you described is the real one:

> V(s) = max over actions of [ reward + γ · V(next state) ]

You're right that it looks hard to compute, because every state's value depends on the values of the states after it. In a big problem like driving you can't solve it directly. Learning (below) is how the agent estimates these values from experience.

**Q:** Almost. You left out the immediate reward:

> Q(s, a) = reward for taking a + γ · V(s')

In the corridor, the immediate reward is 0 for every step except the last, so your version gave the right numbers. If a step has a reward of its own, as in a crash or the final +10, Q needs it. For example, Q(C, right) = 10 + 0.9 × 0 = 10.

### What a policy looks like

A policy is just a function from a state to an action. In its simplest form it's a lookup table:

| State | Action |
|---|---|
| A | right |
| B | right |
| C | right |

In a big problem like your highway, it's a neural network. You feed in the state (positions and speeds of nearby cars, as numbers) and it outputs something like:

| speed up | slow down | left | right | keep |
|---|---|---|---|---|
| 0.05 | 0.10 | 0.60 | 0.05 | 0.20 |

A policy may be *stochastic* (it samples from these probabilities) or *deterministic* (it always takes the top choice). Value-based methods like DQN don't store the policy at all. They learn Q and the policy is simply "pick the action with the highest Q."

### The learning process from start to finish

**The environment** is a simulator with two basic functions:

- `reset()` starts a new episode and returns the starting state.
- `step(action)` returns the new state, the reward, and whether the episode is over.

Highway-env is exactly this. It simulates the other cars, and the rewards are computed by code you control. The agent doesn't know the rules of the road or how the traffic behaves. It only sees states and rewards.

**Starting from nothing:** the agent's Q-values (or network weights) are all zeros or random, so its behavior is essentially random. It learns by repeating a loop: try something, see the reward, update its estimates.

Here is the corridor learned this way. Q starts at 0 everywhere (γ = 0.9). After each step, update the entry you just used:

> Q(s, a) ← reward + γ · (best Q in the new state)

- **Early episodes:** the agent wanders randomly, and every update gives 0 because nothing has been rewarded yet. Eventually it happens to step from C into the goal and gets +10, so **Q(C, right) = 10**.
- **A later episode:** the agent reaches C again, moving from B. Now **Q(B, right) = 0 + 0.9 × 10 = 9**.
- **Later still:** from A, **Q(A, right) = 0 + 0.9 × 9 = 8.1**.

Information about the reward spreads backward, one step per visit, from the goal toward the start. That's how the agent "calculates" values without ever solving the equations. Each update is a small local correction, repeated many times. In practice you take a small step toward the new value instead of replacing the old one outright, so noisy experience gets averaged out.

**Exploration:** if the agent only ever did the best-looking action, it would never discover better ones. So it takes a random action some fraction of the time (called epsilon-greedy), often starting high and decaying.

**Going from the table to deep RL:** a table has one entry per state, which is impossible for highway driving. Instead, a neural network takes the state and outputs Q-values, and each "update" is a gradient step that nudges the network's output toward reward + γ · (best Q in the next state). That is exactly the three-line core in `dqn_from_scratch.py` from earlier. Methods like PPO skip Q and nudge the policy network directly: actions that led to better-than-expected outcomes become more likely.

**What it looks like over a training run on your merge task:**

1. **Start:** random behavior, with frequent crashes and low rewards.
2. **Early:** the agent learns that crashing is bad and mostly stays in its lane.
3. **Middle:** it learns that it must find a gap and merge, and some attempts succeed.
4. **Late:** merges are consistent, and it improves speed and exit-taking.

You'll plot the average reward per episode over time, which rises as it learns. Then you test the trained policy on fresh episodes, which is where your collision rate and exit time come from.

I can write a tiny runnable version of this corridor learning (about 20 lines, no neural networks) so you can watch the Q-values fill in. It's a good way to make this feel concrete before moving on to the deep versions.

 ## Summary 4 and Prompt 5 Check understanding foundational DRL and step by step DRL example

 So an example of an episode in the tiny corridor

 The agent starts in A. There's only one action, Q(A, right), which is 0. It takes that action, receives a reward of 0, and computes Q(A, right) <- 0 + 0.9(0) = 0

 Now it's in B. Q(B, left) and Q(B, right) are both 0, so it chooses randomly. Lets say it chooses Q(B, right). That also gets updated to 0.

 Now it's in C. Q(C, left) and Q(C, right) are both 0. It picks randomly, chooses Q(C,right), and gets a reward of +10. Now, Q(C,right) is set to 10 + 0.9(0) = 10. 

 New episode

 Starts in A, takes same action.

 Now in B, Q(B, right) is still 0, but if it takes it, it gets updated with 0 + (0.9)(10) since Max(Q(C, a)) is now 10, so Q(B, right) is now set to 9, and Max(Q(B, a)) is 9

 New episode

 starts in A, now when it takes Q(A, right), that gets updated to Q(A, right) <- 0 + (0.9)(9) = 8.1

 Every subsequent episode, when taking the greedy approach now always takes the same three step path since all the appropriate Qs are filled. 

 Is this correct?

## Response 5 Step by step DRL example

Yes, your trace is correct, including the order of the updates. You also handled the +10 step properly: the goal ends the episode, so there's no future term, and Q(C, right) = 10 + 0.9 × 0 = 10. What you've worked out by hand is **Q-learning**, the tabular ancestor of DQN.

### Three details worth noticing

**1. Your episodes were lucky.** In episode 2, the agent at B had two zero-valued options and you picked "right." If it had picked "left," it would have gone back to A, and Q(B, left) would have been updated to 0 + 0.9 × max Q(A, ·) = 0, which changes nothing. The agent would keep wandering until it happened to reach C again. Real runs are noisier, and the number of episodes before the values fill in depends on that luck. This is why exploration matters.

**2. The values flow backward, one step per episode.** You saw it: the +10 reached C in episode 1, B in episode 2 and A in episode 3. In a longer corridor, you'd need roughly one episode per cell of distance for the signal to reach the start. This slow backward spread is a main reason deep RL needs so much data.

**3. Unvisited actions keep their starting guess.** After your three episodes, Q(B, left) is still 0 because you never tried it. Its true value is 0.9 × 8.1 = 7.29. That doesn't matter here, since 7.29 is still below 9 and "right" is correctly preferred. But in general, if the starting guess is lower than the truth, the agent may never discover that an action is good. Epsilon-greedy exploration (a small chance of random action) fixes this, because it lets the agent occasionally try an action that currently looks worse and correct its estimate.

### One simplification you made

You replaced the old Q-value entirely with the new estimate. That works here because the corridor is deterministic: the same action always gives the same result. In a noisy environment like highway traffic, one experience may be unrepresentative. So the real update moves only part of the way:

> Q(s, a) ← Q(s, a) + α · [ (reward + γ · max Q(s', ·)) − Q(s, a) ]

Here α is a small learning rate, say 0.1. Setting α = 1 gives exactly what you did. A smaller α averages over many experiences. In deep RL, this same role is played by the gradient step that nudges the network toward the target.

Next I could write the runnable tabular version of your corridor, with the printed Q-table after each episode, so you can check your hand trace against it. Or we could move on to the step from this table to a neural network. Which would you prefer?

## Summary 5

Okay, so in a real environment, the Qs for each state gets updated gradually over time per the learning rate alpha. 

what are some common metrics associated with DRL? How do you evaluate models?
(above not prompted accidentally)

So, I think we can try to start understanding DQN? and ask what Bellman's equation is 


## Prompt 6 DQN and Bellman

Okay, I think I get what simple RL generally looks like. can you explain what Bellman's equation is? and what DQN is at a high level? Also with a simple example. 

## Response 6 DQN and Bellman

## Bellman's equation

The idea is that **the value of a situation equals what you get right now, plus the discounted value of the situation you land in.** A long-term question ("how much total reward from here?") becomes a one-step question ("what do I get now, and where does that leave me?").

For the best possible behavior:

> **Q\*(s, a) = r + γ · max Q\*(s′, a′)**

The max is over the actions a′ you could take in the next state s′. Equivalently, V\*(s) = max over a of Q\*(s, a). This is the formula you've already been using. In the corridor, **Q(B, right) = 0 + 0.9 × max Q(C, ·) = 0.9 × 10 = 9**.

There are two things to know about it:

- **It's a consistency check.** The true Q-values must satisfy this equation at every state-action pair. If your current estimates don't, they're wrong somewhere. The gap between the two sides (the estimate Q(s, a) versus r + γ · max Q(s′, ·)) is called the **Bellman error**.
- **It gives a recipe for learning.** Whenever the two sides disagree, move the left side toward the right side. That's the update rule you traced by hand: Q-learning is the Bellman equation turned into an update. When the Bellman error is zero everywhere, you've found the right values.

In a noisy environment the right-hand side becomes an average over the possible outcomes (an expectation), but the idea is the same.

## DQN at a high level

**DQN (Deep Q-Network)** is Q-learning where the table is replaced by a neural network.

- **Input:** the state, as a vector of numbers (in your project, the positions and speeds of nearby cars).
- **Output:** one Q-value per action, e.g. [Q(speed up), Q(slow down), Q(left), Q(right), Q(keep)].
- **Acting:** pick the action with the highest output, or a random one occasionally (epsilon-greedy).
- **Learning:** treat the Bellman right-hand side as the *label* and train the network on the gap:

> target = r + γ · max Q(s′, ·)
> loss = (Q(s, a) − target)²

That's the same update you did by hand, except instead of overwriting a table cell, a gradient step adjusts the network's weights so its output moves toward the target.

### A numeric example

Suppose the network currently says Q(B, right) = 2.0, while the true value is 9. The agent takes "right" in B and sees reward 0 and next state C. For C, the network says the best Q is 10.

- target = 0 + 0.9 × 10 = **9.0**
- prediction = 2.0
- loss = (2.0 − 9.0)² = 49

The gradient step then moves the network's output for (B, right) toward 9, say to 2.7. After many such steps, across many states and actions, the predictions converge on the true values.

One new consequence: because it's a network, changing the weights for one state also changes the predictions for *similar* states. That's the benefit, since the agent can handle states it has never seen, which is essential in driving. It's also the risk, because updates can interfere with each other.

### Why plain "network + Q-learning" breaks, and DQN's two fixes

A naive version is unstable. DQN's contribution was two stabilizing tricks.

1. **Replay buffer.** Consecutive steps are highly similar (the car is in nearly the same situation for many steps), and training on a stream of near-duplicates makes the network overfit to the latest events and forget older ones. So the agent stores transitions (s, a, r, s′, done) in a big buffer and trains on **random mini-batches** from it. This breaks the correlation and lets each experience be reused many times.

2. **Target network.** The target r + γ · max Q(s′, ·) is computed by the very network you're updating, so each update moves the target as well, like chasing your own shadow. DQN keeps a second, **frozen copy** of the network to compute targets and only syncs it with the main network every few hundred steps. This holds the target still long enough for learning to make progress.

Both appear in `dqn_from_scratch.py` as `ReplayBuffer` and `q_target`.

## Limitations to be aware of

- **Discrete actions only.** The `max` over actions requires you to evaluate each one. That works for "left / right / faster / slower" (highway-env's discrete mode) but not for continuous steering and throttle. That's why SAC and TD3 exist.
- **Overestimation.** Taking the max of noisy estimates tends to produce values that are too high, which Double DQN corrects.
- **Many later methods are DQN plus a fix.** Dueling networks, prioritized replay, distributional Q (C51, QR-DQN) and Rainbow all keep this same core, so understanding it makes the rest of the family much easier. One of the 2026 papers I mentioned (off-policy distributional Q(λ)) builds on this family.

Next, I could walk through the PPO or policy-gradient idea in the same style, since it's the main alternative to Q-learning and would be one of your benchmarks. Or I can write the runnable tabular corridor so you can see the Bellman updates in action first. Tell me which sounds more useful.

## Summary 6 DQN and Bellman (and TDL 2)

(not prompted)
TDL 
- acquire baseline understand of DRL
  - what are the metrics
  - understand the core models in the various family of functions
    - *DQN - Q learning
    - *PPO - policy gradient 
    - SAC - off-policy actor-critic
- research
  - Starting points: 
    - PPO+, 
    - ExO-PPO, 
    - Off-policy Distributional Q(lambda), 
    - Reversal Q-learning, and 
    - SimBa. 
    - NOTE: But first check if they do not contain public repos or have an implementation somewhere. We cannot implement them if they do not contain public repos. The five models we need to consider should have implementations somewhere.
      - NOTE: possible implementations include: Stable-Baselines3, SB3-Contrib, and SBX. But recent models are likely not be included in libraries.
- finish assignment

In other words:
1) Bellman's equation is how to find the estimate of `Q` - which is `Q*`. 
2) `Q` is the return of an action in a particular state. 
3) Return is the sum of max future rewards possible, while you are in a state
4) You estimate `Q` by taking the current reward `r` of taking an action `a` in state `s`, and adding a discounted (discounted because applying `lambda`) max reward of the new state `s'`, represented by `max Q*(s', a')`. In all, `Q*(s,a) = r + lambda * max Q*(s', .)`.
5) `max Q*(s', .)` approximates `V*(s)` 
6) `V*(s)` is the estimate of `V(s)`
7) `V(s)` is the maximum return of being in s. 

When you say Bellman's equation is a consistency check, the LHS of the equation is what the model currently thinks Q is, but when it takes that action in s, it calculates the RHS and updates its knowledge accordingly. The RHS may not always be accurate (particularly when it isn't familiar with the new state s'), but it will eventually approach the truth over (potentially) many episodes. This is also why it's a recipe for learning. I can see why this is a foundational equation for DRL. 

For DQN:
1) The previous process involves storing `Q(s,a)` (or what the model thinks `Q(s,a)` is) for every possible action in every possible state, which is a significant memory overhead, and unfeasble for situations such as automated driving (with many states that have each have many actions).
2) DQN improves on that process, by replacing that table with a deep neural network that learns `Q(s,a)` from `s`, with a predefined set of actions `a` in each state. The input is a state, the output is the Q of each action in that state. 
3) Thus, if `a` is continuous or varies between states (such that two states may have different sets of actions), DQN may not be the appropriate model
4) DQN uses the two sides of Bellman's equation, LHS: `Q(s,a)` and RHS: `r + lambda * max Q(s', .)` to compute the loss. The LHS is the current prediction (`y-hat`), the RHS is the target or label or ground truth (`y`).
5) The most naive way to apply neural network + Q learning is to train on every new subsequent state. The main problems with this are 1) a skewed distribution in the variance of states (consecutive states tend to be similar), and 2) that the agent doesn't actually *know* the target, and the target changes with learning. 
6) The first point 1) is addressed in DQN with a replay buffer, where the agent accumulates every transition into a memory (replay buffer), and trains on mini-batches from its remembered tranistions instead of consecutive states.
7) Transitions consist of (s, a, r, s', done), which are the prior state, the current action, the reward of that action, the new state, and whether the agent accomplished the goal, which is defined by the environment (e.g. in the corridor example, making it to the last room). These are all the variables that the neural network needs to predict Q. 
8) The second point 2) is addressed by creating a duplicate network that computes the target, and gets updated significantly less frequently. This allows the target to be static enough for loss to be calculated without the DQN chasing its own tail. Thus, the first network computes `Q(s,a)` when making a prediction (`y-hat`), and the second frozen networks computes `max Q(s, .)` when computing the target `y` when computing the loss. 

Follow-up questions for DQN:
1) When every transition is stored in the replay buffer, isn't there still a higher proportion of very similar states (e.g. the car is driving and isn't near anyone), which adds bias to training? Or is the original problem not the proporition of similar states but their consecutive nature?
2) Isn't the storage of transitions in the replay buffer a significant memory overhead? Or is there a limit to the amount of transitions that it stores, where it only keeps the X most recent (or highest value?) transitions?
3) What activation functions does the network in DQN typically use? ReLU? And I assume it uses softmax to compute the output?


## Prompt 7 Clarify DQN and Bellman

In other words:
1) Bellman's equation is how to find the estimate of `Q` - which is `Q*`. 
2) `Q` is the return of an action in a particular state. 
3) Return is the sum of max future rewards possible, while you are in a state
4) You estimate `Q` by taking the current reward `r` of taking an action `a` in state `s`, and adding a discounted (discounted because applying `lambda`) max reward of the new state `s'`, represented by `max Q*(s', a')`. In all, `Q*(s,a) = r + lambda * max Q*(s', .)`.
5) `max Q*(s', .)` approximates `V*(s)` 
6) `V*(s)` is the estimate of `V(s)`
7) `V(s)` is the maximum return of being in s. 

When you say Bellman's equation is a consistency check, the LHS of the equation is what the model currently thinks Q is, but when it takes that action in s, it calculates the RHS and updates its knowledge accordingly. The RHS may not always be accurate (particularly when it isn't familiar with the new state s'), but it will eventually approach the truth over (potentially) many episodes. This is also why it's a recipe for learning. I can see why this is a foundational equation for DRL. 

For DQN:
1) The previous process involves storing `Q(s,a)` (or what the model thinks `Q(s,a)` is) for every possible action in every possible state, which is a significant memory overhead, and unfeasble for situations such as automated driving (with many states that have each have many actions).
2) DQN improves on that process, by replacing that table with a deep neural network that learns `Q(s,a)` from `s`, with a predefined set of actions `a` in each state. The input is a state, the output is the Q of each action in that state. 
3) Thus, if `a` is continuous or varies between states (such that two states may have different sets of actions), DQN may not be the appropriate model
4) DQN uses the two sides of Bellman's equation, LHS: `Q(s,a)` and RHS: `r + lambda * max Q(s', .)` to compute the loss. The LHS is the current prediction (`y-hat`), the RHS is the target or label or ground truth (`y`).
5) The most naive way to apply neural network + Q learning is to train on every new subsequent state. The main problems with this are 1) a skewed distribution in the variance of states (consecutive states tend to be similar), and 2) that the agent doesn't actually *know* the target, and the target changes with learning. 
6) The first point 1) is addressed in DQN with a replay buffer, where the agent accumulates every transition into a memory (replay buffer), and trains on mini-batches from its remembered tranistions instead of consecutive states.
7) Transitions consist of (s, a, r, s', done), which are the prior state, the current action, the reward of that action, the new state, and whether the agent accomplished the goal, which is defined by the environment (e.g. in the corridor example, making it to the last room). These are all the variables that the neural network needs to predict Q. 
8) The second point 2) is addressed by creating a duplicate network that computes the target, and gets updated significantly less frequently. This allows the target to be static enough for loss to be calculated without the DQN chasing its own tail. Thus, the first network computes `Q(s,a)` when making a prediction (`y-hat`), and the second frozen networks computes `max Q(s, .)` when computing the target `y` when computing the loss. 

Follow-up questions for DQN:
1) When every transition is stored in the replay buffer, isn't there still a higher proportion of very similar states (e.g. the car is driving and isn't near anyone), which adds bias to training? Or is the original problem not the proporition of similar states but their consecutive nature?
2) Isn't the storage of transitions in the replay buffer a significant memory overhead? Or is there a limit to the amount of transitions that it stores, where it only keeps the X most recent (or highest value?) transitions?
3) What activation functions does the network in DQN typically use? ReLU? And I assume it uses softmax to compute the output?

## Response 7 Clarify DQN and Bellman

Your understanding is mostly solid, and the consistency-check paragraph is exactly right. A few of the numbered points need fixing, though.

### Corrections to your summary

1. **Q\* is the true answer, not an estimate.** Q\* is the real optimal Q-value, which the Bellman equation describes. What the agent learns is **Q**, an estimate that gets closer to Q\* over time. "Bellman's equation is the condition Q\* satisfies" is the clean way to say it.
2. **Return is the sum of *discounted* future rewards.** The "max" comes from behaving optimally, not from the definition of return. Q(s, a) is the *expected* return if you take a in s and then act well afterward.
3. **The discount is γ (gamma), not λ.** This matters because λ is a different parameter used in methods like TD(λ), including the "Distributional Q(λ)" paper I mentioned earlier.
4. **max Q\*(s′, ·) equals V\*(s′), not V\*(s).** It's the value of the *next* state. V\*(s) = max over a of Q\*(s, a), and V\* is also the true value, not an estimate of V. In general, V(s) is the expected return from s under some policy, and V\* is that value under the best policy.
5. **A transition is not "all the variables the network needs to predict Q."** The network's input is only the state s, and its outputs are the Q-values. The action a picks which output to train. The reward r, next state s′ and done flag are used to build the *target*. Also, `done` means the episode ended, whether by reaching the goal or by crashing. When it's true, the future term is dropped: target = r.
6. **On the target network:** it computes max Q(s′, ·), the *next* state, rather than s.

Two refinements to your other points. For DQN point 3, the set of actions can vary by state if you use *action masking* (illegal actions are excluded from the max), as in Maskable PPO from your list. And in the consistency-check paragraph, using your own estimates of s′ to update s is called **bootstrapping**. It's why early targets can be wrong, and why convergence takes many episodes.

### Your follow-up questions

**1) Similar states and bias.** Both concerns are real, but the replay buffer only fixes one of them.

- Replay fixes the *consecutive* problem. Neural networks assume training examples are roughly independent. A stream of near-identical consecutive frames violates that, and shuffling restores it.
- It does **not** fix *imbalance*. If 95% of the buffer is "driving alone on an empty road," the rare, critical situations (near-collisions, merge gaps) are underrepresented, and the network may learn them poorly. This is a real issue for your project.
- Why the skewed data doesn't invalidate Q-learning: the Bellman equation holds for any transition, regardless of which policy generated it. That's why Q-learning is **off-policy** and can learn from old or random data.

Common remedies for imbalance are **prioritized experience replay** (sample transitions with a large Bellman error more often), designing your custom environment so episodes start in dense, difficult traffic, and rewarding the right behaviors.

**2) Memory.** Yes, the buffer has a fixed capacity (typically 10⁵ to 10⁶ transitions). It works as a first-in-first-out queue: when full, the oldest transitions are overwritten. For your project, memory is small. A highway-env state might be about 25 numbers, so a million transitions is a few hundred MB at most. My template uses a 15,000-transition buffer. Memory becomes a concern for image inputs like Atari, which need tricks like storing 8-bit pixels.

**3) Activations and output.** Hidden layers typically use **ReLU** (the original DQN did, and I believe SB3's DQN does by default). The output layer has **no softmax** and no activation at all. It's a plain linear layer that outputs raw Q-values. Q-values aren't probabilities. They can be 9, -10 or 0.37 and don't need to sum to 1. The agent just takes the argmax.

You're right that softmax appears in the other family. A policy-gradient network like PPO's with discrete actions outputs probabilities over actions, so it uses softmax at the end. That's a good setup for the next topic.

I'd suggest policy gradients and PPO next, since it's likely one of your benchmark models and the 2026 candidates PPO+ and ExO-PPO build on it. Or I can write the tabular corridor code first. Which would you prefer?

## Summary 7 Clarify DQN and Bellman

refined: 

In other words:
1) Bellman's equation is how to find the estimate of `Q*` - which is `Q`. 
2) `Q` is the expected return of an action in a particular state, and do well afterwards. 
3) Return is the sum of discounted future rewards, represented by `max Q(s',a')`
4) You estimate `Q` by taking the current reward `r` of taking an action `a` in state `s`, and adding a discounted (discounted because applying `gamma`) max expected return of the new state `s'`, represented by `max Q*(s', a')`. In all, `Q*(s,a) = r + gamma * max Q*(s', .)`.
5) `max Q*(s', .)` approximates `V*(s')` 
6) `V*(s)` is the truth, and is estimated by `V(s)`
7) `V(s)` is the expected return of being in s, under some policy. 

When you say Bellman's equation is a consistency check, the LHS of the equation is what the model currently thinks Q is, but when it takes that action in s, it calculates the RHS and updates its knowledge accordingly. The RHS may not always be accurate (particularly when it isn't familiar with the new state s') (bootstrapping = using own estimate of s'), but it will eventually approach the truth over (potentially) many episodes. This is also why it's a recipe for learning. I can see why this is a foundational equation for DRL. 

For DQN:
1) The previous process involves storing `Q(s,a)` (or what the model thinks `Q(s,a)` is) for every possible action in every possible state, which is a significant memory overhead, and unfeasble for situations such as automated driving (with many states that have each have many actions).
2) DQN improves on that process, by replacing that table with a deep neural network that learns `Q(s,a)` from `s`, with a predefined set of actions `a` in each state. The input is a state, the output is the Q of each action in that state. 
3) Thus, if `a` is continuous or varies between states (such that two states may have different sets of actions), DQN may not be the appropriate model. (The inconsistency between states can be fixed with action masking)
4) DQN uses the two sides of Bellman's equation, LHS: `Q(s,a)` and RHS: `r + lambda * max Q(s', .)` to compute the loss. The LHS is the current prediction (`y-hat`), the RHS is the target or label or ground truth (`y`).
5) The most naive way to apply neural network + Q learning is to train on every new subsequent state. The main problems with this are 1) a skewed distribution in the variance of states (consecutive states tend to be similar), and 2) that the agent doesn't actually *know* the target, and the target changes with learning. 
6) The first point 1) is addressed in DQN with a replay buffer, where the agent accumulates every transition into a memory (replay buffer), and trains on mini-batches from its remembered tranistions instead of consecutive states.
7) Transitions consist of (s, a, r, s', done), which are the prior state, the current action, the reward of that action, the new state, and whether the agent accomplished the goal, which is defined by the environment (e.g. in the corridor example, making it to the last room), or if the agent crashed. These are all the variables that the agent needs to predict Q, and compute the loss. 
8) The second point 2) is addressed by creating a duplicate network that computes the target, and gets updated significantly less frequently. This allows the target to be static enough for loss to be calculated without the DQN chasing its own tail. Thus, the first network computes `Q(s,a)` when making a prediction (`y-hat`), and the second frozen networks computes `max Q(s', .)` when computing the target `y` when computing the loss. 

Follow-up questions for DQN:
1) When every transition is stored in the replay buffer, isn't there still a higher proportion of very similar states (e.g. the car is driving and isn't near anyone), which adds bias to training? Or is the original problem not the proporition of similar states but their consecutive nature?
 - Response: Yes that is a problem, replay buffer fixes the consecutive problem. 
2) Isn't the storage of transitions in the replay buffer a significant memory overhead? Or is there a limit to the amount of transitions that it stores, where it only keeps the X most recent (or highest value?) transitions?
 - yes, typically 10^5 or 10^6 transitions
3) What activation functions does the network in DQN typically use? ReLU? And I assume it uses softmax to compute the output?
 - ReLU yes, not softmax just does raw linear


## Prompt 8 Metrics

Before discussing PPO, I first want to ask what metrics are common to evaluate DRL models on? categorical-based DL models utilize accuracy, recall, f1, etc., while regression-based models are compared on MSE, R2, etc. What about DRL?

## Response 8 Metrics

The main difference from supervised learning is that there's no fixed dataset or ground-truth label, so you evaluate **behavior** by running the trained agent in the environment. Loss isn't a performance metric either. A DQN's TD loss can go up while the policy improves, because the targets are moving.

### 1) Return-based metrics (the standard core)

- **Episodic return:** the total (usually undiscounted) reward per episode. You report the mean over many evaluation episodes, with the standard deviation or a confidence interval. This is the RL counterpart of accuracy.
- **Learning curve:** return plotted against training steps. It is the most common figure in RL papers.
- **Final performance:** the mean return at the end of training, or averaged over the last N evaluations.
- **Normalized score:** return scaled between a random policy and a reference (human-normalized on Atari). It lets you aggregate across environments with different reward scales.

### 2) Efficiency and stability

- **Sample efficiency:** how many environment steps are needed to reach a given performance, or the area under the learning curve. This is the main selling point of most recent methods (CrossQ, SimBa, BRO).
- **Wall-clock time and compute:** training time and hardware. This matters for fair comparison, since a method that is 3x more sample-efficient but 10x slower per step is not obviously better.
- **Variance across seeds:** RL results are notoriously variable. Two runs of the same algorithm with different random seeds can differ a lot. Henderson et al.'s *Deep Reinforcement Learning That Matters* (2018) is the classic paper on this.
- **Stability:** whether performance collapses or oscillates during training.

### 3) Robustness and generalization

- **Evaluation on unseen conditions:** different traffic densities, speeds or random seeds not used in training. A policy can overfit to its training scenarios.
- **Sensitivity to hyperparameters:** how much performance depends on careful tuning.

### 4) Task-specific metrics (most relevant to your project)

Return alone can hide what you care about, since a reward function trades off competing goals. For highway merging you'd also report:

- **Collision rate** (fraction of episodes ending in a crash). This is your safety metric.
- **Success rate** (merged and took the exit).
- **Time-to-exit** or episode length for successful runs, which is your "fastest time."
- **Average speed**, and possibly comfort measures like jerk or the number of lane changes.
- **Constraint cost:** in safe RL, the cumulative safety violations, reported separately from reward.

Because your original question was "fastest without accidents," I'd make collision rate and time-to-exit your headline metrics, with return as supporting evidence. Otherwise a result can look good simply because your reward weights favored one model.

### 5) Statistical rigor

Recent practice is to avoid reporting only a mean over 3 seeds. The influential paper here is *Deep Reinforcement Learning at the Edge of the Statistical Precipice* (Agarwal et al., NeurIPS 2021), which recommends:

- Many seeds, at least 5 and ideally 10 or more.
- The **interquartile mean (IQM)** instead of the plain mean, since it's less sensitive to outlier runs.
- **Bootstrapped confidence intervals** and **performance profiles** instead of single numbers.
- The **`rliable`** Python library, which computes these.

For a class project, a realistic target is 5 seeds per model, 50 or more evaluation episodes per seed, and means with confidence intervals. That would be more rigorous than most student projects.

### 6) Evaluation protocol details

- Evaluate with the policy **deterministic** (no exploration noise) on separate evaluation episodes, not the training episodes.
- Use the **same environment, reward and episode seeds** for every model.
- Report the training budget (total steps), so the comparison is fair.

`highway_sb3_benchmark.py` from earlier already does a basic version: several seeds, deterministic evaluation, and collision rate, speed, return and episode length.

Do you want to move on to policy gradients and PPO now, or first sketch which metrics and statistical tools you'd put in your midterm report?

## Summary 8 Metrics

In general:
 - Episodic return
 - learning curve
 - final performance
 - normalized score
 - sample efficiency
 - compute time
 - variance across seeds
 - stability
 - evaluation on unseen conditions
 - sensitivty to hyperparemetrs
for our situation
 - collision rate
 - success rate
 - time-to-exit
 - average speed
 - constraint cost - cumulative safety violations
 - maybe return (episodic return?)
 - apply statistcal rigor - 5-10+ seeds, IQM, bootstrapped confidence intervals and performance profiles, and use rliable

 I'll return to what each of these mean exactly later. 


## Prompt 9 PPO

Yes, can you explain what PPO is?

## Response 9 PPO

### The core difference from DQN

DQN learns Q-values and derives a policy from them ("pick the highest Q"). PPO skips that step and **learns the policy directly**. A network takes a state and outputs the probability of each action. For discrete actions this is the softmax output I mentioned earlier, e.g. [speed up 0.05, slow down 0.10, left 0.60, right 0.05, keep 0.20]. For continuous actions it outputs the mean and spread of a Gaussian over, say, steering and throttle. The agent *samples* from this distribution, which gives it built-in exploration.

PPO builds on two earlier ideas, so I'll go through those first.

### Step 1: The basic policy gradient (REINFORCE)

The idea is trial and error:

1. Play an episode using the current policy.
2. If the episode went well, make every action taken in it **more likely**.
3. If it went badly, make those actions **less likely**.

The loss is roughly `-log π(a|s) × G`, where G is the return. Minimizing it increases the probability of actions that were followed by high returns.

This works but is very noisy. If one merge succeeds, every action in that episode gets rewarded, including a pointless lane change at the start. It takes many episodes to average out which actions actually mattered.

### Step 2: Advantages and the critic (actor-critic)

Instead of asking "was the return high?", ask "was the return higher than *expected* from this state?" That quantity is the **advantage**:

> A(s, a) ≈ Q(s, a) − V(s)

Suppose the state's value is V(s) = 5, meaning you typically get about 5 from here:

- You take an action and the outcome is worth 8, so A = +3 and that action becomes more likely.
- You take another action and the outcome is worth 2, so A = −3 and it becomes less likely.

If everything in this state scores around 5, the advantages are near zero and the policy barely changes. This removes much of the noise.

To estimate V(s) you need a second network, called the **critic**, which learns values the way DQN learns Q. The policy network is the **actor**. PPO is therefore an **actor-critic** method, and it uses both ideas from earlier: values from the Bellman equation, and a directly learned policy. In practice the advantage is computed with **GAE** (Generalized Advantage Estimation), which uses a parameter λ to balance noisy-but-unbiased estimates against smooth-but-biased ones. That's the other place λ shows up in RL.

### Step 3: The step-size problem, and PPO's fix

Updating a policy is delicate. The data you train on was generated by the *current* policy, so if one update makes the policy much worse, the next batch of data is also bad and it can spiral. Training collapses and may not recover. DQN doesn't have this problem as acutely, because its replay buffer holds old data from many policies.

PPO's solution is to keep each update small. For each action in the batch, it computes a **ratio**:

> ratio = π_new(a|s) / π_old(a|s)

A ratio of 1 means no change, 1.5 means the action is now 50% more likely, and 0.5 means half as likely. PPO then **clips** how much credit the update can get from a change:

> loss = −min( ratio × A, clip(ratio, 1−ε, 1+ε) × A )   with ε typically 0.2

The intuition, for a good action (A > 0): the update raises its probability, but once the ratio passes 1.2, the objective stops rewarding further increases, so the gradient there goes to zero. For a bad action (A < 0), the same happens when the ratio falls below 0.8. Policy changes are therefore capped at roughly ±20% per update round. This clipping, which is the "proximal" in the name, is what makes PPO stable and simple.

It also lets PPO reuse each batch for several passes. A naive policy gradient can use a batch once, but with the clip you can safely do several epochs on the same data.

### The full loop

```
repeat:
    1. collect e.g. 2048 steps using the current policy
    2. compute advantages (with the critic and GAE)
    3. for several epochs: update on mini-batches using
         clipped policy loss + value loss + small entropy bonus
    4. throw the batch away
```

The **entropy bonus** rewards the policy for staying a bit random, which preserves exploration. The data is discarded at step 4 because it came from the old policy. This makes PPO **on-policy**.

### PPO versus DQN

| | DQN | PPO |
|---|---|---|
| Learns | Q-values; policy is "argmax Q" | The policy directly (plus a value critic) |
| Actions | Discrete only | Discrete and continuous |
| Data | Off-policy: replay buffer reuses old data | On-policy: data used briefly, then discarded |
| Sample efficiency | Higher | Lower |
| Stability and tuning | Can be finicky | Robust, and a common default |
| Parallelism | Harder | Easy: run many environments at once |

PPO's weakness is sample efficiency, since it throws data away. Its strengths are stability and generality, which is why it became the default (Schulman et al., 2017, *Proximal Policy Optimization Algorithms*). For your project it handles both discrete and continuous highway-env action spaces, so it's a good baseline.

The 2026 PPO variants I mentioned earlier (PPO+ and ExO-PPO) target exactly these weaknesses: wasted data and a gap to off-policy methods in final performance. That makes them a natural fit for a "classic vs. new" comparison. I'd still check each paper's details and code availability before committing.

From here I can walk through how SAC works, since it's the off-policy actor-critic and the other major family, or write the tabular corridor code. Which would you prefer?

## Summary 9 PPO 


PPO:
 1) There are four general steps
 2) The first step, collect a batch e.g. 2048 steps with the current policy
 3) The current current policy initializes the likelihood of all actions to 0 or the same probability
 4) The second step, compute the advantage A
 5) To compute A, you need a critic and GAE
 6) The critic is a neural network that outputs the expected return of being in state s, V(s)
 7) The third step, divide the results of step 1) into mini batches and loop on each mini batch:
 8) For some predefined set of epochs, update the model on the current mini batch with clipped policy loss, value loss, and a small entropy bonus
 9) To compute the clipped policy loss, first compute the ratio = new pi(a|s) / old pi(a|s)
 10) Next, compute clipped policy loss = -min(ratio x A, clip(ratio, 1-epsilon, 1+epsilon) x A)
 11) epsilon is typically set to 0.2
 12) The clipping is to keep the updated small, which prevents spiraling and training collapse, leading to model instability
 13) For example, the ratio does not provide any more update if it is above 1.2, and any less update if it is below 0.8.  
 14) The entropy bonus rewards the policy for staying random, which promotes exploration
 14) The fourth step, through the batch away, and repeat at step 1



Follow up for previous ideas: 
 - In the first step, when you say collect a batch of steps, what are you collecting? A sequence of states, actions, and rewards?
 - what is the pi()?
 - what is the input for the critic network in PPO? s?
 - What is the equation for GAE?
 - how many epochs is PPO typically run on each mini batch?
 - how long do you repeat the loop in PPO? when does training stop?
 - How do you compute value loss?
 - what is the entropy bonus? how is it computed?


## Prompt 10 PPO 2

PPO:
 1) There are four general steps
 2) The first step, collect a batch e.g. 2048 steps with the current policy
 3) The current current policy initializes the likelihood of all actions to 0 or the same probability
 4) The second step, compute the advantage A
 5) To compute A, you need a critic and GAE
 6) The critic is a neural network that outputs the expected return of being in state s, V(s)
 7) The third step, divide the results of step 1) into mini batches and loop on each mini batch:
 8) For some predefined set of epochs, update the model on the current mini batch with clipped policy loss, value loss, and a small entropy bonus
 9) To compute the clipped policy loss, first compute the ratio = new pi(a|s) / old pi(a|s)
 10) Next, compute clipped policy loss = -min(ratio x A, clip(ratio, 1-epsilon, 1+epsilon) x A)
 11) epsilon is typically set to 0.2
 12) The clipping is to keep the updated small, which prevents spiraling and training collapse, leading to model instability
 13) For example, the ratio does not provide any more update if it is above 1.2, and any less update if it is below 0.8.  
 14) The entropy bonus rewards the policy for staying random, which promotes exploration
 14) The fourth step, through the batch away, and repeat at step 1



Follow up: 
 - In the first step, when you say collect a batch of steps, what are you collecting? A sequence of states, actions, and rewards?
 - what is the pi()?
 - what is the input for the critic network in PPO? s?
 - What is the equation for GAE?
 - how many epochs is PPO typically run on each mini batch?
 - how long do you repeat the loop in PPO? when does training stop?
 - How do you compute value loss?
 - what is the entropy bonus? how is it computed?


## Response 10 PPO 2

Your summary matches the overall structure. A few points need fixing first, then your follow-up questions.

### Corrections

- **Point 3:** The policy network starts with random weights, so its outputs are *roughly* uniform, e.g. [0.2, 0.2, 0.2, 0.2, 0.2] for five actions. Probabilities are never "0," since an action at 0 could never be sampled. Training then shifts these probabilities.
- **Points 7-8:** The epoch loop wraps the mini-batch loop, not the other way around. For each epoch you shuffle the whole 2048-step batch, split it into mini-batches (say 32 of size 64), and update on each. Repeat for all epochs. Each sample is therefore used once per epoch, about 10 times in total.
- **Point 13:** Clipping depends on the sign of A. For a **good** action (A > 0), the update stops rewarding once the ratio exceeds 1.2. For a **bad** action (A < 0), it stops once the ratio falls below 0.8. In the other direction, the objective doesn't clip, so mistakes get corrected. Clipping also isn't a hard limit. It removes the incentive to go further, but the ratio can still drift past the bounds.
- **Point 14:** Yes, but "random" means *high entropy*, not uniformly random behavior. The bonus is only a gentle push.

### Follow-up questions

**What is collected in step 1?**
For each step: the state s, action a, reward r and done flag. You also store two things computed at collection time: **log π_old(a|s)**, the log-probability the policy assigned to the chosen action, and **V(s)**, the critic's estimate. You need the first to compute the ratio later and the second for advantages. Often this is gathered from several environments in parallel, e.g. 8 environments × 256 steps = 2048.

**What is π()?**
It's the policy: π(a|s) is the probability of taking action a in state s. π_θ means the policy network with weights θ.

- Discrete actions: the softmax output at the index of a.
- Continuous actions: the Gaussian probability density at a.

π_old is the policy that collected the batch (frozen), and π_new is the network as it's being updated. In code the ratio is computed from log-probabilities: `ratio = exp(logp_new − logp_old)`. At the very first update of a batch, π_new = π_old, so the ratio is exactly 1.

**What's the critic's input?**
Yes, just the state s. It outputs a single number V(s). Implementations differ on whether the actor and critic are separate networks or share early layers. SB3 uses separate networks by default.

**GAE equation**
First, the one-step "surprise":

> δ_t = r_t + γ · V(s_{t+1}) − V(s_t)

This is the Bellman error from earlier: the gap between what the critic predicted for s_t and the one-step Bellman target. Then GAE sums these surprises with decaying weights:

> A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + …

It's computed backward through the batch: `A_t = δ_t + γλ · (1 − done_t) · A_{t+1}`. The `done` flag stops advantages leaking across episode boundaries, and V(s_{t+1}) is treated as 0 if the episode ended.

The parameter λ sets the trade-off between bias and variance:

- **λ = 0:** A_t = δ_t. It uses one step and the critic's guess, so low variance but biased when the critic is wrong.
- **λ = 1:** the full discounted return minus V(s_t). It's unbiased but noisy.
- **Typical:** γ = 0.99, λ = 0.95. Advantages are usually normalized (mean 0, std 1) per batch.

**How many epochs?**
Typically 3 to 10. The PPO paper used 10 for the continuous-control experiments and 3 for Atari, and SB3's default is 10. More epochs squeeze more out of each batch, but the policy drifts further from π_old, and the clipping only partly guards against that.

**How long does the loop repeat?**
You choose a total training budget in environment steps, and it runs until then. With 2048 steps per iteration and 1 million total steps, that's about 488 iterations. There's no automatic convergence stop in the basic algorithm. People watch the evaluation curve and pick a budget where it plateaus. Highway-env tasks usually use 10⁵ to 10⁶ steps, though this depends on your environment. Some implementations add an early-stop rule: end the epochs for a batch if the policy's KL divergence from π_old gets too large.

**Value loss**
It's a regression. First compute a target return for each step: `R_t = A_t + V_old(s_t)`, using the advantages and the values stored at collection time. Then:

> L_V = mean( (V_θ(s_t) − R_t)² )

This is the same pattern as DQN: the critic is trained toward a Bellman-style target that is held fixed during the update. Some implementations also clip the value update.

**Entropy bonus**
Entropy measures how spread out the policy is:

> H(π(·|s)) = − Σ_a π(a|s) · log π(a|s)

For five actions:

- Uniform [0.2 ×5]: H = ln 5 ≈ 1.61, the maximum.
- [0.6, 0.2, 0.1, 0.05, 0.05]: H ≈ 1.16.
- A fully deterministic policy [1, 0, 0, 0, 0]: H = 0.

For continuous Gaussian policies, H = ½ ln(2πe σ²) per action dimension, so a larger standard deviation means higher entropy. It's averaged over the batch and *subtracted* from the loss, so the optimizer is rewarded for staying spread out.

### Putting the loss together

PPO combines the three terms into one loss:

```python
ratio = torch.exp(logp_new - logp_old)
policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
value_loss = ((values - returns) ** 2).mean()
entropy = new_dist.entropy().mean()

loss = policy_loss + 0.5 * value_loss - 0.01 * entropy   # coefficients vary
```

The coefficients are hyperparameters. 0.5 is a common weight for the value loss, and the entropy coefficient is usually between 0 and 0.01. SB3 defaults to 0.

Your summary now covers the full algorithm. If you want to test your understanding, try computing δ and A for a short three-step corridor episode by hand.

For the next step, I can explain SAC, the off-policy actor-critic and the other major family you'll encounter. Or we could return to your project and map PPO, DQN and SAC onto your 5-model choice and the metrics you'll report.

## Summary 10 PPO 2


PPO:
 1) There are two phases:
    1) The first phase consits of two steps: step 1 collect a batch e.g. 2048 steps with the current policy  
       1) The first policy initializes the likelihood of all actions to the same probability
       2) Collect state s, action a, reward r, the done flag, log(pi_old(a|s)) for the ratio, and the critics estimate for that sample V(s) for advantages
       3) This collection is often over many environments, e.g. 8 environmnts * 256 steps = 20458
    2) The second step, compute the advantage A, which is how good the action was compared to what was expected, across each step. 
       1) To compute A, you need a critic and GAE
       2) The critic is a neural network that inputs the state s outputs the expected return of being in state s, V(s). 
          1) Sometimes the critic shares the same early layers as the actor network, but many times the actor and critic are completely separate networks
       3) For GAE, first compute delta_t = r_t + gamma * V(s_{t+1}) - V(s_t). This is the Bellman error
          1) Next GAE computes the the advantage A_t = delta_t + (gamma*lambda)\*delta_{t+1} + (gamma\*delta)^2\*delta_{t+2}...
          2) This previous equation is computed backward through the batch as A_t = delta_t + gamma*lambda * (1-done_t) + A_{t+1}
          3) The done_t flag stops advantages from leaking across episodic boundaries
          4) lambda sets the trade-off between bias and variance. 
             1) lambda = 0, A_t = delta_t, low variance but biased if critic is wrong
             2) lambda = 1, is full discounted return, which is unbiased but noisy
          5) Typically gamma=0.99, lambda=0.95, and advantages are normalized (mean 0, std 1) per batch
    3) The second phase consists of two steps: the first step: repeat for some predefined set of epochs (usually 3-10):
       1) Divide the results of phase 1 into mini batches and on each mini batch update the model on the with clipped policy loss, value loss, and a small entropy bonus
          1) To compute the clipped policy loss 
             1) first compute the ratio = pi_new(a|s) / pi_old(a|s)
               1) pi(a|s) is the policy, the probability of taking action a in state s. the theta in pi_theta(a|s) (e.g. old or new) refers to the weights of the policy network. 
               2) This is often done with log-probabilities, exp(logp_new - logp_old)
             2) Next, compute clipped policy loss = -min(ratio x A, clip(ratio, 1-epsilon, 1+epsilon) x A)
               1) epsilon is typically set to 0.2
               2) The clipping is to keep the updated small, which prevents spiraling and training collapse, leading to model instability
                  1) For example, the ratio stops updating a good action (A > 0) once the ratio is above 1.2, and stops updating a bad action (A < 0), when the update  is below 0.8.  
          3) Calculate value loss with a regression
             1) First calculate target return for a step, R_T = A_t + V_old(s_t) using advantages and V_old (which is stored at collection time)
             2) Then calculate L_V = mean( (V_theta(s-t) - R_t)^2 ), where V_theta is the return value with the current weights of the critic network
          4) The entropy bonus gently rewards the policy for high entropy, which promotes exploration
             1) The entropy bonus is calculated with H(pi(.|s)) = -Sigma_a[ pi(a|s) * log(pi(a|s)) ]. 
             2) Entropy bonus is max when the distribution of action probabilities is uniform, and decreases when it becomes more deterministic (e.g. chooses the same action every time)
             3) For policies that are continues and Gaussian, entropy H is calculated as (1/2)ln(2(pi)(3)*sigma^2) per action dimension (more std = higher entropy). It is averaged over the batch and subtracted from the loss, the optimizer is rewarded for staying spread out. 
          5) loss is calculated with something like policy_loss + 0.5*value_loss - 0.01\*entropy
             1) Those coefficients are hyperparameters and flexible. The current values are common.  
      4) The fourth step, throw the batch away, and repeat at step 1 until reached e.g. a max environment steps (e.g. 1 million/10^6 steps)
        1) (Some algorithms implement an early stop where the loop ends for a batch when the policy's KL divergence from pi_old gets too large)


## Prompt 11 PPO 3

PPO:
 1) There are two phases:
    1) The first phase consits of two steps: step 1 collect a batch e.g. 2048 steps with the current policy  
       1) The first policy initializes the likelihood of all actions to the same probability
       2) Collect state s, action a, reward r, the done flag, log(pi_old(a|s)) for the ratio, and the critics estimate for that sample V(s) for advantages
       3) This collection is often over many environments, e.g. 8 environmnts * 256 steps = 20458
    2) The second step, compute the advantage A, which is how good the action was compared to what was expected, across each step. 
       1) To compute A, you need a critic and GAE
       2) The critic is a neural network that inputs the state s outputs the expected return of being in state s, V(s). 
          1) Sometimes the critic shares the same early layers as the actor network, but many times the actor and critic are completely separate networks
       3) For GAE, first compute delta_t = r_t + gamma * V(s_{t+1}) - V(s_t). This is the Bellman error
          1) Next GAE computes the the advantage A_t = delta_t + (gamma*lambda)\*delta_{t+1} + (gamma\*delta)^2\*delta_{t+2}...
          2) This previous equation is computed backward through the batch as A_t = delta_t + gamma*lambda * (1-done_t) + A_{t+1}
          3) The done_t flag stops advantages from leaking across episodic boundaries
          4) lambda sets the trade-off between bias and variance. 
             1) lambda = 0, A_t = delta_t, low variance but biased if critic is wrong
             2) lambda = 1, is full discounted return, which is unbiased but noisy
          5) Typically gamma=0.99, lambda=0.95, and advantages are normalized (mean 0, std 1) per batch
    3) The second phase consists of two steps: the first step: repeat for some predefined set of epochs (usually 3-10):
       1) Divide the results of phase 1 into mini batches and on each mini batch update the model on the with clipped policy loss, value loss, and a small entropy bonus
          1) To compute the clipped policy loss 
             1) first compute the ratio = pi_new(a|s) / pi_old(a|s)
               1) pi(a|s) is the policy, the probability of taking action a in state s. the theta in pi_theta(a|s) (e.g. old or new) refers to the weights of the policy network. 
               2) This is often done with log-probabilities, exp(logp_new - logp_old)
             2) Next, compute clipped policy loss = -min(ratio x A, clip(ratio, 1-epsilon, 1+epsilon) x A)
               1) epsilon is typically set to 0.2
               2) The clipping is to keep the updated small, which prevents spiraling and training collapse, leading to model instability
                  1) For example, the ratio stops updating a good action (A > 0) once the ratio is above 1.2, and stops updating a bad action (A < 0), when the update  is below 0.8.  
          3) Calculate value loss with a regression
             1) First calculate target return for a step, R_T = A_t + V_old(s_t) using advantages and V_old (which is stored at collection time)
             2) Then calculate L_V = mean( (V_theta(s-t) - R_t)^2 ), where V_theta is the return value with the current weights of the critic network
          4) The entropy bonus gently rewards the policy for high entropy, which promotes exploration
             1) The entropy bonus is calculated with H(pi(.|s)) = -Sigma_a[ pi(a|s) * log(pi(a|s)) ]. 
             2) Entropy bonus is max when the distribution of action probabilities is uniform, and decreases when it becomes more deterministic (e.g. chooses the same action every time)
             3) For policies that are continues and Gaussian, entropy H is calculated as (1/2)ln(2(pi)(3)*sigma^2) per action dimension (more std = higher entropy). It is averaged over the batch and subtracted from the loss, the optimizer is rewarded for staying spread out. 
          5) loss is calculated with something like policy_loss + 0.5*value_loss - 0.01\*entropy
             1) Those coefficients are hyperparameters and flexible. The current values are common.  
      4) The fourth step, throw the batch away, and repeat at step 1 until reached e.g. a max environment steps (e.g. 1 million/10^6 steps)
        1) (Some algorithms implement an early stop where the loop ends for a batch when the policy's KL divergence from pi_old gets too large)

Follow up: 
 - For clarification, what is a step? A full episode? Just one action/transition? Or until the model stops?
 - what is the exp() function?
 - what are log probabilities? How do they relate to non-log probabilitiesS
 - why are log probabilities used for calculating things like the ratio?
 - What do you mean by one-step Bellman target?
 - I feel like i'm missing another network in my explanation. Is there an actor network somewhere? where does it fit in? how/where is the loss applied?
 - what is delta_t? Is that just the Bellman error? It matches the simpler equation of advantage computed earlier
 - what is the _t? Does that refer to the value of something at each step?

## Response 11 PPO 3


Your notes are very close to complete. I'll go through the fixes first, then your questions.

### Fixes to your notes

1. **8 environments × 256 steps = 2048**, not 20458.
2. **GAE expansion:** the third term should be (γ·**λ**)²·δ_{t+2}. You wrote (γ·δ)².
3. **Backward recursion:** the pieces are multiplied, not added: `A_t = δ_t + γ·λ·(1 − done_t)·A_{t+1}`. You wrote `+` between the last two terms.
4. **Gaussian entropy:** it's ½ ln(2π**e**σ²), where e ≈ 2.718 is Euler's number, not 3.
5. **A detail about value targets:** compute R_t = A_t + V_old(s_t) from the *raw* advantages, before normalizing them. Normalization is only applied to the advantages used in the policy loss.
6. **Minor:** reshuffle the batch into mini-batches at the start of each epoch.

Everything else in your notes is correct.

### 1) What is a "step"?

A step is **one transition**: the agent sees state s_t, takes one action a_t, and the environment returns reward r_t and the next state. An episode is a sequence of many steps, from `reset()` until a crash, a success or a time limit.

A batch of 2048 steps doesn't line up with episode boundaries. It may contain several complete episodes plus a partial one at the end. For the partial one, the `done` flag is 0 at the cut-off, and the critic's V(s_{t+1}) is used to bootstrap beyond it. That's why GAE uses the critic's estimate of the final state instead of treating it as the end.

In highway-env, one step is one decision, and the default is one decision per second of simulated time. The built-in `merge-v0` runs about 40 steps per episode.

### 2) What is exp()?

`exp(x)` means eˣ, where e ≈ 2.718. Some values: exp(0) = 1, exp(0.18) ≈ 1.2, exp(−0.69) ≈ 0.5. It's the inverse of the natural logarithm.

### 3) What are log probabilities?

The natural log of the probability: **log p = ln(p)**. Since p is between 0 and 1, log p is always zero or negative.

| p | log p |
|---|---|
| 1.0 | 0 |
| 0.6 | −0.51 |
| 0.5 | −0.69 |
| 0.1 | −2.30 |
| 0.01 | −4.61 |

A higher probability gives a log closer to 0, and a tiny probability gives a very negative number. You can always recover p with `exp(log p)`.

### 4) Why use logs for the ratio?

1. **The ratio becomes a subtraction.** π_new/π_old = exp(log π_new − log π_old). Take π_old = 0.50 and π_new = 0.60: the logs are −0.693 and −0.511, the difference is 0.182, and exp(0.182) = 1.2. Same answer.
2. **Numerical stability.** Probabilities can be extremely small. A continuous policy with several action dimensions, or a long sequence, multiplies many numbers below 1, which can underflow to 0 in floating point. Taking logs turns products into sums (log(a·b) = log a + log b), which stays well behaved.
3. **It's what the network naturally produces.** Softmax outputs are computed stably as log-softmax, and the log-probability is what the gradient of the policy loss needs anyway.

### 5) "One-step Bellman target"

The target is r_t + γ·V(s_{t+1}). It's "one-step" because it uses only **one real reward**, then replaces everything after that with the critic's estimate of the next state. A two-step version would use r_t + γ·r_{t+1} + γ²·V(s_{t+2}), and an infinite-step one would use real rewards until the episode ends (the full return). GAE blends all of these.

### 6) The missing network: the actor

You're right, it's missing. PPO has **two networks**:

| | Actor (policy) π_θ | Critic V_φ |
|---|---|---|
| Input | state s | state s |
| Output | action probabilities (or Gaussian mean and std) | one number, V(s) |
| Used for | choosing actions during collection; computing log π | computing advantages and the value loss |
| Trained by | clipped policy loss + entropy bonus | value loss |

The actor also determines the ratio, because π_new is the actor's current output, while π_old doesn't need a separate copy of the network. You saved log π_old(a|s) numbers at collection time, and that's all the ratio needs.

Where the loss is applied depends on the architecture. With separate networks, the policy and entropy terms only touch the actor's weights, and the value term only touches the critic's. With shared early layers, both contribute gradients to the shared layers, which is why the coefficient (0.5) matters. Only the actor is needed after training, since the critic is a training tool.

### 7) What is δ_t?

Yes, δ_t = r_t + γV(s_{t+1}) − V(s_t) is the Bellman error. It also matches the earlier simple advantage, because A(s, a) ≈ Q(s, a) − V(s), and Q(s, a) ≈ r + γV(s′) gives exactly δ. So δ_t is a **one-step advantage estimate** (what GAE gives with λ = 0), and GAE sums the discounted δ's from the following steps to get a smoother estimate.

### 8) What does _t mean?

It's the **time-step index**. r_t is the reward at step t, s_t is the state at step t, V(s_t) is the critic's estimate there, and A_t is the advantage of the action taken at step t. t+1 is the next step. In a batch from multiple environments, t runs along each environment's own sequence.

### A worked example (the corridor)

Take one episode A → B → C → Goal, with γ = 0.9, λ = 0.95 (so γλ = 0.855). The critic is currently poor: V(A) = 2, V(B) = 3, V(C) = 4. The true values are 8.1, 9 and 10.

**δ's:**
- δ₂ (at C, reaching the goal, done): 10 + 0 − 4 = **6**
- δ₁ (at B): 0 + 0.9 × 4 − 3 = **0.6**
- δ₀ (at A): 0 + 0.9 × 3 − 2 = **0.7**

**Advantages (backward):**
- A₂ = 6
- A₁ = 0.6 + 0.855 × 6 = **5.73**
- A₀ = 0.7 + 0.855 × 5.73 ≈ **5.60**

All are positive because the episode went much better than the critic expected, so the actor makes these actions more likely. The value targets R_t = A_t + V(s_t) are 7.6, 8.73 and 10, which are far closer to the true 8.1, 9 and 10 than the critic's guesses. The value loss pulls the critic toward them, so the next batch has smaller errors.

Next I could explain SAC, the off-policy actor-critic that rounds out the three classics, or write a compact PPO implementation in the same style as `dqn_from_scratch.py`. Which do you prefer?


## Summary 11 PPO 3

log probabilities = log p = ln(p)
used to turn division/multiplication into addition/subtraction and to make numbers better
typically probabilities are really small, so this makes them slightly more sizeable which prevents roundoff error
ln(p) is also always negative

PPO:
 1) There are two phases:
    1) The first phase consits of two steps: step 1 collect a batch e.g. 2048 steps with the current policy  
       1) The first policy initializes the likelihood of all actions to the same probability
       2) A step is one transition: you're in state s, you take action a, you get a reward r
       3) Collect state s, action a, reward r, the done flag, log(pi_old(a|s)) for the ratio, and the critics estimate for that sample V(s) for advantages
       4) This collection is often over many environments, e.g. 8 environmnts * 256 steps = 2048
    2) The second step, compute the advantage A, which is how good the action was compared to what was expected, across each step. 
       1) To compute A, you need a critic and GAE, it's used to compute the loss where really good improvements are rewarded and bad changes aren't
       2) A < 0 is an action was worse than expected, and A > 0 if it's better than expected 
       3) The critic is a neural network that inputs the state s outputs the expected return of being in state s, V(s). 
          1) Sometimes the critic shares the same early layers as the actor network, but many times the actor and critic are completely separate networks
          2) V(s) was collected in step 1
       4) GAE
          1) first compute delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), for each step. delta_t is the Bellman error, or one-step advantage estimate. The final step delta_final will just be V(s_final)
             1) _t refers to a step, s_t is the state at step t
             2) r_t + gamma * V(s_{t+1}) is the one-step Bellman target, it is the target, V(s_t) is the prediction, it is one-step ahead of s_t, delta_t = target - prediction
          2) Next, GAE computes the sum of discounted advantages A_t = delta_t + (gamma*lambda)\*delta_{t+1} + (gamma\*lambda)^2\*delta_{t+2}...
             1) This previous equation is computed backward through the batch with the recursive equation A_t = delta_t + gamma*lambda * (1-done_t) * A_{t+1}
                1) Where A_final = delta_final
             2) The done_t flag stops advantages from leaking across episodic boundaries
             3) lambda sets the trade-off between bias and variance. 
                1) lambda = 0, A_t = delta_t, low variance but biased if critic is wrong
                2) lambda = 1, is full discounted return, which is unbiased but noisy
             4) Typically gamma=0.99, lambda=0.95, and advantages are normalized (mean 0, std 1) per batch
    3) The second phase consists of two steps: the first step: repeat for some predefined set of epochs (usually 3-10):
       1) Divide the results of phase 1 into mini batches and on each mini batch update the actor and critic networks with clipped policy loss, value loss, and a small entropy bonus
          1) To compute the clipped policy loss 
             1) first compute the ratio = pi_new(a|s) / pi_old(a|s), < 1 if an action probability decreased, > 1 if increased 
               1) pi(a|s) is the policy, the probability of taking action a in state s. the theta in pi_theta(a|s) (e.g. old or new) refers to the weights of the policy network. 
               2) pi_new(a|s) is calculated with the actor network, which inputs state and outputs action probabibilities (discrete) or distribution (continuous) (gaussian mean/std)
               3) This is often done with log-probabilities, exp(ln p_new - ln p_old)
                  1) exp(x) = e^x
             2) Next, compute clipped policy loss = -min(ratio x A, clip(ratio, 1-epsilon, 1+epsilon) x A)
               1) A is normalized, < 0 if worse than expected, > 0 if better than expected
               2) epsilon is typically set to 0.2
               3) The clipping is to keep the updated small, which prevents spiraling and training collapse, leading to model instability
                  1) For example, the ratio stops updating a good action (A > 0) once the ratio is above 1.2, and stops updating a bad action (A < 0), when the update  is below 0.8.  
          2) Calculate value loss with a regression
             1) First calculate target return for a step, R_T = A_t + V_old(s_t) using raw, non normalized advantages and V_old (which is stored at collection time)
                1) Normalized advantages are only used in the policy loss
             2) Then calculate L_V = mean( (V_theta(s-t) - R_t)^2 ), where V_theta is the return value with the current weights of the critic network
          3) The entropy bonus gently rewards the policy for high entropy, which promotes exploration
             1) The entropy bonus is calculated with H(pi(.|s)) = -Sigma_a[ pi(a|s) * log(pi(a|s)) ]. 
                1) Entropy bonus is max when the distribution of action probabilities is uniform, and decreases when it becomes more deterministic (e.g. chooses the same action every time)
             2) For policies that are continues and Gaussian, entropy H is calculated as (1/2)ln(2(pi)(e)*sigma^2) per action dimension (more std = higher entropy). It is averaged over the batch and subtracted from the loss, the optimizer is rewarded for staying spread out. 
          4) Loss is calculated with something like policy_loss + 0.5*value_loss - 0.01\*entropy
             1) Those coefficients are hyperparameters and flexible. The current values are common. 
          5) But loss is applied such that policy_loss and entropy are applied to the actor network, and value_loss is applied to the critic network (if they don't have any shared layers).
          6) Reshuffle the batch into mini-batches at the end of each epoch  
      4) The fourth step, throw the batch away, and repeat at step 1 until reached e.g. a max environment steps (e.g. 1 million/10^6 steps)
        1) (Some algorithms implement an early stop where the loop ends for a batch when the policy's KL divergence from pi_old gets too large)

## Prompt 12 PPO 4

Still on PPO, can you check this final iteration of my notes, and address some final questions, before we move on? I want a solid foundation of PPO, so I can understand the papers that I'll have to read later on.

PPO:
 1) There are two phases:
    1) The first phase consits of two steps: step 1 collect a batch e.g. 2048 steps with the current policy  
       1) The first policy initializes the likelihood of all actions to the same probability
       2) A step is one transition: you're in state s, you take action a, you get a reward r
       3) Collect state s, action a, reward r, the done flag, log(pi_old(a|s)) for the ratio, and the critics estimate for that sample V(s) for advantages
       4) This collection is often over many environments, e.g. 8 environmnts * 256 steps = 2048
    2) The second step, compute the advantage A, which is how good the action was compared to what was expected, across each step. 
       1) To compute A, you need a critic and GAE, it's used to compute the loss where really good improvements are rewarded and bad changes aren't
       2) A < 0 is an action was worse than expected, and A > 0 if it's better than expected 
       3) The critic is a neural network that inputs the state s outputs the expected return of being in state s, V(s). 
          1) Sometimes the critic shares the same early layers as the actor network, but many times the actor and critic are completely separate networks
          2) V(s) was collected in step 1
       4) GAE
          1) first compute delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), for each step. delta_t is the Bellman error, or one-step advantage estimate. The final step delta_final will just be V(s_final)
             1) _t refers to a step, s_t is the state at step t
             2) r_t + gamma * V(s_{t+1}) is the one-step Bellman target, it is the target, V(s_t) is the prediction, it is one-step ahead of s_t, delta_t = target - prediction
          2) Next, GAE computes the sum of discounted advantages A_t = delta_t + (gamma*lambda)\*delta_{t+1} + (gamma\*lambda)^2\*delta_{t+2}...
             1) This previous equation is computed backward through the batch with the recursive equation A_t = delta_t + gamma*lambda * (1-done_t) * A_{t+1}
                1) Where A_final = delta_final
             2) The done_t flag stops advantages from leaking across episodic boundaries
             3) lambda sets the trade-off between bias and variance. 
                1) lambda = 0, A_t = delta_t, low variance but biased if critic is wrong
                2) lambda = 1, is full discounted return, which is unbiased but noisy
             4) Typically gamma=0.99, lambda=0.95, and advantages are normalized (mean 0, std 1) per batch
    3) The second phase consists of two steps: the first step: repeat for some predefined set of epochs (usually 3-10):
       1) Divide the results of phase 1 into mini batches and on each mini batch update the actor and critic networks with clipped policy loss, value loss, and a small entropy bonus
          1) To compute the clipped policy loss 
             1) first compute the ratio = pi_new(a|s) / pi_old(a|s), < 1 if an action probability decreased, > 1 if increased 
               1) pi(a|s) is the policy, the probability of taking action a in state s. the theta in pi_theta(a|s) (e.g. old or new) refers to the weights of the policy network. 
               2) pi_new(a|s) is calculated with the actor network, which inputs state and outputs action probabibilities (discrete) or distribution (continuous) (gaussian mean/std)
               3) This is often done with log-probabilities, exp(ln p_new - ln p_old)
                  1) exp(x) = e^x
             2) Next, compute clipped policy loss = -min(ratio x A, clip(ratio, 1-epsilon, 1+epsilon) x A)
               1) A is normalized, < 0 if worse than expected, > 0 if better than expected
               2) epsilon is typically set to 0.2
               3) The clipping is to keep the updated small, which prevents spiraling and training collapse, leading to model instability
                  1) For example, the ratio stops updating a good action (A > 0) once the ratio is above 1.2, and stops updating a bad action (A < 0), when the update  is below 0.8.  
          2) Calculate value loss with a regression
             1) First calculate target return for a step, R_T = A_t + V_old(s_t) using raw, non normalized advantages and V_old (which is stored at collection time)
                1) Normalized advantages are only used in the policy loss
             2) Then calculate L_V = mean( (V_theta(s-t) - R_t)^2 ), where V_theta is the return value with the current weights of the critic network
          3) The entropy bonus gently rewards the policy for high entropy, which promotes exploration
             1) The entropy bonus is calculated with H(pi(.|s)) = -Sigma_a[ pi(a|s) * log(pi(a|s)) ]. 
                1) Entropy bonus is max when the distribution of action probabilities is uniform, and decreases when it becomes more deterministic (e.g. chooses the same action every time)
             2) For policies that are continues and Gaussian, entropy H is calculated as (1/2)ln(2(pi)(e)*sigma^2) per action dimension (more std = higher entropy). It is averaged over the batch and subtracted from the loss, the optimizer is rewarded for staying spread out. 
          4) Loss is calculated with something like policy_loss + 0.5*value_loss - 0.01\*entropy
             1) Those coefficients are hyperparameters and flexible. The current values are common. 
          5) But loss is applied such that policy_loss and entropy are applied to the actor network, and value_loss is applied to the critic network (if they don't have any shared layers).
          6) Reshuffle the batch into mini-batches at the end of each epoch  
      4) The fourth step, throw the batch away, and repeat at step 1 until reached e.g. a max environment steps (e.g. 1 million/10^6 steps)
        1) (Some algorithms implement an early stop where the loop ends for a batch when the policy's KL divergence from pi_old gets too large)

Follow up: 
 - in phase 2, are you applying the same advantage value from the GAE to every policy loss computation? The advantage doesn't change at all across epochs or mini batches? 
 - the policy pi_new is calculated before the actor gets updated, and pi_old is calculated after the actor gets updated, so aren't they always the same, per this flow? Or wait, pi_new is updated with each epoch, whereas pi_old always refers to the weights used during the first collection phase. 
 - can you describe the outputs of clip()? I'm struggling to visualize how it works, e.g. a range of inputs and their outputs
 - the policy loss is computed separately for each action yes?


## Response 12 PPO 4

Your notes are close to final. There are a few fixes and then your four questions.

### Corrections

1. **The final-step delta (GAE, 1) is wrong.** It isn't just V(s_final). The general form is δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t). For the last step in the batch there are two cases:
   - **The episode ended** (done = 1): δ = r_t − V(s_t). The future term is 0.
   - **The batch was cut off mid-episode** (done = 0): δ = r_t + γ·V(s_{t+1}) − V(s_t), where s_{t+1} is the state after the last step. You need one extra critic evaluation on that state. This is called bootstrapping.

   Your A_final = δ_final is right, since nothing follows it.
2. **Point 2.1 (the role of A) needs rewording.** A doesn't reward "improvements." It's a signed weight on each action's probability in the policy loss. A > 0 pushes the action's probability up, A < 0 pushes it down, and the size of A sets how hard. Clipping then limits how far the probability can move.
3. **GAE point 2:** the sum is of discounted **δ's** (one-step advantage estimates), not "advantages."
4. **Clipping point 3.1.1:** the ratio stops being rewarded above 1.2 for good actions and below 0.8 for bad ones. Your wording "when the update is below 0.8" should say "ratio."
5. **Minor:** "all actions equal" at initialization is only *approximately* true for a randomly initialized network. Also R_T should be R_t.

Everything else is correct, including the Gaussian entropy, the raw-versus-normalized advantage point, and which network each loss term trains.

### 1) Does A stay fixed across epochs and mini-batches?

Yes. After collection, advantages are computed **once**, using the critic's values stored at collection time (V_old). They and the value targets R_t = A_t + V_old(s_t) are treated as constants for all epochs and mini-batches of that iteration.

The reason is that the advantage measures how good the action was relative to the data-collection policy, and that's a property of the collected data. Recomputing it with a changing critic would make the target shift during the update. (Normalization is the one exception: some implementations, like SB3, normalize advantages per mini-batch instead of per batch, but the underlying values are still fixed.)

### 2) π_new versus π_old

Your second realization is right, and the first version in your question is mixed up. π_old is the actor's weights at the moment of collection, and π_new is the actor's weights *as it's being updated*.

- Nothing like "π_old computed after the update" exists. You saved log π_old(a|s) as numbers at collection time and never recompute it.
- π_new changes after **every mini-batch gradient step**, not just every epoch. So in the first mini-batch of the first epoch, π_new = π_old and every ratio is exactly 1. As updates accumulate, the ratios drift away from 1, and that's where clipping starts to matter.
- After the epochs finish, the updated actor is used to collect the next batch, so it becomes the new "old" policy.

### 3) What clip() does

`clip(x, lo, hi)` returns x if it lies in [lo, hi], and otherwise returns the nearest boundary. With ε = 0.2, the range is [0.8, 1.2]:

| ratio | 0.5 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.5 |
|---|---|---|---|---|---|---|---|
| clip(ratio) | 0.8 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.2 |

It's the min with the unclipped term that makes the behavior depend on the sign of A. Here is the *objective* min(ratio·A, clip·A) (the loss is its negative):

**Good action, A = +2**

| ratio | ratio·A | clip·A | min | what happens |
|---|---|---|---|---|
| 0.5 | 1.0 | 1.6 | **1.0** | unclipped: gradient flows, so the probability is pushed back up |
| 1.0 | 2.0 | 2.0 | **2.0** | normal |
| 1.2 | 2.4 | 2.4 | **2.4** | at the boundary |
| 1.5 | 3.0 | 2.4 | **2.4** | clipped: flat, so no more incentive to raise it |

**Bad action, A = −2**

| ratio | ratio·A | clip·A | min | what happens |
|---|---|---|---|---|
| 0.5 | −1.0 | −1.6 | **−1.6** | clipped: flat, so no more incentive to lower it |
| 0.8 | −1.6 | −1.6 | **−1.6** | at the boundary |
| 1.0 | −2.0 | −2.0 | **−2.0** | normal |
| 1.5 | −3.0 | −2.4 | **−3.0** | unclipped: gradient flows, so the probability is pushed back down |

The pattern: the objective goes flat in the direction where the action has already moved enough, and stays sloped where the policy moved the wrong way. So the policy can't overshoot in the helpful direction, but errors still get corrected. Where the objective is flat, the gradient is zero.

### 4) Is the policy loss computed separately per action?

It's computed per *sample* (one transition), then averaged. Each step in the batch contributes one term, using only the action **that was actually taken** at that step: its ratio, its advantage A_t and its clipped objective. The probabilities of the other actions don't appear in that term. They're affected indirectly, since the probabilities must sum to 1. The loss over the mini-batch is the mean of these per-sample terms, and one backward pass through π_new gives the gradient.

You now have enough to read the PPO paper itself. Clipping and GAE are the two parts that take effort. If you read the paper next, it would be useful to read what the 2026 PPO variants change. After that, I can explain SAC, or write a compact PPO implementation in the style of `dqn_from_scratch.py`. Which would you prefer?


## Summary 12 PPO 4

PPO:
 1) There are two phases:
    1) The first phase consits of two steps: step 1 collect a batch e.g. 2048 steps with the current policy  
       1) The first policy initializes the likelihood of all actions to the same probability
       2) A step is one transition: you're in state s, you take action a, you get a reward r
       3) Collect state s, action a, reward r, the done flag, log(pi_old(a|s)) for the ratio, and the critics estimate for that sample V(s) for advantages
       4) This collection is often over many environments, e.g. 8 environmnts * 256 steps = 2048
    2) The second step, compute the advantage A, which is how good the action was compared to what was expected, across each step. 
       1) To compute A, you need a critic and GAE, it's used to compute the loss where > 0 pushes the probability up, < 0 pushes the probability down, and |A| determines how much
       2) A < 0 is an action was worse than expected, and A > 0 if it's better than expected.
       3) Advantage is computed for each action, maybe
       4) The critic is a neural network that inputs the state s outputs the expected return of being in state s, V(s). 
          1) Sometimes the critic shares the same early layers as the actor network, but many times the actor and critic are completely separate networks
          2) V(s) was collected in step 1
       5) GAE
          1) first compute delta_t = r_t + gamma * V(s_{t+1})(1-done_t) - V(s_t), for each step. delta_t is the Bellman error, or one-step advantage estimate. If the final state ended in the middle of an episode, you have to bootstrap V(s_{t+1})
             1) _t refers to a step, s_t is the state at step t
             2) r_t + gamma * V(s_{t+1}) is the one-step Bellman target, it is the target, V(s_t) is the prediction, it is one-step ahead of s_t, delta_t = target - prediction
          2) Next, GAE computes the sum of discounted one-step advantages estimates (deltas) A_t = delta_t + (gamma*lambda)\*delta_{t+1} + (gamma\*lambda)^2\*delta_{t+2}...
             1) This previous equation is computed backward through the batch with the recursive equation A_t = delta_t + gamma*lambda * (1-done_t) * A_{t+1}
                1) Where A_final = delta_final
             2) The done_t flag stops advantages from leaking across episodic boundaries
             3) lambda sets the trade-off between bias and variance. 
                1) lambda = 0, A_t = delta_t, low variance but biased if critic is wrong
                2) lambda = 1, is full discounted return, which is unbiased but noisy
             4) Typically gamma=0.99, lambda=0.95, and advantages are normalized (mean 0, std 1) per batch
    3) The second phase consists of two steps: the first step: repeat for some predefined set of epochs (usually 3-10):
       1) Divide the results of phase 1 into mini batches and on each mini batch update the actor and critic networks with clipped policy loss, value loss, and a small entropy bonus
          1) To compute the clipped policy loss 
             1) first compute the ratio = pi_new(a|s) / pi_old(a|s), < 1 if an action probability decreased, > 1 if increased 
               1) pi(a|s) is the policy, the probability of taking action a in state s. the theta in pi_theta(a|s) (e.g. old or new) refers to the weights of the policy network. 
               2) pi_new(a|s) is calculated with the actor network, which inputs state and outputs action probabibilities (discrete) or distribution (continuous) (gaussian mean/std)
               3) This is often done with log-probabilities, exp(ln p_new - ln p_old)
                  1) exp(x) = e^x
             2) Next, compute clipped policy loss = -min(ratio x A, clip(ratio, 1-epsilon, 1+epsilon) x A)
               1) A is normalized, < 0 if worse than expected, > 0 if better than expected
               2) epsilon is typically set to 0.2
               3) The clipping is to keep the updated small, which prevents spiraling and training collapse, leading to model instability
                  1) For example, the ratio stops updating a good action (A > 0) once the ratio is above 1.2, and stops updating a bad action (A < 0), when the ratio is below 0.8.
              3) Policy loss is computed per sample (transition), then averaged
          2) Calculate value loss with a regression
             1) First calculate target return for a step, R_t = A_t + V_old(s_t) using raw, non normalized advantages and V_old (which is stored at collection time)
                1) Normalized advantages are only used in the policy loss
             2) Then calculate L_V = mean( (V_theta(s-t) - R_t)^2 ), where V_theta is the return value with the current weights of the critic network
          3) The entropy bonus gently rewards the policy for high entropy, which promotes exploration
             1) The entropy bonus is calculated with H(pi(.|s)) = -Sigma_a[ pi(a|s) * log(pi(a|s)) ]. 
                1) Entropy bonus is max when the distribution of action probabilities is uniform, and decreases when it becomes more deterministic (e.g. chooses the same action every time)
             2) For policies that are continues and Gaussian, entropy H is calculated as (1/2)ln(2(pi)(e)*sigma^2) per action dimension (more std = higher entropy). It is averaged over the batch and subtracted from the loss, the optimizer is rewarded for staying spread out. 
          4) Loss is calculated with something like policy_loss + 0.5*value_loss - 0.01\*entropy
             1) Those coefficients are hyperparameters and flexible. The current values are common. 
          5) But loss is applied such that policy_loss and entropy are applied to the actor network, and value_loss is applied to the critic network (if they don't have any shared layers).
          6) Reshuffle the batch into mini-batches at the end of each epoch  
      4) The fourth step, throw the batch away, and repeat at step 1 until reached e.g. a max environment steps (e.g. 1 million/10^6 steps)
        1) (Some algorithms implement an early stop where the loop ends for a batch when the policy's KL divergence from pi_old gets too large)

next, read the fucking paper at this point. it's right. 
fair enough. i think we have a general idea of how it works. 

next we need...

## Prompt 13



## Response 13

## Summary 13