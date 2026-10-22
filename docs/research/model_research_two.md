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

## Prompt 1

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


## Response 1

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

## Summary 1 and follow-up questions
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


## Prompt 2

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


## Response 2

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

## Summary 2

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
 

## Prompt 3

Those questions might've been too loaded, a lot of it went over my head. I'll start with trying to understand RL/DRL.

To try to clarify, a state is what the agent is in, the action is what the agent can do in that state, and the reward is how good the new state is (e.g. V(s')). The agent wants to be in the state with the highest reward (the highest V). Q(s,a) is what the new state that the action will lead the agent to, and the value of that new state? And the policy is the pattern of actions to take on the state that lead to the highest value? 

Could you also give one or two simple examples?


## Response 3

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

## Summary 3 and Prompt 4
  
My Understanding:
 - Okay, reward is from the environment and its what we specify. It's tied to specific states? So if I'm in a car on a typical, non-problematic state, take an action (turn left) and crash (new state), because I crashed I get a reward of -10. We - the designers of the environment - determine what those rewards are
 - Value is the current reward of that state, or if you followed a series of optimal steps (an optimal policy) to get to that reward? In the corridor example, V(C) is 10 because the action there that has a reward of 10. It's the maximum of the values of the actions at those states, in addition to the decremented (by gamma) value of the optimal actions of future states. It seems quite difficult to calculate. 
 - Q(s,a) returns gamma * V(s'). Q(B, right) returns 0.9 * V(C) = 9, Q(B, left) returns 0.9 * V(A)

 Questions: 
 - What does a policy look like? 
 - What does the learning process look like from beginning to end? What does the environment look like? the model presumably starts from nothing, how does it learn a policy, V(s), etc.?

## Response 4

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

 ## Summary 4 and Prompt 5

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

## Response 5

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

So, I think we can try to start understanding DQN? and ask what Bellman's equation is 


## Prompt 6

Okay, I think I get what simple RL generally looks like. can you explain what Bellman's equation is? and what DQN is at a high level? Also with a simple example. 

## Response 6

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

## Summary 6

Explain Bellman's equation and DQN