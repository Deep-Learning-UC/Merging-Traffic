

merging traffic

Autonomous Highway Merging	A model that is trained to navigate a highway simulation where the car must learn to change lanes and merge into high-speed traffic without colliding.	

1) This idea would use deep reinforcement learning so their is no static dataset. Instead the model learns through making mistakes. The python library highway-env can be used to simulate traffic conditions.
2) related research paper:  https://arxiv.org/abs/1811.11329 (https://arxiv.org/pdf/1811.11329)

to be added

Project will likely be limited to a specific highway merge geometry rather than a full open-world environment to keep feasible. 



# ideas

a lot of the models in the documentation and code seem to be pretty good?
we could do a baseline of reinforcement learning models - that would be sufficient for the project
so we need a list of models 


# highway-env python library
 - overview page - https://pypi.org/project/highway-env/
 - several types of environments (merging, roundabout, highway, parking, etc.)
 - documentation - https://highway-env.farama.org/
 - example agents from
   - eleurent/rl-agents - https://github.com/eleurent/rl-agents
   - openai/baselines - https://github.com/openai/baselines
   - Stable Baselines3 - https://github.com/DLR-RM/stable-baselines3


# gymnasium python library
 - related to highway-env?
 - https://gymnasium.farama.org/
 - fork of OpenAI gym library for reinforcment learning problems
 - i would imagine highway-env is built on top of it
 



# research paper - Deep Reinforcement Learning for Autonomous Driving - Wang, Jia, Weng
 - https://arxiv.org/pdf/1811.11329
 - DDPG model / algorithm
 - TORCS (simulator)


# TORCS
 - https://torcs.sourceforge.net/
 - papers that use torcs - https://torcs.sourceforge.net/research/
 - racing simulator, it was used in the research paper above (DRLfAD)