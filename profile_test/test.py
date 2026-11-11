"""
connor
also from claude, this is to test if going hard left fails i think
"""

import numpy as np
from profile_test_environment import make_env
env = make_env()
for name, act in {"straight": [0.3, 0.0], "hard left": [0.3, 1.0]}.items():
    env.reset(seed=0); total, done = 0.0, False
    while not done:
        _, r, term, trunc, _ = env.step(np.array(act, np.float32)); total += r; done = term or trunc
    print(name, round(total, 1), env.outcome())