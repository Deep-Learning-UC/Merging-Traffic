# entry point: python main.py [train|videos] [options], e.g. videos --model PPO
# also holds shared helpers
import importlib
import sys
from pathlib import Path

import numpy as np

from variables import MODEL, PERSONALITIES, VIDEO
from profile_test_environment import WRECK_COLOR

HERE = Path(__file__).resolve().parent
PANEL, TEXT, DIM, BLUE = (18, 22, 30), (235, 238, 245), (150, 158, 172), (90, 170, 255)
ENDING_COLORS = {"crashed": (240, 80, 80), "made exit": (80, 210, 120), "missed exit": (240, 190, 60), "timed out": DIM, "skipped": DIM}
ENDINGS = {"skipped": "SKIPPED", "crashed": "CRASHED", "made exit": "MADE THE EXIT", "missed exit": "MISSED THE EXIT"}
# a pileup can block every lane forever, so lives give up here
MAX_STEPS = 300

ENDING_COLORS |= {"off road": (240, 80, 80), "spun out": (240, 80, 80)}
ENDINGS |= {"off road": "WENT OFF ROAD", "spun out": "SPUN OUT"}



# color key for videos and watch.py
def key_items():
    return [("you", VIDEO["you_color"])] + [(k, p["color"]) for k, p in PERSONALITIES.items()] + [("wreck", WRECK_COLOR)]


# returns (outcome or "stuck", steps, total reward)
def play(env, obs, choose, each_step=None):
    done, steps, total = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        obs, reward, done, _, _ = env.step(choose(obs))
        steps, total = steps + 1, total + reward
        if each_step:
            each_step(env)
    return env.outcome() or "stuck", steps, total

# returns None if it cant load or was trained on a different obs shape
def load_model(env):
    try:
        algo = getattr(importlib.import_module(MODEL["library"]), MODEL["name"])
        model = algo.load((HERE / MODEL["path"]).resolve(), device="cpu")
    except Exception as e:
        print(f"couldnt load {MODEL['name']} from {MODEL['path']} ({e.__class__.__name__}: {e}), driving without it")
        return None
    if model.observation_space.shape != env.observation_space.shape:
        print(f"{MODEL['path']} was trained on a different input {model.observation_space.shape}, this road gives {env.observation_space.shape}. skipping it")
        return None
    return model

# returns (action, scores, kind). q values for DQN, probabilities for PPO style
def ask_model(model, obs):
    if model is None:
        return 1, None, ""
    import torch
    with torch.no_grad():
        if hasattr(model, "q_net"):
            scores, kind = model.q_net(model.q_net.obs_to_tensor(obs)[0])[0].numpy(), "q values"
        elif hasattr(model.policy, "get_distribution") and hasattr(model.action_space, "n"):
            scores, kind = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs[0].numpy(), "chances"
        else:
            # TODO model.predict() cannot be int for continuousaction
            # return int(model.predict(obs, deterministic=True)[0]), None, ""
            return model.predict(obs, deterministic=True)[0], None, ""
    return int(np.argmax(scores)), scores, kind


# imports inside so other files can import main without loading train/videos
def menu():
    args = sys.argv[1:]
    choice = args.pop(0) if args and args[0] in ("train", "videos") else None
    while choice is None:
        pick = input("1) train the models (then asks about the live window)\n2) test drive and make videos\npick 1 or 2: ").strip()
        choice = {"1": "train", "2": "videos"}.get(pick)
    if choice == "train":
        import train
        train.run(args)
    else:
        import videos
        videos.run(args)


if __name__ == "__main__":
    menu()
