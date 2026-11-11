# trains TRAIN["models"] in parallel, stops each early at goal_exit_rate.
# --ui / --no-ui skip the watcher prompt
import argparse
import importlib
import os
import pickle
import socket
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Process
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import variables as v
from profile_test_environment import make_env

HERE = Path(__file__).resolve().parent
LOG = HERE / v.TRAIN["folder"] / "training_log.txt"


# print and log, watch.py reads the log
def say(text):
    print(text, flush=True)
    with open(LOG, "a") as f:
        f.write(text + "\n")


# streams road, car frames, and endings to watch.py over udp. harmless if nothing is listening
class Broadcast(gym.Wrapper):
    def __init__(self, env, port, rank):
        super().__init__(env)
        self.sock, self.addr, self.rank = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), ("127.0.0.1", port), rank

    def send(self, *msg):
        try:
            self.sock.sendto(pickle.dumps((self.rank, *msg)), self.addr)
        except OSError:
            pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        env, self.score = self.env.unwrapped, 0.0
        lanes = [(*l.start, *l.end, l.width, int(l.line_types[0]), int(l.line_types[1]))
                 for f in env.road.network.graph for t in env.road.network.graph[f] for l in env.road.network.graph[f][t]]
        self.send("road", lanes, [tuple(o.position) for o in env.road.objects])
        step, me = env.road.step, env.vehicle

        def send_cars(dt):
            step(dt)
            cars = np.array([[c.position[0], c.position[1], c.heading, *(getattr(c, "color", None) or (255, 255, 255))]
                             for c in env.road.vehicles], np.float32)
            self.send("cars", cars, me.speed, env.split_x - me.position[0], env.exit_side)
        env.road.step = send_cars
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.score += result[1]
        if result[2] or result[3]:
            self.send("end", result[4]["outcome"] or "timed out", self.score)
        return result


# runs in a subprocess, so YOU has to be set here too
def training_env(rank, port, stream):
    v.YOU["driver"] = "model"
    env = make_env(max_seconds=v.TRAIN["max_life_seconds"])
    return Broadcast(env, port, rank) if stream else env


# logs every 1000 steps, saves checkpoints, stops at the goal. rates are over the last 100 lives
class Progress(BaseCallback):
    def __init__(self, name, total):
        super().__init__()
        self.name, self.total, self.next, self.next_save, self.lives = name, total, 1000, v.TRAIN["checkpoint_every"], 0

    def _on_training_start(self):
        self.start = time.time()

    def _on_step(self):
        self.lives += int(sum(self.locals["dones"]))
        if self.num_timesteps >= self.next_save:
            self.next_save += v.TRAIN["checkpoint_every"]
            self.model.save(HERE / v.TRAIN["folder"] / self.name)
        if self.num_timesteps < self.next:
            return True
        self.next += 1000
        lives = list(self.model.ep_info_buffer)
        n = max(len(lives), 1)
        exits = sum(e["outcome"] == "made exit" for e in lives) / n
        crashes = sum(e["outcome"] in ("crashed", "off road", "spun out") for e in lives) / n
        timeouts = sum(e["outcome"] is None for e in lives) / n
        rewards = [e["r"] for e in lives] or [0]
        speed = self.num_timesteps / (time.time() - self.start)
        left = (self.total - self.num_timesteps) / speed / 60
        say(f"{self.name:>4}: {self.num_timesteps:>7}/{self.total} steps | exits {exits:4.0%} crashes {crashes:4.0%} timeouts {timeouts:4.0%}"
            f" | avg reward {sum(rewards) / len(rewards):6.2f} (min {min(rewards):6.2f}, max {max(rewards):6.2f})"
            f" | {self.lives:>6} lives | {speed:5.1f} steps/s | {left:6.1f} min left")
        if len(lives) == 100 and exits >= v.TRAIN["goal_exit_rate"]:
            say(f"{self.name:>4}: hit the goal, {exits:.0%} of the last 100 lives made the exit")
            return False
        return True


def train(name, port, steps, stream):
    algo = getattr(importlib.import_module(v.MODEL["library"]), name)
    env = VecMonitor(SubprocVecEnv([partial(training_env, i, port, stream) for i in range(v.TRAIN["envs"])]), info_keywords=("outcome",))
    env.seed(v.TRAIN["seed"])
    model = algo("MlpPolicy", env, gamma=v.TRAIN["gamma"], seed=v.TRAIN["seed"], policy_kwargs=dict(net_arch=[256, 256], **v.TRAIN["policy_kwargs"].get(name, {})),
                 verbose=0, **v.TRAIN["settings"].get(name, {}))
    model.learn(steps, callback=Progress(name, steps))
    out = HERE / v.TRAIN["folder"] / name
    model.save(out)
    env.close()
    say(f"{name:>4}: done, saved {out}.zip")


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=v.TRAIN["models"])
    parser.add_argument("--steps", type=int, default=v.TRAIN["max_steps"])
    parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=None, help="--no-ui for servers")
    args = parser.parse_args(argv)
    if args.ui is None:
        if sys.stdin.isatty():
            args.ui = input("show the live training window? [Y/n] ").strip().lower() not in ("n", "no")
        else:
            args.ui = v.TRAIN["show_ui"]
    (HERE / v.TRAIN["folder"]).mkdir(exist_ok=True)
    LOG.write_text("")
    if args.ui:
        subprocess.Popen([sys.executable, str(HERE / "watch.py"), "--models", *args.models])
        # lower priority so the watcher stays smooth. no os.nice on windows
        if hasattr(os, "nice"):
            os.nice(5)
    else:
        say("training without the ui. progress goes here and to " + str(LOG))
    # port by TRAIN["models"] order so each model keeps its watch.py panel
    port = lambda name: v.TRAIN["watch_port"] + (v.TRAIN["models"].index(name) if name in v.TRAIN["models"] else len(v.TRAIN["models"]))
    jobs = [Process(target=train, args=(name, port(name), args.steps, args.ui)) for name in args.models]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()


if __name__ == "__main__":
    run()
