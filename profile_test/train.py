# trains every model in TRAIN["models"] at the same time, saves checkpoints to TRAIN["folder"],
# and stops a model early once it makes the exit TRAIN["goal_exit_rate"] of its last 100 lives
import argparse
import importlib
import os
import pickle
import socket
import time
from functools import partial
from multiprocessing import Process
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import variables as v
from environment import make_env

HERE = Path(__file__).resolve().parent


# sends one road copy's lives to watch.py as tiny udp packets: the road once per life, then car positions every
# physics frame. if watch.py isnt open the packets just go nowhere, so training doesnt care either way
class Broadcast(gym.Wrapper):
    def __init__(self, env, port):
        super().__init__(env)
        self.sock, self.addr, self.life = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), ("127.0.0.1", port), 0

    def send(self, msg):
        try:
            self.sock.sendto(pickle.dumps(msg), self.addr)
        except OSError:
            pass

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        env, self.life = self.env.unwrapped, self.life + 1
        lanes = [(*l.start, *l.end, l.width, int(l.line_types[0]), int(l.line_types[1]))
                 for f in env.road.network.graph for t in env.road.network.graph[f] for l in env.road.network.graph[f][t]]
        self.send(("road", lanes, [tuple(o.position) for o in env.road.objects], self.life))
        step, me = env.road.step, env.vehicle

        def send_cars(dt):
            step(dt)
            cars = np.array([[c.position[0], c.position[1], c.heading, *(getattr(c, "color", None) or (255, 255, 255))]
                             for c in env.road.vehicles], np.float32)
            self.send(("cars", cars, me.speed, env.split_x - me.position[0], env.exit_side))
        env.road.step = send_cars
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if result[2] or result[3]:
            self.send(("end", result[4]["outcome"] or "timed out"))
        return result


# each road copy runs in its own process, so YOU has to be set in there too or it would be dummydriver.
# only copy 0 streams to watch.py
def training_env(rank, port):
    v.YOU["driver"] = "model"
    env = make_env(max_seconds=v.TRAIN["max_life_seconds"])
    return Broadcast(env, port) if rank == 0 else env


# prints every 1000 steps and saves checkpoints. exits/crashes are out of the last 100 lives, exits going up = learning
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
        crashes = sum(e["outcome"] == "crashed" for e in lives) / n
        timeouts = sum(e["outcome"] is None for e in lives) / n
        rewards = [e["r"] for e in lives] or [0]
        speed = self.num_timesteps / (time.time() - self.start)
        left = (self.total - self.num_timesteps) / speed / 60
        print(f"{self.name:>4}: {self.num_timesteps:>7}/{self.total} steps | exits {exits:4.0%} crashes {crashes:4.0%} timeouts {timeouts:4.0%}"
              f" | avg reward {sum(rewards) / len(rewards):6.2f} (min {min(rewards):6.2f}, max {max(rewards):6.2f})"
              f" | {self.lives:>6} lives | {speed:5.1f} steps/s | {left:6.1f} min left", flush=True)
        if len(lives) == 100 and exits >= v.TRAIN["goal_exit_rate"]:
            print(f"{self.name:>4}: hit the goal, {exits:.0%} of the last 100 lives made the exit", flush=True)
            return False
        return True


def train(name, port, steps):
    algo = getattr(importlib.import_module(v.MODEL["library"]), name)
    env = VecMonitor(SubprocVecEnv([partial(training_env, i, port) for i in range(v.TRAIN["envs"])]), info_keywords=("outcome",))
    env.seed(v.TRAIN["seed"])
    model = algo("MlpPolicy", env, gamma=v.TRAIN["gamma"], seed=v.TRAIN["seed"], policy_kwargs=dict(net_arch=[256, 256]),
                 verbose=0, **v.TRAIN["settings"].get(name, {}))
    model.learn(steps, callback=Progress(name, steps))
    out = HERE / v.TRAIN["folder"] / name
    model.save(out)
    env.close()
    print(f"{name:>4}: done, saved {out}.zip", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=v.TRAIN["models"])
    parser.add_argument("--steps", type=int, default=v.TRAIN["max_steps"])
    args = parser.parse_args()
    (HERE / v.TRAIN["folder"]).mkdir(exist_ok=True)
    # slightly lower priority so watch.py stays smooth. training only loses what the watcher actually uses
    os.nice(5)
    jobs = [Process(target=train, args=(name, v.TRAIN["watch_port"] + k, args.steps)) for k, name in enumerate(args.models)]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
