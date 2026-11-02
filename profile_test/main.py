# runs episodes, draws the HUD, and saves each one as a video in saved_videos/
import argparse
import importlib
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from variables import MODEL, PERSONALITIES, VIDEO, YOU
from viewer import make_viewer
from environment import WRECK_COLOR, make_env

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parent / "saved_videos"
WIDTH, HEIGHT = 1280, 448
ACTIONS = ["Left", "Idle", "Right", "Faster", "Slower"]
PANEL, TEXT, DIM, BLUE = (18, 22, 30, 200), (235, 238, 245), (150, 158, 172), (90, 170, 255)
ENDINGS = {"skipped": ("SKIPPED", (150, 158, 172)), "crashed": ("CRASHED", (240, 80, 80)), "made exit": ("MADE THE EXIT", (80, 210, 120)), "missed exit": ("MISSED THE EXIT", (240, 190, 60))}

def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)

BIG, MED, SMALL = font(26, True), font(16, True), font(14)

# MODEL["name"] is looked up in MODEL["library"], so new models need no code changes
def load_model():
    try:
        algo = getattr(importlib.import_module(MODEL["library"]), MODEL["name"])
        return algo.load((HERE / MODEL["path"]).resolve(), device="cpu")
    except Exception as e:
        print(f"couldnt load {MODEL['name']} from {MODEL['path']} ({e.__class__.__name__}: {e}), driving without it")
        return None

# DQN-style models expose q values, PPO-style expose action probabilities
def ask_model(model, obs):
    if model is None:
        return 1, None, ""
    import torch
    with torch.no_grad():
        if hasattr(model, "q_net"):
            scores, kind = model.q_net(model.q_net.obs_to_tensor(obs)[0])[0].numpy(), "q values"
        elif hasattr(model.policy, "get_distribution"):
            scores, kind = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs[0].numpy(), "chances"
        else:
            return int(model.predict(obs, deterministic=True)[0]), None, ""
    return int(np.argmax(scores)), scores, kind

class FrameGrabber:
    # highway-env only gives the in between frames if it thinks a video recorder is attached. so we pretend to be one
    def __init__(self, env):
        self.env, self.frames = env, []

    def _capture_frame(self):
        self.frames.append(self.env.render())

# draws the info box, model panel, color key, and end banner onto a frame
def draw_hud(frame, hud, ending=None):
    img = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", img.size)
    d = ImageDraw.Draw(layer)

    d.rounded_rectangle((16, 16, 340, 132), 12, fill=PANEL)
    d.text((32, 24), "Merging Traffic", font=BIG, fill=TEXT)
    d.text((32, 60), f"episode {hud['ep']}   t = {hud['t']:4.1f}s   {hud['lanes']} lanes   {hud['cars']} cars", font=SMALL, fill=DIM)
    d.text((32, 80), f"you: {hud['me']} ({YOU['driver']})   {hud['speed'] * 2.237:4.0f} mph", font=MED, fill=VIDEO["you_color"])
    d.text((32, 104), f"{hud['side']} exit in {max(hud['to_exit'], 0):4.0f} m   reward {hud['reward']:+.2f}", font=SMALL, fill=DIM)

    x0, y0 = WIDTH - 316, 16
    d.rounded_rectangle((x0, y0, WIDTH - 16, y0 + 200), 12, fill=PANEL)
    title = f"{MODEL['name']} driving" if YOU["driver"] == "model" else f"{MODEL['name']} would pick"
    d.text((x0 + 16, y0 + 12), title if hud["has_model"] else "No model", font=MED, fill=TEXT)
    if hud["q"] is not None:
        d.text((WIDTH - 32 - d.textlength(hud["kind"], font=SMALL), y0 + 14), hud["kind"], font=SMALL, fill=DIM)
        q, a = hud["q"], hud["action"]
        span = max(q.max() - q.min(), 1e-6)
        for i, name in enumerate(ACTIONS):
            y, picked = y0 + 48 + i * 28, i == a
            fill = int(150 * (0.15 + 0.85 * (q[i] - q.min()) / span))
            d.text((x0 + 16, y), name, font=MED if picked else SMALL, fill=TEXT if picked else DIM)
            d.rounded_rectangle((x0 + 90, y + 3, x0 + 240, y + 15), 6, fill=(45, 52, 66))
            d.rounded_rectangle((x0 + 90, y + 3, x0 + 90 + fill, y + 15), 6, fill=BLUE if picked else (95, 105, 125))
            value = f"{q[i] * 100:.0f}%" if hud["kind"] == "chances" else f"{q[i]:.2f}"
            d.text((x0 + 248, y), value, font=SMALL, fill=TEXT if picked else DIM)

    items = [("you", VIDEO["you_color"])] + [(k, p["color"]) for k, p in PERSONALITIES.items()] + [("wreck", WRECK_COLOR)]
    d.rounded_rectangle((16, HEIGHT - 48, 30 + sum(30 + d.textlength(k, font=SMALL) for k, _ in items), HEIGHT - 14), 10, fill=PANEL)
    x = 30
    for name, color in items:
        d.rounded_rectangle((x, HEIGHT - 38, x + 18, HEIGHT - 26), 3, fill=color)
        d.text((x + 24, HEIGHT - 40), name, font=SMALL, fill=TEXT)
        x += 30 + d.textlength(name, font=SMALL)

    if ending:
        text, color = ENDINGS[ending]
        tw = d.textlength(text, font=BIG)
        d.rounded_rectangle(((WIDTH - tw) / 2 - 30, HEIGHT / 2 - 30, (WIDTH + tw) / 2 + 30, HEIGHT / 2 + 30), 14, fill=PANEL)
        d.text(((WIDTH - tw) / 2, HEIGHT / 2 - 16), text, font=BIG, fill=color)
    return np.array(Image.alpha_composite(img, layer).convert("RGB"))

# plays one episode and saves it as an mp4
def record(env, model, ep, seed, out_path, viewer):
    obs, _ = env.reset(seed=seed)
    me, grabber = env.vehicle, FrameGrabber(env)
    env._record_video_wrapper = grabber
    hud = {"ep": ep, "t": 0.0, "lanes": env.lanes, "cars": len(env.road.vehicles) - 1, "me": me.personality,
           "speed": me.speed, "to_exit": env.split_x - me.position[0], "side": env.exit_side, "reward": 0.0, "action": 1, "q": None,
           "kind": "", "has_model": model is not None}
    writer = imageio.get_writer(out_path, fps=VIDEO["fps"], codec="libx264", quality=8, macro_block_size=16)
    count = 0

    def show(frame, ending=None):
        nonlocal count
        count += 1
        if ending is None and count % VIDEO["speed_up"]:
            return
        picture = draw_hud(frame, hud, ending)
        writer.append_data(picture)
        if viewer:
            viewer.show(picture, f"episode {ep}  ->  {out_path.name}")

    frame, done, skipped = env.render(), False, False
    show(frame)
    if viewer:
        viewer.skip = False
    while not done:
        if (viewer and viewer.skip) or hud["t"] > VIDEO["give_up_seconds"]:
            skipped = True
            break
        hud["action"], hud["q"], hud["kind"] = ask_model(model, obs)
        grabber.frames = []
        obs, reward, done, _, _ = env.step(hud["action"])
        # forgot this once and every video was 3 seconds long. never again
        grabber.frames.append(env.render())
        hud["reward"] += reward
        for frame in grabber.frames:
            hud["t"] += 1 / len(grabber.frames)
            hud["speed"], hud["to_exit"] = me.speed, env.split_x - me.position[0]
            show(frame)
    result = "skipped" if skipped else env.outcome()
    for _ in range(int(VIDEO["fps"] * VIDEO["end_card_seconds"])):
        show(frame, result)
    writer.close()
    return result, hud["t"]

# parses args, loads the model, and records each episode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=VIDEO["episodes"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    print(f"seed {seed}")

    VIDEO_DIR.mkdir(exist_ok=True)
    model = load_model()
    show = VIDEO["show_window"] and not args.no_window
    viewer = None if not show else make_viewer("Merging Traffic", VIDEO["fps"] if args.realtime else None)
    env = make_env("rgb_array", screen_width=WIDTH, screen_height=HEIGHT, scaling=6.0,
                   centering_position=[0.35, 0.62], simulation_frequency=VIDEO["fps"], policy_frequency=1)
    for ep in range(args.episodes):
        out = VIDEO_DIR / f"profiles_merge_ep{ep}.mp4"
        result, t = record(env, model, ep, seed + ep, out, viewer)
        print(f"episode {ep}: {result} after {t:.0f}s -> {out}")
    env.close()
    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
