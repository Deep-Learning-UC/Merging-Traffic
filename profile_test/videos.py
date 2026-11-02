# test drives the road with the dummydriver or a trained model, draws the HUD, and saves each life as a video in saved_videos/
import argparse
import random
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from environment import make_env
from main import BLUE, DIM, ENDING_COLORS, ENDINGS, PANEL, TEXT, ask_model, key_items, load_model
from variables import MODEL, TRAIN, VIDEO, YOU

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE.parent / "saved_videos"
WIDTH, HEIGHT = 1280, 448
ACTIONS = ["Left", "Idle", "Right", "Faster", "Slower"]


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


BIG, MED, SMALL = font(26, True), font(16, True), font(14)


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
    box = (*PANEL, 200)

    d.rounded_rectangle((16, 16, 340, 132), 12, fill=box)
    d.text((32, 24), "Merging Traffic", font=BIG, fill=TEXT)
    d.text((32, 60), f"episode {hud['ep']}   t = {hud['t']:4.1f}s   {hud['lanes']} lanes   {hud['cars']} cars", font=SMALL, fill=DIM)
    d.text((32, 80), f"you: {hud['me']} ({YOU['driver']})   {hud['speed'] * 2.237:4.0f} mph", font=MED, fill=VIDEO["you_color"])
    d.text((32, 104), f"{hud['side']} exit in {max(hud['to_exit'], 0):4.0f} m   reward {hud['reward']:+.2f}", font=SMALL, fill=DIM)

    x0, y0 = WIDTH - 316, 16
    d.rounded_rectangle((x0, y0, WIDTH - 16, y0 + 200), 12, fill=box)
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

    items = key_items()
    d.rounded_rectangle((16, HEIGHT - 48, 30 + sum(30 + d.textlength(k, font=SMALL) for k, _ in items), HEIGHT - 14), 10, fill=box)
    x = 30
    for name, color in items:
        d.rounded_rectangle((x, HEIGHT - 38, x + 18, HEIGHT - 26), 3, fill=color)
        d.text((x + 24, HEIGHT - 40), name, font=SMALL, fill=TEXT)
        x += 30 + d.textlength(name, font=SMALL)

    if ending:
        text, color = ENDINGS[ending], ENDING_COLORS[ending]
        tw = d.textlength(text, font=BIG)
        d.rounded_rectangle(((WIDTH - tw) / 2 - 30, HEIGHT / 2 - 30, (WIDTH + tw) / 2 + 30, HEIGHT / 2 + 30), 14, fill=box)
        d.text(((WIDTH - tw) / 2, HEIGHT / 2 - 16), text, font=BIG, fill=color)
    return np.array(Image.alpha_composite(img, layer).convert("RGB"))

# tkinter window that shows the frames live while the videos get made, with a Skip button
class Viewer:
    def __init__(self, title, fps=None):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        bar = tk.Frame(self.root, bg="black")
        bar.pack(fill="x")
        self.status = tk.Label(bar, text="", bg="black", fg="#9aa3b2", font=("DejaVu Sans", 11))
        self.status.pack(side="left", fill="x", expand=True)
        self.skip = False
        tk.Button(bar, text="Skip", command=self.press_skip).pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.open = True
        self.wait = 1 / fps if fps else 0
        self.last = 0

    # puts a frame on screen, waits if --realtime
    def show(self, frame, status=""):
        if not self.open:
            return
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.label.configure(image=photo)
        # if you dont keep a reference tkinter garbage collects the image and shows nothing. lost an hour to this
        self.label.image = photo
        self.status.configure(text=status)
        if self.wait:
            left = self.wait - (time.time() - self.last)
            if left > 0:
                time.sleep(left)
            self.last = time.time()
        # no mainloop, we just poke the window every frame. hacky but it works
        self.root.update()

    def press_skip(self):
        self.skip = True

    def close(self):
        if self.open:
            self.open = False
            self.root.destroy()


def make_viewer(title, fps=None):
    try:
        return Viewer(title, fps)
    except Exception as e:
        print(f"no tkinter window ({e}), still saving the videos")
        return None


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

# parses options, loads the model, and records each episode. main.py calls this
def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=VIDEO["episodes"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--model", help="name of a model in TRAIN folder, like PPO. makes it drive")
    args = parser.parse_args(argv)
    if args.model:
        MODEL.update(name=args.model, path=f"{TRAIN['folder']}/{args.model}.zip")
        YOU["driver"] = "model"
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    print(f"seed {seed}")

    VIDEO_DIR.mkdir(exist_ok=True)
    show = VIDEO["show_window"] and not args.no_window
    viewer = None if not show else make_viewer("Merging Traffic", VIDEO["fps"] if args.realtime else None)
    env = make_env("rgb_array", screen_width=WIDTH, screen_height=HEIGHT, scaling=6.0,
                   centering_position=[0.35, 0.62], simulation_frequency=VIDEO["fps"], policy_frequency=1)
    model = load_model(env)
    driver = MODEL["name"] if YOU["driver"] == "model" else YOU["driver"]
    for ep in range(args.episodes):
        out = VIDEO_DIR / f"profiles_{driver}_ep{ep}.mp4"
        result, t = record(env, model, ep, seed + ep, out, viewer)
        print(f"episode {ep}: {result} after {t:.0f}s -> {out}")
    env.close()
    if viewer:
        viewer.close()


if __name__ == "__main__":
    run()
