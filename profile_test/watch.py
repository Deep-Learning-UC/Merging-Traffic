import argparse
import pickle
import re
import socket
from collections import deque
from pathlib import Path

import numpy as np
import pygame

import variables as v
from main import DIM, ENDING_COLORS, PANEL, TEXT, key_items

LOG = Path(__file__).resolve().parent / v.TRAIN["folder"] / "training_log.txt"
PW, PH, CHART_H, FPS = 1000, 220, 240, 60
KEY_H = 30
PX_PER_M = 4.5
BUFFER = 40  # playback speeds up when more than this is waiting and slows down when less. keeps it smooth
GREY, WHITE = (100, 100, 100), (255, 255, 255)
COLORS = [(90, 170, 255), (240, 110, 200), (80, 210, 120), (240, 190, 60)]
TIME_LEFT = re.compile(r"(\w+):\s+(\d+)/\d+ steps .*?([\d.]+) min left")


# newest training steps and time left per model, from the training log
def read_log():
    latest = {}
    if LOG.exists():
        for name, steps, left in TIME_LEFT.findall(LOG.read_text()):
            latest[name] = (int(steps), float(left))
    return latest


# one chart: a filled line for a % of the last 100 lives, plus connected dots counting up a running total.
# used twice, once for exits and once for everything that went wrong. it moves every time a life ends
def draw_chart(font, big, title, names, panels, pct_key, dots_key, unit, legend, goal=None):
    chart = pygame.Surface((PW, CHART_H))
    chart.fill(PANEL)
    # fills go on their own see-through layer so they blend instead of stacking up into a solid block
    fills = pygame.Surface((PW, CHART_H), pygame.SRCALPHA)
    left, top, right, bottom = 60, 60, PW - 110, CHART_H - 30
    most = max([len(p[pct_key]) for p in panels] + [100])
    to_xy = lambda life, pct: (left + (right - left) * life / most, bottom - (bottom - top) * pct / 100)
    chart.blit(big.render(title, True, TEXT), (left, 10))
    # dots get their own scale on the right: a running count of exits, or an average life score
    values = [n for p in panels for _, n in p[dots_key]]
    lo, hi = (min(values + [0]), max(values + [0])) if unit == "score" else (0, max(values + [10]))
    hi = hi if hi > lo else lo + 1
    to_count_y = lambda n: bottom - (bottom - top) * (n - lo) / (hi - lo)
    for n in (lo, hi):
        chart.blit(font.render(f"{n:+.1f} score" if unit == "score" else f"{n:.0f} {unit}", True, DIM), (right + 40, to_count_y(n) - 8))
    for pct in (0, 25, 50, 75, 100):
        y = to_xy(0, pct)[1]
        pygame.draw.line(chart, (36, 42, 54), (left, y), (right, y))
        chart.blit(font.render(f"{pct}%", True, DIM), (left - 40, y - 8))
    if goal is not None:
        for x in range(left, right, 14):
            pygame.draw.line(chart, (240, 190, 60), (x, to_xy(0, goal)[1]), (x + 7, to_xy(0, goal)[1]), 2)
        chart.blit(font.render("goal", True, (240, 190, 60)), (right + 8, to_xy(0, goal)[1] - 8))
    chart.blit(font.render(f"lives (0 to {most})", True, DIM), (PW // 2 - 50, bottom + 8))
    for k, (name, panel) in enumerate(zip(names, panels)):
        color, points = COLORS[k % len(COLORS)], [to_xy(i + 1, pct) for i, pct in enumerate(panel[pct_key])]
        now = panel[pct_key][-1] if panel[pct_key] else 0
        chart.blit(font.render(legend(name, panel, now), True, color), (left + k * 310, 32))
        if len(points) > 1:
            pygame.draw.polygon(fills, (*color, 45), [(points[0][0], bottom)] + points + [(points[-1][0], bottom)])
            chart.blit(fills, (0, 0))
            fills.fill((0, 0, 0, 0))
            pygame.draw.lines(chart, color, False, points, 3)
            pygame.draw.circle(chart, color, points[-1], 5)
            chart.blit(font.render(f"{now:.0f}%", True, color), (points[-1][0] + 8, points[-1][1] - 8))
    # second pass so the dots always sit on top of every fill
    for k, panel in enumerate(panels):
        color = COLORS[k % len(COLORS)]
        dots = [(to_xy(life, 0)[0], to_count_y(n)) for life, n in panel[dots_key]]
        if len(dots) > 1:
            pygame.draw.lines(chart, color, False, dots, 2)
        # at most ~60 dots drawn, spread evenly, so they never smear into a solid band
        for dot in dots[::max(1, len(dots) // 60)] + dots[-1:]:
            pygame.draw.circle(chart, PANEL, dot, 5)
            pygame.draw.circle(chart, color, dot, 3)
        if dots:
            value = panel[dots_key][-1][1]
            chart.blit(font.render(f"{value:+.1f}" if unit == "score" else f"{value:.0f}", True, color), (dots[-1][0] + 6, dots[-1][1] - 14))
    return chart


# road, cars, and one small label. cars slide between the two physics frames they're between, thats the smoothness
def draw_panel(surface, panel, name, fonts):
    big, small = fonts
    surface.fill(GREY)
    # keeps drawing the last life until the next one has its first frame, so the road never blinks out
    life = panel["drawn"]
    if life is None or panel["now"] is None:
        surface.blit(big.render(f"{name}: waiting for train.py...", True, TEXT), (20, 20))
        return
    cars, speed, to_exit, side = panel["now"]
    after = panel["next"]
    if after is not None and after[0].shape == cars.shape:
        cars = cars + (after[0] - cars) * panel["blend"]
    cam = cars[0][:2]
    to_px = lambda x, y: (float(PW * 0.35 + (x - cam[0]) * PX_PER_M), float(PH * 0.6 + (y - cam[1]) * PX_PER_M))
    view = (cam[0] - PW * 0.35 / PX_PER_M - 10, cam[0] + PW * 0.65 / PX_PER_M + 10)
    for sx, sy, ex, ey, width, left, right in life["lanes"]:
        d = np.array([ex - sx, ey - sy])
        length = np.linalg.norm(d)
        d = d / length
        n = np.array([-d[1], d[0]])
        lo, hi = max(0, (view[0] - sx) / max(d[0], 1e-6)), min(length, (view[1] - sx) / max(d[0], 1e-6))
        for offset, kind in ((-0.5, left), (0.5, right)):
            base = np.array([sx, sy]) + n * width * offset
            if kind in (2, 3):
                pygame.draw.line(surface, WHITE, to_px(*(base + d * lo)), to_px(*(base + d * hi)), 2)
            elif kind == 1:
                for s in np.arange(lo - lo % 7.33, hi, 7.33):
                    pygame.draw.line(surface, WHITE, to_px(*(base + d * s)), to_px(*(base + d * min(s + 3, hi))), 2)
    for bx, by in life["blocks"]:
        x, y = to_px(bx, by)
        pygame.draw.rect(surface, (200, 200, 0), (x - 5, y - 5, 10, 10))
    for x, y, heading, r, g, b in cars:
        c, s = np.cos(heading), np.sin(heading)
        corners = [to_px(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((2.5, 1), (2.5, -1), (-2.5, -1), (-2.5, 1))]
        pygame.draw.polygon(surface, (int(r), int(g), int(b)), corners)
        pygame.draw.polygon(surface, (40, 40, 40), corners, 1)
    last = panel["last100"]
    pct = lambda k: f"{100 * last.count(k) / max(len(last), 1):.0f}%"
    pygame.draw.rect(surface, PANEL, (10, 10, 520, 48), border_radius=10)
    surface.blit(big.render(f"{name}  watching life #{life['id']}   ({panel['lives']} done)", True, TEXT), (22, 14))
    surface.blit(small.render(f"of the last 100 lives: {pct('made exit')} made the exit, {pct('crashed')} crashed    {speed * 2.237:3.0f} mph   "
                              f"{side} exit in {max(to_exit, 0):4.0f} m", True, DIM), (22, 36))
    if panel["banner"] > 0:
        text = big.render(panel["ending"].upper(), True, ENDING_COLORS[panel["ending"]])
        box = text.get_rect(center=(PW // 2, PH // 2)).inflate(40, 20)
        pygame.draw.rect(surface, PANEL, box, border_radius=10)
        surface.blit(text, text.get_rect(center=box.center))


# color key across the very top: you, every npc personality from variables.py, and wrecks
def draw_key(font):
    key = pygame.Surface((PW, KEY_H))
    key.fill(PANEL)
    items = key_items()
    x = 14
    for name, color in items:
        pygame.draw.rect(key, color, (x, 10, 18, 10), border_radius=3)
        key.blit(font.render(name, True, TEXT), (x + 24, 7))
        x += 40 + font.size(name)[0]
    return key


# sorts one packet from training: a new life starting, a frame of one, or one ending
def receive(panel, msg):
    rank, kind, *data = msg
    if kind == "road":
        panel["started"] += 1
        panel["copies"][rank] = {"id": panel["started"], "lanes": data[0], "blocks": data[1], "frames": deque()}
    elif rank in panel["copies"]:
        life = panel["copies"][rank]
        life["frames"].append((kind, *data))
        if kind == "end":
            panel["lives"] += 1
            panel["last100"].append(data[0])
            n = len(panel["last100"])
            panel["history"].append(100 * panel["last100"].count("made exit") / n)
            panel["crash_history"].append(100 * panel["last100"].count("crashed") / n)
            # average score of every 25 lives becomes one dot on the bottom chart
            panel["batch"].append(data[1])
            if len(panel["batch"]) == 25:
                panel["scores"].append((panel["lives"], sum(panel["batch"]) / 25))
                panel["batch"].clear()
            # every exit adds a dot to the success count
            if data[0] == "made exit":
                panel["records"].append((panel["lives"], len(panel["records"]) + 1))


# moves playback forward. when the life being watched ends, jump to the newest life still going
def advance(panel):
    panel["banner"] -= 1
    life = panel["playing"]
    if life is None:
        if not panel["copies"] or panel["banner"] > 0:
            return
        life = panel["playing"] = max(panel["copies"].values(), key=lambda l: l["id"])
        panel["blend"], panel["level"], panel["fresh"] = 0.0, 0.0, True
    frames = life["frames"]
    panel["level"] = 0.95 * panel["level"] + 0.05 * len(frames)
    panel["blend"] += max(panel["level"], 1) / BUFFER
    while panel["blend"] >= 1 and frames:
        panel["blend"] -= 1
        kind, *data = frames.popleft()
        if kind == "end":
            panel["ending"], panel["banner"], panel["playing"] = data[0], FPS // 2, None
            return
        if panel["fresh"]:
            panel["drawn"], panel["next"], panel["fresh"] = life, None, False
        panel["now"], panel["next"] = panel["next"] or data, data
    panel["blend"] = min(panel["blend"], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=v.TRAIN["models"])
    args = parser.parse_args()

    socks = []
    for k in range(len(args.models)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        sock.bind(("127.0.0.1", v.TRAIN["watch_port"] + k))
        sock.setblocking(False)
        socks.append(sock)
    panels = [{"copies": {}, "playing": None, "started": 0, "lives": 0, "last100": deque(maxlen=100), "history": [], "records": [], "crash_history": [], "batch": [], "scores": [],
               "drawn": None, "fresh": True, "now": None, "next": None, "blend": 0.0, "level": 0.0, "banner": 0, "ending": ""} for _ in args.models]

    pygame.init()
    screen = pygame.display.set_mode((PW, KEY_H + PH * len(panels) + CHART_H * 2))
    pygame.display.set_caption("training time lapse")
    fonts = (pygame.font.SysFont("DejaVu Sans", 15, bold=True), pygame.font.SysFont("DejaVu Sans", 12))
    surfaces = [pygame.Surface((PW, PH)) for _ in panels]
    key = draw_key(pygame.font.SysFont("DejaVu Sans", 13))
    clock, log, log_time = pygame.time.Clock(), {}, -9999
    try:
        while True:
            if any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE) for e in pygame.event.get()):
                return
            for sock, panel in zip(socks, panels):
                while True:
                    try:
                        receive(panel, pickle.loads(sock.recv(65536)))
                    except BlockingIOError:
                        break
                advance(panel)
            for k, (name, panel) in enumerate(zip(args.models, panels)):
                draw_panel(surfaces[k], panel, name, fonts)
                screen.blit(surfaces[k], (0, KEY_H + k * PH))
            if pygame.time.get_ticks() - log_time > 2000:
                log, log_time = read_log(), pygame.time.get_ticks()
            pct = lambda p, kind: f"{100 * p['last100'].count(kind) / max(len(p['last100']), 1):.0f}%"
            good = draw_chart(fonts[1], fonts[0], "% of the last 100 lives that made the exit      (dots = total exits)", args.models, panels,
                              "history", "records", "exits", goal=v.TRAIN["goal_exit_rate"] * 100,
                              legend=lambda name, p, now: f"{name}: {now:.0f}% made exit   {len(p['records'])} total   {log.get(name, (0, 0))[1]:.0f} min left")
            bad = draw_chart(fonts[1], fonts[0], "% of the last 100 lives that crashed      (dots = average score of every 25 lives)", args.models, panels,
                             "crash_history", "scores", "score",
                             legend=lambda name, p, now: f"{name}: crashed {pct(p, 'crashed')}  missed {pct(p, 'missed exit')}  timed out {pct(p, 'timed out')}")
            screen.blit(key, (0, 0))
            screen.blit(good, (0, KEY_H + PH * len(panels)))
            screen.blit(bad, (0, KEY_H + PH * len(panels) + CHART_H))
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
