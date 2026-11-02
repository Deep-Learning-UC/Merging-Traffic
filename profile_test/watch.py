# live time lapse of the actual training lives, one panel per model, plus an exit rate chart from the training log.
# train.py streams car positions from one road copy per model. this window just draws them, smoothly sliding
# between physics frames, about a second behind live. close the window or press esc, training keeps going
import argparse
import pickle
import re
import socket
from collections import deque
from pathlib import Path

import numpy as np
import pygame

import variables as v

LOG = Path(__file__).resolve().parent / v.TRAIN["folder"] / "training_log.txt"
PW, PH, CHART_H, FPS = 1000, 280, 190, 60
PX_PER_M = 4.5
BUFFER = 60  # playback speeds up when more than this is waiting and slows down when less. keeps it smooth
GREY, WHITE, PANEL, TEXT, DIM = (100, 100, 100), (255, 255, 255), (18, 22, 30), (235, 238, 245), (150, 158, 172)
COLORS = [(90, 170, 255), (240, 110, 200), (80, 210, 120), (240, 190, 60)]
ENDINGS = {"crashed": (240, 80, 80), "made exit": (80, 210, 120), "missed exit": (240, 190, 60), "timed out": (150, 158, 172)}
LINE = re.compile(r"(\w+):\s+(\d+)/\d+ steps \| exits\s+(\d+)% crashes\s+(\d+)% .*?\|\s+(\d+) lives .*?\|\s+([\d.]+) min left")


# {name: [(steps, exit %, crash %, lives, min left), ...]} from the training log
def read_log():
    curves = {}
    if LOG.exists():
        for name, steps, exits, crashes, lives, left in LINE.findall(LOG.read_text()):
            curves.setdefault(name, []).append((int(steps), int(exits), int(crashes), int(lives), float(left)))
    return curves


# exit rate of the last 100 lives vs training steps. thick = exits, faded = crashes
def draw_chart(font, curves):
    chart = pygame.Surface((PW, CHART_H))
    chart.fill(PANEL)
    if not curves:
        chart.blit(font.render("waiting for train.py...", True, DIM), (20, 20))
        return chart
    max_steps = max(c[-1][0] for c in curves.values())
    to_xy = lambda s, pct: (50 + (PW - 70) * s / max_steps, CHART_H - 26 - (CHART_H - 70) * pct / 100)
    for pct in (0, 50, 100):
        chart.blit(font.render(f"{pct}%", True, DIM), (8, to_xy(0, pct)[1] - 8))
        pygame.draw.line(chart, (40, 46, 58), to_xy(0, pct), to_xy(max_steps, pct))
    for k, (name, curve) in enumerate(sorted(curves.items())):
        color = COLORS[k % len(COLORS)]
        if len(curve) > 1:
            pygame.draw.lines(chart, tuple(c // 2 for c in color), False, [to_xy(p[0], p[2]) for p in curve], 1)
            pygame.draw.lines(chart, color, False, [to_xy(p[0], p[1]) for p in curve], 3)
        steps, exits, crashes, lives, left = curve[-1]
        chart.blit(font.render(f"{name}: exits {exits}%  crashes {crashes}%  {lives} lives  {left:.0f} min left", True, color), (50 + k * 320, 8))
    chart.blit(font.render(f"exit rate of the last 100 lives (faded = crashes)   training steps 0 to {max_steps}", True, DIM), (50, CHART_H - 20))
    return chart


# road, cars, and one small label. cars slide between the two physics frames they're between, thats the smoothness
def draw_panel(surface, panel, name, fonts):
    big, small = fonts
    surface.fill(GREY)
    if panel["road"] is None or panel["now"] is None:
        surface.blit(big.render(f"{name}: waiting for train.py...", True, TEXT), (20, 20))
        return
    lanes, blocks, life = panel["road"]
    cars, speed, to_exit, side = panel["now"]
    after = panel["next"]
    if after is not None and after[0].shape == cars.shape:
        cars = cars + (after[0] - cars) * panel["blend"]
    cam = cars[0][:2]
    to_px = lambda x, y: (float(PW * 0.35 + (x - cam[0]) * PX_PER_M), float(PH * 0.6 + (y - cam[1]) * PX_PER_M))
    view = (cam[0] - PW * 0.35 / PX_PER_M - 10, cam[0] + PW * 0.65 / PX_PER_M + 10)
    for sx, sy, ex, ey, width, left, right in lanes:
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
    for bx, by in blocks:
        x, y = to_px(bx, by)
        pygame.draw.rect(surface, (200, 200, 0), (x - 5, y - 5, 10, 10))
    for x, y, heading, r, g, b in cars:
        c, s = np.cos(heading), np.sin(heading)
        corners = [to_px(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((2.5, 1), (2.5, -1), (-2.5, -1), (-2.5, 1))]
        pygame.draw.polygon(surface, (int(r), int(g), int(b)), corners)
        pygame.draw.polygon(surface, (40, 40, 40), corners, 1)
    pygame.draw.rect(surface, PANEL, (10, 10, 360, 48), border_radius=10)
    surface.blit(big.render(f"{name}  life {life}", True, TEXT), (22, 14))
    surface.blit(small.render(f"{speed * 2.237:3.0f} mph   {side} exit in {max(to_exit, 0):4.0f} m", True, DIM), (22, 36))
    if panel["banner"] > 0:
        text = big.render(panel["ending"].upper(), True, ENDINGS[panel["ending"]])
        box = text.get_rect(center=(PW // 2, PH // 2)).inflate(40, 20)
        pygame.draw.rect(surface, PANEL, box, border_radius=10)
        surface.blit(text, text.get_rect(center=box.center))


# moves playback forward. how many frames per screen frame depends on how much is waiting, so it never stalls or runs away
def advance(panel):
    buf = panel["buf"]
    panel["level"] = 0.95 * panel["level"] + 0.05 * len(buf)
    panel["blend"] += max(panel["level"], 0.5) / BUFFER
    panel["banner"] -= 1
    while panel["blend"] >= 1 and buf:
        panel["blend"] -= 1
        msg = buf.popleft()
        if msg[0] == "road":
            panel["road"], panel["now"], panel["next"] = msg[1:], None, None
        elif msg[0] == "end":
            panel["ending"], panel["banner"] = msg[1], FPS
        else:
            panel["now"], panel["next"] = panel["next"] or msg[1:], msg[1:]
    panel["blend"] = min(panel["blend"], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=v.TRAIN["models"])
    args = parser.parse_args()

    socks = []
    for k in range(len(args.models)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", v.TRAIN["watch_port"] + k))
        sock.setblocking(False)
        socks.append(sock)
    panels = [{"buf": deque(), "road": None, "now": None, "next": None, "blend": 0.0, "level": 0.0, "banner": 0, "ending": ""}
              for _ in args.models]

    pygame.init()
    screen = pygame.display.set_mode((PW, PH * len(panels) + CHART_H))
    pygame.display.set_caption("training time lapse")
    fonts = (pygame.font.SysFont("DejaVu Sans", 15, bold=True), pygame.font.SysFont("DejaVu Sans", 12))
    surfaces = [pygame.Surface((PW, PH)) for _ in panels]
    clock, chart, chart_time = pygame.time.Clock(), None, -9999
    try:
        while True:
            if any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE) for e in pygame.event.get()):
                return
            for sock, panel in zip(socks, panels):
                while True:
                    try:
                        panel["buf"].append(pickle.loads(sock.recv(65536)))
                    except BlockingIOError:
                        break
                advance(panel)
            for k, (name, panel) in enumerate(zip(args.models, panels)):
                draw_panel(surfaces[k], panel, name, fonts)
                screen.blit(surfaces[k], (0, k * PH))
            if pygame.time.get_ticks() - chart_time > 2000:
                chart, chart_time = draw_chart(fonts[1], read_log()), pygame.time.get_ticks()
            screen.blit(chart, (0, PH * len(panels)))
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
