# runs a bunch of episodes and checks the variables actually change how things drive
from collections import Counter

import variables as v
from environment import make_env

OUTCOMES = {"made exit", "missed exit", "crashed", "stuck"}
# the env has no time limit. but a total pileup can block every lane forever, so the test calls that "stuck" and moves on
MAX_STEPS = 300


# plays full episodes and collects outcomes and speeds
def run(seeds, you="random"):
    v.YOU["personality"] = you
    env, results = make_env(), []
    speeds, lane_speeds = {k: [] for k in v.PERSONALITIES}, {"left": [], "right": []}
    for s in seeds:
        env.reset(seed=s)
        info = {"lanes": env.lanes, "cars": len(env.road.vehicles) - 1, "me": env.vehicle.personality,
                "side": env.exit_side, "length": round(env.split_x)}
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            _, _, done, _, _ = env.step(1)
            steps += 1
            for car in env.road.vehicles[1:]:
                if car.crashed:
                    continue
                speeds[car.personality].append(car.speed)
                f, t, i = car.lane_index
                if (f, t) == ("b", "c") and i in (0, env.lanes - 1):
                    lane_speeds["left" if i == 0 else "right"].append((car.target_speed, car.speed))
        info["out"] = env.outcome() or "stuck"
        results.append(info)
    return results, speeds, lane_speeds


def avg(xs):
    return sum(xs) / max(len(xs), 1)


# the actual checks
def main():
    results, speeds, lane_speeds = run(range(12))
    for r in results:
        assert v.ROAD["lanes"][0] <= r["lanes"] <= v.ROAD["lanes"][1] and r["cars"] <= v.TRAFFIC["max_cars"]
        assert r["out"] in OUTCOMES
        print(f"{r['lanes']} lanes, {r['cars']:2} cars, {r['side']:5} exit, you = {r['me']:6} -> {r['out']}")
    assert len({r["length"] for r in results}) > 1, "road never changed"
    if v.ROAD["exit_side"] == "random":
        assert len({r["side"] for r in results}) > 1, "exit never switched sides"
    if len(v.PERSONALITIES) > 1:
        assert len({r["me"] for r in results}) > 1, "you were always the same personality"

    by_speed = sorted(v.PERSONALITIES, key=lambda k: v.PERSONALITIES[k]["speed"])
    print()
    for k in by_speed:
        print(f"{k:>8} avg {avg(speeds[k]):.1f} m/s")
    assert avg(speeds[by_speed[0]]) < avg(speeds[by_speed[-1]])

    # left lane should be the fast lane. checks what drivers want, since in a jam every lane crawls the same
    want = {k: avg([w for w, _ in xs]) for k, xs in lane_speeds.items()}
    real = {k: avg([r for _, r in xs]) for k, xs in lane_speeds.items()}
    print(f"left lane wants {want['left']:.1f} (actual {real['left']:.1f}), right lane wants {want['right']:.1f} (actual {real['right']:.1f}) m/s")
    assert want["left"] > want["right"]

    name = by_speed[0]
    fixed, _, _ = run(range(3), you=name)
    assert all(r["me"] == name for r in fixed), "setting YOU didnt stick"

    print("outcomes:", dict(Counter(r["out"] for r in results)))
    print("passed")


if __name__ == "__main__":
    main()
