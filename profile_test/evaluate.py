# scores every trained model plus the dummydriver baseline on the same unseen roads, writes results.md
from collections import Counter
from pathlib import Path

import variables as v
from environment import make_env
from main import ask_model, load_model

HERE = Path(__file__).resolve().parent
# training used seeds near TRAIN["seed"]. these are far away so nobody gets tested on a road it trained on
FIRST_SEED = 10_000
# same as the test. a total pileup can block every lane forever
MAX_STEPS = 300


def score(driver):
    v.YOU["driver"] = "dummydriver" if driver == "dummydriver" else "model"
    env = make_env()
    model = None
    if driver != "dummydriver":
        v.MODEL.update(name=driver, path=f"{v.TRAIN['folder']}/{driver}.zip")
        model = load_model(env)
        if model is None:
            return None
    outcomes, exit_times, rewards = Counter(), [], []
    for i in range(v.TRAIN["eval_episodes"]):
        obs, _ = env.reset(seed=FIRST_SEED + i)
        done, steps, total = False, 0, 0.0
        while not done and steps < MAX_STEPS:
            obs, reward, done, _, _ = env.step(ask_model(model, obs)[0])
            steps, total = steps + 1, total + reward
        out = env.outcome() or "stuck"
        outcomes[out] += 1
        rewards.append(total)
        if out == "made exit":
            exit_times.append(steps)
    n = v.TRAIN["eval_episodes"]
    pct = lambda k: f"{100 * outcomes[k] / n:.0f}%"
    avg_time = f"{sum(exit_times) / len(exit_times):.1f}" if exit_times else "-"
    return [driver, pct("made exit"), pct("crashed"), pct("missed exit"), pct("stuck"), avg_time, f"{sum(rewards) / n:.2f}"]


if __name__ == "__main__":
    header = ["driver", "made exit", "crashed", "missed exit", "stuck", "avg s to exit", "avg reward"]
    rows = [r for r in (score(d) for d in ["dummydriver"] + v.TRAIN["models"]) if r]
    table = "\n".join("| " + " | ".join(r) + " |" for r in [header, ["---"] * len(header)] + rows)
    text = f"{v.TRAIN['eval_episodes']} episodes each, same roads for everyone (seeds {FIRST_SEED}+)\n\n{table}\n"
    (HERE / "results.md").write_text(text)
    print(text)
