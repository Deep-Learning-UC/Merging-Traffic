# scores each model and the dummydriver on the same unseen roads, writes results.md
from collections import Counter
from pathlib import Path

import variables as v
from environment import make_env
from main import ask_model, load_model, play

HERE = Path(__file__).resolve().parent
# far from TRAIN["seed"] so no model is tested on a training road
FIRST_SEED = 10_000


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
        out, steps, total = play(env, obs, lambda o: ask_model(model, o)[0])
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
