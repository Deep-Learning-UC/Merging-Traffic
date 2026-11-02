# profile test

my sandbox. doesnt touch the main code. just testing driver personalities, not the full project.

## files

- `variables.py` - change everything here
- `environment.py` - the world the model trains in. road, traffic, rewards, endings
- `simulated_drivers.py` - the hand written driving rules. traffic always uses them, you only with `"dummydriver"`
- `personality_test.py` - checks the variables actually do stuff
- `main.py` - live window + saves videos to `saved_videos/`
- `viewer.py` - the tkinter window

## setup

```bash
cd profile_test
python -m venv .venv
.venv/bin/pip install gymnasium highway-env stable-baselines3 imageio imageio-ffmpeg pillow
source .venv/bin/activate
```

not on linux just `pip install gymnasium highway-env stable-baselines3 imageio imageio-ffmpeg pillow`

turn the live window on or off with `VIDEO["show_window"]` in `variables.py`.

## run

```bash
python personality_test.py
python main.py
python main.py --realtime
python main.py --no-window --episodes 5 --seed 3
```

## notes

- random road every episode. start on the right ramp, merge, take the exit.
- exit can be left or right. `ROAD["exit_side"]` is `"random"`, `"left"`, or `"right"`.
- no time limit. ends as made exit, missed exit, or crashed. videos also end if you hit Skip.
- set personality for dummy driver
- left lane is faster. lane speeds are in `LANES`.
- highway-env secretly caps lanes at 20 m/s. thats why everything was slow before. turned that off.
- `speed` is a fraction of the speed limit. `follows_limit` is the chance a driver never goes over it, so some normal drivers speed and some dont.
- `exit_priority` = crash vs exit tradeoff. low misses exits, high crashes more. only shows up in heavy traffic now.
- `hurry` = how bad they want to get there. faster, closer, passes more.
- `keep_right` keeps slow people out of the fast lane.
- `sight` shrinks the faster you go (`sight_speed_loss`). `stop_gap` is the gap when stopped, highway gap comes from `follow_time`.
- crashed cars stay on the road as red wrecks. everyone goes around stopped cars and wrecks, unless theyre lined up for their exit, then they wait.
- if enough wrecks block every lane its a dead end. the test calls that stuck, the video window has a Skip button for it.
- the ramp has a block at the end so cars stop and wait for a gap instead of getting shoved in.
- every `main.py` run is a new random test. it prints the seed, use `--seed` to replay one.
- rewards get calculated and show in the video. nothing trains on them yet.
- `model.zip` is connors DQN, 20k steps on plain highway. it loads and fits but never takes the exit.
- `YOU["driver"]` is `"dummydriver"` or `"model"`. dummydriver is just rules from `simulated_drivers.py`, not deep learning or any models. only `"model"` is deep learning.

## plans

goal from the main readme: find the model that merges on and off the highway as fast as possible without crashing. teacher wants us to benchmark models, including 2025/2026 ones. so only stuff that gets us there:

- ~~add a fast personality~~ done. normal, slow, fast, plus a rare reckless one.
- match the MVP road. 3 lanes, 1 km between ramp and exit.
- move this into the team `environment.py` so everyone trains on the same thing.
- add `train.py` so the DQN actually learns on this road. right now nothing trains.
- train the other models on the exact same road and rewards. DQN, PPO, and at least one 2026 one.
- score them all the same way. exit rate, crash rate, time to exit, 5+ seeds.
- record videos of each model. meetings said recordings are the main output.
- MVP draft 11/10, tested 11/15, final presentation 11/30 to 12/4.

## model pick

- DQN - easy, already in SB3, already have one. **going with this.**
- PPO - same library, more stable, slower to train. good second.
- PPO+ - 2026 paper with code but too much work for MVP.
- since were benchmarking, DQN is the first one we get working, not the only one.
