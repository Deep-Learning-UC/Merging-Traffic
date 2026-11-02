# My(Eli) test

my sandbox. doesnt touch the main code. just testing driver personalities, not the full project.

## files

- `variables.py` - change everything here
- `environment.py` - the world the model trains in. road, traffic and the `Driver` rules they drive with, rewards, endings
- `personality_test.py` - checks the variables actually do stuff
- `main.py` - live window + saves videos to `saved_videos/`

## setup

if on linux
```bash
cd profile_test
python -m venv .venv
.venv/bin/pip install gymnasium highway-env stable-baselines3 imageio imageio-ffmpeg pillow
source .venv/bin/activate
```

not on linux just `cd profile_test && pip install gymnasium highway-env stable-baselines3 imageio imageio-ffmpeg pillow`

turn the live window on or off with `VIDEO["show_window"]` in `variables.py`.

## run

```bash
python personality_test.py
python main.py
python main.py --realtime
python main.py --no-window --episodes 5 --seed 3
```

## notes

- random road every episode. start on the right ramp, merge, take the exit (left or right, see `ROAD["exit_side"]`).
- no time limit. ends as made exit, missed exit, or crashed.
- `YOU["driver"]`: `"dummydriver"` = the same `Driver` rules the traffic uses, `"model"` = whatever `MODEL` points at. only `"model"` is deep learning.
- what each personality variable does is commented in `variables.py`.
- highway-env secretly caps lanes at 20 m/s. turned that off, `LANES["speed_limit"]` is the real one.
- wrecks stay on the road. if they block every lane its a dead end, the test calls it stuck and the window has a Skip button.
- every `main.py` run is a new random seed. it prints it, `--seed` replays one.
- `model.zip` is connors DQN, 20k steps on plain highway. it loads but never takes the exit.

## plans

goal from the main readme: find the model that merges on and off the highway as fast as possible without crashing. teacher wants us to benchmark models, including 2025/2026 ones. so only stuff that gets us there:

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
