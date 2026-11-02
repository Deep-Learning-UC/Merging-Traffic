# My(Eli) test

my sandbox. doesnt touch the main code. just testing driver personalities, not the full project.

## files

- `variables.py` - change everything here
- `environment.py` - the world the model trains in. road, traffic and the `Driver` rules they drive with, rewards, endings
- `personality_test.py` - checks the variables actually do stuff
- `main.py` - live window + saves videos to `saved_videos/`. `--model PPO` records a trained model
- `train.py` - trains DQN, PPO, A2C at the same time on this road, saves to `trained/`
- `watch.py` - live time lapse of the actual training lives + exit rate chart
- `evaluate.py` - scores every model and the dummydriver on the same unseen roads, writes `results.md`

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
python train.py          # all 3 models, stops when 90% of the last 100 lives make the exit
python watch.py          # watch it learn, run while train.py is going
python evaluate.py       # after training
python main.py --model PPO
```

## notes

- random road every episode. start on the right ramp, merge, take the exit (left or right, see `ROAD["exit_side"]`).
- no time limit. ends as made exit, missed exit, or crashed.
- `YOU["driver"]`: `"dummydriver"` = the same `Driver` rules the traffic uses, `"model"` = whatever `MODEL` points at. only `"model"` is deep learning.
- what each personality variable does is commented in `variables.py`.
- highway-env secretly caps lanes at 20 m/s. turned that off, `LANES["speed_limit"]` is the real one.
- wrecks stay on the road. if they block every lane its a dead end, the test calls it stuck and the window has a Skip button.
- every `main.py` run is a new random seed. it prints it, `--seed` replays one.
- `model.zip` is connors DQN, 20k steps on plain highway. it doesnt fit this road anymore (different input), main.py skips it.
- the model sees 8 cars around it (front and behind), the ramp block, the exit side, and how far the exit is. highway-env's default only saw 5 cars in front.
- the model can pick 0 to 35 m/s (`YOU["model_speeds"]`). highway-env's default was only 20/25/30 so slower did nothing.
- training runs physics at 5 steps per second (~3x faster than 15). training lives get cut at 120s so a parked model cant stall it.
- rewards: crash -5 is the worst, exit +3. small costs for time, sitting still, and lane changes. a full timeout still costs less than a crash.

## plans

goal from the main readme: find the model that merges on and off the highway as fast as possible without crashing. teacher wants us to benchmark models, including 2025/2026 ones. so only stuff that gets us there:

- match the MVP road. 3 lanes, 1 km between ramp and exit.
- move this into the team `environment.py` so everyone trains on the same thing.
- ~~train.py, train DQN/PPO/A2C on the same road and rewards~~ done.
- ~~score them the same way~~ done with `evaluate.py`. still only 1 training seed each, should be 5+.
- add a 2026 model to the benchmark.
- record videos of each model. meetings said recordings are the main output.
- MVP draft 11/10, tested 11/15, final presentation 11/30 to 12/4.

## model pick

- DQN - easy, already in SB3, already have one. **going with this.**
- PPO - same library, more stable, slower to train. good second.
- PPO+ - 2026 paper with code but too much work for MVP.
