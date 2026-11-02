YOU = {
    "personality": "normal",  # "random" or a personality name like "slow"
    "driver": "model",  # "dummydriver" (Driver rules in environment.py) or "model" (MODEL below)
    "start_speed": 22,  # m/s
    "model_speeds": [0, 5, 10, 15, 20, 25, 30, 35],  # m/s the model can pick between with faster/slower. 0 lets it stop and wait
    "cars_seen": 8,  # how many nearby cars the model sees, in front and behind
}

# every driver starts with these. this is basically the normal driver
DEFAULT_DRIVER = {
    "chance": 0.7,  # how often this personality spawns, compared to the others. higher = more of them on the road
    "speed": 1.05,  # speed it wants, as a fraction of the speed limit. 1.0 = right at the limit, higher = faster
    "follows_limit": 0.4,  # chance it never goes over the limit. 1 = always obeys, 0 = always speeds if it wants to
    "speed_wiggle": 2,  # random +/- m/s added per car. higher = cars of the same personality vary more
    "follow_time": 1.5,  # seconds of gap kept behind the car ahead at speed. lower = tailgates, higher = big gaps
    "stop_gap": 3,  # bumper gap in meters when stopped. lower = packs in tighter, too low and it cant steer out
    "accel": 3.0,  # how quickly it speeds up (m/s^2). higher = jumps off the line and closes gaps fast
    "brake": 8.0,  # hardest it can brake (m/s^2). lower = needs more room to stop, more rear ends
    "politeness": 0.2,  # 0 to 1. how much it cares about slowing others when changing lanes. higher = lets people in
    "lane_change_eagerness": 0.2,  # speed gain needed before it changes lanes. lower = changes lanes constantly
    "sight": 120,  # meters it can see ahead when stopped. lower = notices stopped cars and wrecks later
    "sight_speed_loss": 0.25,  # share of sight lost at the speed limit. higher = sees way less when going fast
    "dodge": True,  # True = also goes around much slower cars, not just stopped ones. False = just brakes behind them
    "keep_right": False,  # True = moves back right whenever it can. keeps the left lane open for faster cars
    "hurry": 0.3,  # 0 to 1. higher = up to 20% faster, follows up to 2x closer, changes lanes for less reason
    "exit_priority": 0.3,  # 0 to 1. how small a gap it takes to make the exit. higher = makes more exits but crashes more
    "plan_ahead": 350,  # meters before the exit it starts moving over. lower = leaves it late, misses more exits
    "color": (100, 200, 255),  # color in the video
}

# each personality only lists what it changes from DEFAULT_DRIVER
PERSONALITIES = {
    "normal": {},
    "slow": {
        "chance": 0.3,
        "speed": 0.75,
        "follows_limit": 1.0,
        "speed_wiggle": 1,
        "follow_time": 2.5,
        "stop_gap": 4,
        "accel": 1.5,
        "brake": 7.0,
        "politeness": 0.6,
        "lane_change_eagerness": 0.6,
        "sight": 80,
        "sight_speed_loss": 0.15,
        "dodge": False,
        "keep_right": True,
        "hurry": 0.0,
        "exit_priority": 0.1,
        "plan_ahead": 500,
        "color": (255, 170, 60),
    },
    "fast": {
        "chance": 0.12,
        "speed": 1.12,
        "follows_limit": 0.1,
        "follow_time": 1.2,
        "accel": 4.0,
        "politeness": 0.1,
        "lane_change_eagerness": 0.1,
        "sight": 130,
        "sight_speed_loss": 0.3,
        "hurry": 0.6,
        "exit_priority": 0.4,
        "plan_ahead": 250,
        "color": (240, 110, 200),
    },
    "reckless": {
        "chance": 0.02,
        "speed": 1.2,
        "follows_limit": 0.0,
        "speed_wiggle": 8,
        "follow_time": 0.2,
        "accel": 8.0,
        "politeness": 0.01,
        "lane_change_eagerness": 0.04,
        "sight": 100,
        "sight_speed_loss": 0.45,
        "hurry": 0.9,
        "exit_priority": 0.9,
        "plan_ahead": 150,
        "color": (170, 90, 255),
    },
}

LANES = {
    "speed_limit": 29,  # m/s, about 65 mph
    "left_lane_bonus": 5,  # m/s
    "right_lane_bonus": -3,  # m/s, negative = slower. middle lanes blend between
    "ramp_and_exit_speed": 22,  # m/s
}

TRAFFIC = {
    "density": (8, 20),  # (min, max) cars per km per lane, random each episode. kept steady around you the whole drive
    "around_you": (120, 220),  # m behind and ahead of you where traffic is kept. cars further away either way get removed
    "exit_chance": 0.3,  # share of traffic that also takes the exit
    "spawn_spacing": 20,  # m
    "spawn_speed": 22,  # m/s, same for everyone so nobody rear ends at spawn
}

ROAD = {
    "exit_side": "random",  # "random", "left", or "right"
    "lanes": (3, 3),  # (min, max)
    "ramp_length": (120, 230),  # (min, max) m
    "exit_distance": (250, 500),  # (min, max) m
    "exit_lane_length": (80, 120),  # (min, max) m
    "exit_ramp_length": 80,  # m. halfway up counts as made exit
}

REWARDS = {
    "crash": -5,
    "made_exit": 3,
    "missed_exit": -1,
    "caused_crash": -3,  # for every other car that wrecks within 30 m of you
    "good_speed": (18, 32),  # m/s. about 40 to 72 mph
    "speed": 0.2,  # per second while inside good_speed
    "off_speed": -0.03,  # per second for every m/s too slow or too fast. too slow is free if something is right in front of you
    "lane_change": -0.05,  # per lane change
    "on_ramp": -0.01,  # times seconds spent on the on ramp, every second. grows the longer you stay so it merges instead of hitting the end
    "progress": 0.005,  # per meter driven toward the exit. a learning hint so it knows its getting closer
    "wrong_lane": -0.05,  # per second for each lane you are away from the exit lane, once the exit is close
    "wrong_lane_within": 400,  # m before the exit where wrong_lane starts counting
    "stopped_ok_within": 25,  # m. if a car or the ramp block is this close in front, going slow is free. waiting your turn is fine
    "per_second": -0.02,  # every second costs a bit so parking is bad. a full timeout (-2.4) still has to cost less than a crash
}

VIDEO = {
    "episodes": 3,  # videos per run
    "fps": 15,  # also physics steps per second
    "speed_up": 2,  # 2 = plays twice as fast
    "end_card_seconds": 1,
    "show_window": True,  # False = just save the videos
    "give_up_seconds": 60,  # s. auto skips stuck episodes when theres no window
    "you_color": (50, 200, 0),
}

MODEL = {
    "name": "DQN",  # "DQN", "PPO", "A2C", etc. also shown on the video
    "library": "stable_baselines3",  # "sb3_contrib" for the extra ones
    "path": "trained/DQN.zip",  # relative to this folder. main.py --model NAME overrides name and path
}

TRAIN = {
    "show_ui": False,  # only used when train.py cant ask you (like a server). True = open the watcher, False = no window, fastest
    "models": ["DQN", "PPO", "A2C"],  # train.py trains these at the same time, evaluate.py scores them
    "max_steps": 300_000,  # per model. training stops here even if it never hits the goal
    "goal_exit_rate": 0.9,  # stops early once this share of the last 100 lives made the exit
    "envs": 5,  # copies of the road each model trains on at once. more = faster but more cpu cores
    "gamma": 0.95,  # how much future reward counts. the exit is ~40 steps out, too low and it never learns to care
    "seed": 0,
    "max_life_seconds": 120,  # training only. a model that parks gets cut off here as "timed out". tests have no limit
    "physics_per_second": 5,  # physics steps per simulated second. 5 trains ~3x faster than 15, videos still use VIDEO fps
    "watch_port": 47800,  # watch.py listens here for the live training lives, one port per model after this
    "folder": "trained",  # where train.py saves models, relative to this folder
    "checkpoint_every": 1000,  # steps between saves while training, watch.py shows the newest one
    "eval_episodes": 30,  # episodes per model in evaluate.py. every model gets the same roads
    "settings": {  # extra settings per model, passed straight to stable baselines
        "DQN": {"learning_rate": 5e-4, "buffer_size": 50_000, "learning_starts": 1000, "batch_size": 64,
                "target_update_interval": 500, "exploration_fraction": 0.3},
        "PPO": {"learning_rate": 3e-4, "n_steps": 256, "batch_size": 64},
        "A2C": {"learning_rate": 3e-4, "n_steps": 16, "ent_coef": 0.01},  # ent_coef keeps it exploring. without it A2C locked into crashing
    },
}

# fills in every personality with the defaults for anything it didnt change.
# also catches typos, a misspelled name would just get ignored otherwise
for _name, _changes in PERSONALITIES.items():
    _typos = set(_changes) - set(DEFAULT_DRIVER)
    assert not _typos, f"{_name} has variables that dont exist: {_typos}"
    PERSONALITIES[_name] = {**DEFAULT_DRIVER, **_changes}