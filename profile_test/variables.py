YOU = {
    "personality": "normal",  # "random" or a PERSONALITIES name
    "driver": "model",  # "dummydriver" or "model"
    "start_speed": 22,  # m/s
    "model_speeds": [0, 5, 10, 15, 20, 25, 30, 35],  # m/s, 0 lets it stop and wait
    "cars_seen": 8,
}

# TODO make it change based on what model is being run. e.g. discretemeta for dqn, continuous for ppo, sac, etc.
ACTION = {
    "type": "ContinuousAction",
    "steering_range": (-0.1, 0.1),  # try 0.05-0.15
}
# there might be other options? for continuousaction
# {
#     "type": "DiscreteMetaAction", 
#     "target_speeds": [0, 5, 10, 15, 20, 25, 30, 35]
# }

DEFAULT_DRIVER = {
    "chance": 0.7,  # spawn weight vs other personalities
    "speed": 1.05,  # fraction of speed limit
    "follows_limit": 0.4,  # chance it never speeds
    "speed_wiggle": 2,  # random +/- m/s per car
    "follow_time": 1.5,  # s
    "stop_gap": 3,  # m, too low and it cant steer out
    "accel": 3.0,  # m/s^2
    "brake": 8.0,  # m/s^2
    "politeness": 0.2,  # 0 to 1
    "lane_change_eagerness": 0.2,  # speed gain needed to change lanes
    "sight": 120,  # m, when stopped
    "sight_speed_loss": 0.25,  # share of sight lost at the speed limit
    "dodge": True,  # also passes slow cars, not just stopped ones
    "keep_right": False,
    "hurry": 0.3,  # 0 to 1, up to 20% faster and 2x closer
    "exit_priority": 0.3,  # 0 to 1, smaller gaps taken for the exit
    "plan_ahead": 350,  # m before the exit it starts moving over
    "color": (100, 200, 255),
}

# only overrides of DEFAULT_DRIVER
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
    "speed_limit": 29,  # m/s, ~65 mph
    "left_lane_bonus": 5,  # m/s
    "right_lane_bonus": -3,  # m/s
    "ramp_and_exit_speed": 22,  # m/s
}

TRAFFIC = {
    "density": (8, 20),  # cars per km per lane
    "around_you": (120, 220),  # m behind, ahead. cars outside get removed
    "exit_chance": 0.3,
    "spawn_spacing": 20,  # m
    "spawn_speed": 22,  # m/s, same for all so nobody rear ends at spawn
}

ROAD = {
    "exit_side": "random",  # "random", "left", or "right"
    "lanes": (3, 3),
    "ramp_length": (120, 230),  # m
    "exit_distance": (250, 500),  # m
    "exit_lane_length": (80, 120),  # m
    "exit_ramp_length": 80,  # m, halfway counts as made exit
}

REWARDS = {
    "crash": -12,  # must beat the worst honest try or it gives up on the ramp
    "made_exit": 8,  # must beat cruising past or PPO/A2C never change lanes
    "missed_exit": -4,  # must stay above crash
    "caused_crash": -3,  # per car that wrecks within 30 m
    "good_speed": (18, 32),  # m/s
    "speed": 0.2,  # per s inside good_speed
    "off_speed": -0.03,  # per s per m/s outside good_speed
    "lane_change": -0.05,
    "on_ramp": -0.01,  # times seconds on the ramp, per s
    "progress": 0.005,  # per m toward the exit
    "toward_exit_lane": 0.5,  # per lane, negative when moving away
    "stopped_ok_within": 25,  # m, slow is free if blocked this close
    "per_second": -0.02,  # full timeout must cost less than a crash
    "off_road": -12,
    "spun_out": -12,
    "spin_limit": 1.57,    # rad, ~90 degrees off the lane direction ends the life
    "steer": -0.05,
    "steer_change": -0.05,
    "heading_error": -0.1, # per rad
}

VIDEO = {
    "episodes": 3,
    "fps": 15,  # also physics steps per second
    "speed_up": 2,
    "end_card_seconds": 1,
    "show_window": True,
    "give_up_seconds": 60,  # skips stuck episodes when no window
    "you_color": (50, 200, 0),
}

MODEL = {
    "name": "DQN",
    "library": "stable_baselines3",  # or "sb3_contrib"
    "path": "trained/DQN.zip",  # main.py --model NAME overrides
}

TRAIN = {
    "show_ui": False,  # only used when train.py cant prompt
    # "models": ["DQN", "PPO", "A2C"],
    "models": ["PPO"],
    "max_steps": 300_000,  # per model
    "goal_exit_rate": 0.9,  # stop early at this exit rate over last 100 lives
    "envs": 5,  # parallel envs per model
    "gamma": 0.99,  # lower makes the exit too far away to matter
    "seed": 0,
    "max_life_seconds": 120,  # training only
    "physics_per_second": 5,  # training only, ~3x faster than 15
    "watch_port": 47800,  # one port per model after this
    "folder": "trained",
    "checkpoint_every": 1000,  # steps
    "eval_episodes": 30,
    "settings": {  # passed to stable baselines
        "DQN": {"learning_rate": 5e-4, "buffer_size": 50_000, "learning_starts": 1000, "batch_size": 64,
                "target_update_interval": 500, "exploration_fraction": 0.3},
        "PPO": {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.01},  # ent_coef: else never changes lanes
        "A2C": {"learning_rate": 3e-4, "n_steps": 64, "ent_coef": 0.01},  # ent_coef: else locks into crashing
    },
    "policy_kwargs": {
        "PPO": {"log_std_init": -2.0}, 
        "A2C": {"log_std_init": -2.0}
    },  # quieter start
}

# fill in defaults and catch typos in personality keys
for _name, _changes in PERSONALITIES.items():
    _typos = set(_changes) - set(DEFAULT_DRIVER)
    assert not _typos, f"{_name} has variables that dont exist: {_typos}"
    PERSONALITIES[_name] = {**DEFAULT_DRIVER, **_changes}
