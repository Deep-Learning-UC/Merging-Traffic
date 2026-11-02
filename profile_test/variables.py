YOU = {
    "personality": "normal",  # "random" or a personality name like "slow"
    "driver": "dummydriver",  # "dummydriver" (Driver rules in environment.py) or "model" (MODEL below)
    "start_speed": 22,  # m/s
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
    "min_cars": 15,
    "max_cars": 40,
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
    "speed": 0.2,  # max per step
    "speed_range": (20, 30),  # m/s. bonus goes from 0 to full across this range
    "lane_change": -0.05,  # per lane change
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
    "path": "../highway_dqn/model.zip",  # relative to this folder
}

# fills in every personality with the defaults for anything it didnt change.
# also catches typos, a misspelled name would just get ignored otherwise
for _name, _changes in PERSONALITIES.items():
    _typos = set(_changes) - set(DEFAULT_DRIVER)
    assert not _typos, f"{_name} has variables that dont exist: {_typos}"
    PERSONALITIES[_name] = {**DEFAULT_DRIVER, **_changes}