# the world the model trains in: random road, traffic and how it drives, rewards, and how episodes end
import numpy as np
from gymnasium import spaces
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle

from variables import LANES, PERSONALITIES, REWARDS, ROAD, TRAFFIC, TRAIN, VIDEO, YOU

W = StraightLane.DEFAULT_WIDTH
C, S, N = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
WRECK_COLOR = (240, 80, 80)


# faster = less sight, but never below 30% of it
def sight_at(p, speed):
    return p["sight"] * max(0.3, 1 - p["sight_speed_loss"] * speed / LANES["speed_limit"])


# every non model car is one of these. its highway-env's IDM driver with our personality knobs bolted on
class Driver(IDMVehicle):
    # applies a personality from variables.py to this car
    def setup(self, name, rng, wants_exit, env):
        p = PERSONALITIES[name]
        self.p, self.personality, self.wants_exit, self.env = p, name, wants_exit, env
        # hurry: up to 20% faster, half the following gap, and passes for less reason
        h = p["hurry"]
        self.base_speed = (p["speed"] * LANES["speed_limit"] + rng.uniform(-p["speed_wiggle"], p["speed_wiggle"])) * (1 + h / 5)
        self.obeys_limit = rng.random() < p["follows_limit"]
        self.speed = min(self.speed, self.base_speed)
        # these are highway-env's own IDM/MOBIL knobs, we just fill them from the personality
        self.TIME_WANTED, self.DISTANCE_WANTED = p["follow_time"] * (1 - h / 2), p["stop_gap"] + self.LENGTH
        self.COMFORT_ACC_MAX, self.COMFORT_ACC_MIN, self.ACC_MAX = p["accel"], -p["brake"], p["brake"]
        self.POLITENESS, self.LANE_CHANGE_MIN_ACC_GAIN = p["politeness"], p["lane_change_eagerness"] * (1 - h)
        self.color = p["color"]
        return self

    def act(self, action=None):
        self.target_speed = self.lane_speed()
        super().act(action)

    def front(self):
        return self.road.neighbour_vehicles(self, self.lane_index)[0]

    # cant brake for what you cant see
    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        if ego_vehicle and front_vehicle and abs(ego_vehicle.lane_distance_to(front_vehicle)) > sight_at(self.p, self.speed):
            front_vehicle = None
        normal = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)
        # pulling out from behind a stopped car? idm would sit there forever, so creep forward while steering out
        pulling_out = ego_vehicle is self and self.lane_index != self.target_lane_index
        if pulling_out and front_vehicle and front_vehicle.speed < 2 and self.lane_distance_to(front_vehicle) > self.LENGTH + 0.3:
            return max(normal, min(3 - self.speed, 1))
        return normal

    # needs a real gap that grows with speed. higher exit_priority = smaller gap is "fine"
    def gap_ok(self, lane):
        front, rear = self.road.neighbour_vehicles(self, lane)
        k = 1 - self.p["exit_priority"]
        if front and self.lane_distance_to(front) < max(self.desired_gap(self, front) * k, self.LENGTH + 1):
            return False
        return not rear or rear.lane_distance_to(self) >= max(self.desired_gap(rear, self) * k, self.LENGTH + 1)

    # -1 = go left, 1 = go right, 0 = stay put, None = do whatever normal driving says
    def side_needed(self):
        f, t, i = self.lane_index
        left = self.env.exit_side == "left"
        if (f, t) == ("a", "b") and i == self.env.lanes:
            return -1
        if self.wants_exit and t in "bcd" and self.position[0] > self.env.split_x - self.p["plan_ahead"]:
            want = self.env.exit_lane if t == "d" else (0 if left else self.env.lanes - 1)
            return int(np.sign(want - i))
        if not self.wants_exit and (f, t, i) == ("c", "d", self.env.exit_lane):
            return 1 if left else -1
        if self.p["keep_right"] and t in "bcde" and i < self.env.lanes - 1 + (t == "d" and left):
            return 1
        return None

    # same rule for wrecks and dead stop traffic: if its stopped and close, find a lane around it.
    # dodge personalities also go around cars that are just way slower
    def go_around_side(self, car):
        if not car or self.lane_distance_to(car) > min(self.desired_gap(self, car) + self.LENGTH, sight_at(self.p, self.speed)):
            return None
        if car.speed < 2 or (self.p["dodge"] and car.speed < self.speed - 3):
            for side in (-1, 1):
                lane = self.lane_beside(side)
                if lane and self.gap_ok(lane):
                    return side
        return None

    # the lane next to me (side -1 = left, 1 = right), or None if there isnt one
    def lane_beside(self, side):
        f, t, i = self.lane_index
        return (f, t, i + side) if 0 <= i + side < len(self.road.network.graph[f][t]) else None

    # decides lane changes. our rules first, highway-env's normal MOBIL if none of ours apply
    def change_lane_policy(self):
        car = self.front()
        if self.lane_index != self.target_lane_index:
            # already pulling out around a stopped car? dont let highway-env's abort rule cancel it.
            # wrecks that crashed mid lane change kept cancelling it and cars sat there forever
            if not (car and car.speed < 2):
                super().change_lane_policy()
            return
        side = self.side_needed()
        # already in the exit lane line? wait like normal traffic. unless its a wreck, those never move
        if side == 0 and not (car and car.crashed):
            return
        if not side:
            side = self.go_around_side(car)
        if side is None:
            return super().change_lane_policy()
        lane = self.lane_beside(side)
        if lane and self.gap_ok(lane):
            self.target_lane_index = lane

    # left lane fast, right lane slow, ramp and exit lane slowest. spent way too long getting the lane numbers right
    def lane_speed(self):
        env, (f, t, i) = self.env, self.lane_index
        if (f, t) == ("a", "b") and i == env.lanes or (f, t) == ("c", "d") and i == env.exit_lane or (f, t) == ("d", "x"):
            return min(self.base_speed, LANES["ramp_and_exit_speed"])
        if (f, t) == ("c", "d") and env.exit_side == "left":
            i -= 1
        right = i / max(env.lanes - 1, 1)
        speed = self.base_speed + LANES["left_lane_bonus"] * (1 - right) + LANES["right_lane_bonus"] * right
        return min(speed, LANES["speed_limit"]) if self.obeys_limit else speed


# the world the model trains in. highway-env runs the physics, we just build the road, cars, rewards, and endings
class MergeExitEnv(AbstractEnv):
    # highway-env's defaults only let the model pick 20, 25, or 30 m/s and only see cars in front.
    # so "slower" at 20 did nothing and it changed lanes into cars it couldnt see. this fixes both
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config["observation"] = {"type": "Kinematics", "vehicles_count": YOU["cars_seen"], "see_behind": True, "include_obstacles": True}
        config["action"] = {"type": "DiscreteMetaAction", "target_speeds": YOU["model_speeds"]}
        config["simulation_frequency"] = TRAIN["physics_per_second"]
        return config

    # the model sees the 5 nearest cars plus 2 numbers: exit side (1 left, -1 right) and meters to the exit / 1000.
    # without these it has no clue where the exit is and could never learn to take it
    def define_spaces(self):
        super().define_spaces()
        self.observation_space = spaces.Box(-np.inf, np.inf, (int(np.prod(self.observation_space.shape)) + 3,), np.float32)

    # lanes between you and the exit lane. negative = exit lane is to your left
    def lanes_to_exit(self):
        exit_y = -W if self.exit_side == "left" else self.lanes * W
        return (exit_y - self.vehicle.position[1]) / W

    def observe(self, cars):
        exit_info = [1.0 if self.exit_side == "left" else -1.0, (self.split_x - self.vehicle.position[0]) / 1000, self.lanes_to_exit() / 4]
        return np.concatenate([np.ravel(cars), exit_info]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observe(obs), info

    # new random road and traffic every episode
    def _reset(self):
        r = lambda key: self.np_random.uniform(*ROAD[key])
        self.lanes = int(self.np_random.integers(ROAD["lanes"][0], ROAD["lanes"][1] + 1))
        side = ROAD["exit_side"]
        self.exit_side = side if side != "random" else str(self.np_random.choice(["left", "right"]))
        self.exit_lane = 0 if self.exit_side == "left" else self.lanes
        self.ramp_x = r("ramp_length")
        self.exit_x = self.ramp_x + r("exit_distance")
        self.split_x = self.exit_x + r("exit_lane_length")
        self.end_x = self.split_x + 200
        self.density = self.np_random.uniform(*TRAFFIC["density"])
        self.make_road()
        self.make_cars()

    # lane 0 is the far left. a left exit lane sneaks in as lane 0 on c-d so everyone else shifts over one.
    # a right exit lane just gets tacked on the end
    def make_road(self):
        net, n, left = RoadNetwork(), self.lanes, self.exit_side == "left"
        lines = lambda i, count: [C if i == 0 else N, C if i == count - 1 else S]
        # each road piece: from, to, start x, end x, number of lanes, which lane number is furthest left
        sections = [("a", "b", 0, self.ramp_x, n + 1, 0), ("b", "c", self.ramp_x, self.exit_x, n, 0),
                    ("c", "d", self.exit_x, self.split_x, n + 1, -left), ("d", "e", self.split_x, self.end_x, n, 0)]
        # speed_limit=None turns off highway-env's hidden 20 m/s cap. Driver.lane_speed() handles limits instead
        for f, t, x0, x1, count, first in sections:
            for i in range(count):
                y = (i + first) * W
                net.add_lane(f, t, StraightLane([x0, y], [x1, y], line_types=lines(i, count), speed_limit=None))
        y = -W if left else n * W
        end = [self.split_x + ROAD["exit_ramp_length"], y - 40 if left else y + 40]
        net.add_lane("d", "x", StraightLane([self.split_x, y], end, line_types=[C, C], speed_limit=None))
        self.road = Road(network=net, np_random=self.np_random, record_history=False)
        # little block at the end of the on ramp so cars stop and wait instead of getting shoved in. same trick merge-v0 uses
        self.road.objects.append(Obstacle(self.road, net.get_lane(("a", "b", n)).position(self.ramp_x, 0)))

    def pick(self, you=False):
        if you and YOU["personality"] != "random":
            return YOU["personality"]
        names = list(PERSONALITIES)
        chances = np.array([PERSONALITIES[k]["chance"] for k in names], dtype=float)
        return str(self.np_random.choice(names, p=chances / chances.sum()))

    # you on the ramp, then random traffic on the highway
    def make_cars(self):
        ramp = self.road.network.get_lane(("a", "b", self.lanes))
        assert YOU["driver"] in ("dummydriver", "model"), 'YOU["driver"] has to be "dummydriver" or "model"'
        if YOU["driver"] == "model":
            me = self.action_type.vehicle_class(self.road, ramp.position(30, 0), speed=YOU["start_speed"])
            me.personality = self.pick(you=True)
        else:
            me = Driver(self.road, ramp.position(30, 0), speed=YOU["start_speed"]).setup(self.pick(you=True), self.np_random, True, self)
        me.color = VIDEO["you_color"]
        self.road.vehicles.append(me)
        self.vehicle = me
        self.last_x, self.wrecks_seen, self.ramp_seconds = me.position[0], set(), 0
        self.fill_traffic(start=True)

    # keeps every lane at the traffic density around you so the flow never dries up. at the start it fills the
    # whole stretch, after that new cars only appear at the far edges, out of view, so nothing pops in on screen
    def fill_traffic(self, start=False):
        behind, ahead = TRAFFIC["around_you"]
        me_x = self.vehicle.position[0]
        lo, hi = max(0, me_x - behind), min(self.end_x - 30, me_x + ahead)
        want = round(self.density * (hi - lo) / 1000)
        for i in range(self.lanes):
            taken = [v.position[0] for v in self.road.vehicles if abs(v.position[1] - i * W) < W]
            have = sum(lo <= x <= hi for x in taken)
            for x in (self.np_random.uniform(lo, hi, want * 3) if start else (lo, hi)):
                if have >= want:
                    break
                if all(abs(x - o) > TRAFFIC["spawn_spacing"] for o in taken):
                    car = Driver(self.road, np.array([x, i * W]), speed=TRAFFIC["spawn_speed"])
                    self.road.vehicles.append(car.setup(self.pick(), self.np_random, self.np_random.random() < TRAFFIC["exit_chance"], self))
                    # new cars at the edges come in at their own lane speed so the ones behind actually catch up and pass
                    if not start:
                        car.speed = car.lane_speed()
                    taken.append(x)
                    have += 1

    # made exit, missed exit, crashed, or None while still driving
    def outcome(self):
        me = self.vehicle
        if me.crashed:
            return "crashed"
        if me.lane_index[:2] == ("d", "x"):
            return "made exit" if me.lane.local_coordinates(me.position)[0] > ROAD["exit_ramp_length"] / 2 else None
        if me.lane_index[:2] == ("d", "e") and me.position[0] > self.split_x + 40:
            return "missed exit"
        return None

    # crashed cars stay put and turn red so everyone else has to deal with them.
    # ugh cars kept driving off the end of the map forever, so anyone past the end gets removed
    def step(self, action):
        # what the model sees shrinks with speed too
        self.PERCEPTION_DISTANCE = sight_at(PERSONALITIES[self.vehicle.personality], self.vehicle.speed)
        obs, reward, done, truncated, info = super().step(action)
        for v in self.road.vehicles:
            if v.crashed:
                v.color = WRECK_COLOR
        ramp_end = self.split_x + ROAD["exit_ramp_length"] - 10
        me_x = self.vehicle.position[0]
        keep = (me_x - TRAFFIC["around_you"][0] - 50, me_x + TRAFFIC["around_you"][1] + 50)
        self.road.vehicles = [v for v in self.road.vehicles if v is self.vehicle or not (
            v.position[0] > self.end_x or not keep[0] < v.position[0] < keep[1] or v.lane_index[:2] == ("d", "x") and v.position[0] > ramp_end)]
        self.fill_traffic()
        return self.observe(obs), reward, done, truncated, info

    # per step score from REWARDS, what a model learns from
    def _reward(self, action):
        me = self.vehicle
        # speed: bonus inside the good range, a cost for every m/s outside it. too slow is fine when something is right in front
        lo, hi = REWARDS["good_speed"]
        front = self.road.neighbour_vehicles(me, me.lane_index)[0]
        blocked = front is not None and me.lane_distance_to(front) < REWARDS["stopped_ok_within"]
        too_slow = 0 if blocked else max(lo - me.speed, 0)
        r = REWARDS["speed"] if lo <= me.speed <= hi else REWARDS["off_speed"] * (too_slow + max(me.speed - hi, 0))
        # the longer you sit on the on ramp the more each second there costs
        if me.lane_index[:2] == ("a", "b") and me.lane_index[2] == self.lanes:
            self.ramp_seconds += 1
            r += REWARDS["on_ramp"] * self.ramp_seconds
        # actions 0 and 2 are lane left and lane right
        r += REWARDS["lane_change"] * (action in (0, 2)) + REWARDS["per_second"]
        # other cars that just wrecked close to you count as you causing it
        new_wrecks = [c for c in self.road.vehicles if c.crashed and c is not me and id(c) not in self.wrecks_seen]
        self.wrecks_seen |= {id(c) for c in new_wrecks}
        r += REWARDS["caused_crash"] * sum(abs(c.position[0] - me.position[0]) < 30 for c in new_wrecks)
        # learning hints: a little for every meter closer to the exit, and a cost for being lanes away from it near the end
        r += REWARDS["progress"] * (self.vehicle.position[0] - self.last_x)
        self.last_x = self.vehicle.position[0]
        if self.vehicle.position[0] > self.split_x - REWARDS["wrong_lane_within"]:
            r += REWARDS["wrong_lane"] * abs(self.lanes_to_exit())
        return r + {"crashed": REWARDS["crash"], "made exit": REWARDS["made_exit"], "missed exit": REWARDS["missed_exit"]}.get(self.outcome(), 0)

    # so training can count exits and crashes, not just reward
    def _info(self, obs, action):
        info = super()._info(obs, action)
        info["outcome"] = self.outcome()
        return info

    def _is_terminated(self):
        return self.outcome() is not None

    # no time limit unless max_seconds is set, train.py sets it so a parked model cant stall training forever
    def _is_truncated(self):
        return self.config.get("max_seconds") is not None and self.time >= self.config["max_seconds"]


def make_env(render_mode=None, **config):
    env = MergeExitEnv(render_mode=render_mode)
    env.configure(config)
    return env
