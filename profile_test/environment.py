import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle

from simulated_drivers import Driver, sight_at
from variables import PERSONALITIES, REWARDS, ROAD, TRAFFIC, VIDEO, YOU

W = StraightLane.DEFAULT_WIDTH
C, S, N = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
WRECK_COLOR = (240, 80, 80)


# the world the model trains in. highway-env runs the physics, we just build the road, cars, rewards, and endings
class MergeExitEnv(AbstractEnv):
    # new random road every episode
    def _reset(self):
        r = lambda key: self.np_random.uniform(*ROAD[key])
        # random lane count, exit side, and where the ramp, exit, and split are
        self.lanes = int(self.np_random.integers(ROAD["lanes"][0], ROAD["lanes"][1] + 1))
        side = ROAD["exit_side"]
        self.exit_side = side if side != "random" else str(self.np_random.choice(["left", "right"]))
        self.exit_lane = 0 if self.exit_side == "left" else self.lanes
        self.ramp_x = r("ramp_length")
        self.exit_x = self.ramp_x + r("exit_distance")
        self.split_x = self.exit_x + r("exit_lane_length")
        self.end_x = self.split_x + 200
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

    # random personality using the chance weights, or the one set in YOU
    def pick(self, you=False):
        if you and YOU["personality"] != "random":
            return YOU["personality"]
        names = list(PERSONALITIES)
        chances = np.array([PERSONALITIES[k]["chance"] for k in names], dtype=float)
        return str(self.np_random.choice(names, p=chances / chances.sum()))

    # you on the ramp, then random traffic on the highway
    def make_cars(self):
        ramp = self.road.network.get_lane(("a", "b", self.lanes))
        # dummydriver = the rule based Driver drives you, model = the DQN picks your actions
        assert YOU["driver"] in ("dummydriver", "model"), 'YOU["driver"] has to be "dummydriver" or "model"'
        if YOU["driver"] == "model":
            me = self.action_type.vehicle_class(self.road, ramp.position(30, 0), speed=YOU["start_speed"])
            me.personality = self.pick(you=True)
        else:
            me = Driver(self.road, ramp.position(30, 0), speed=YOU["start_speed"]).setup(self.pick(you=True), self.np_random, True, self)
        me.color = VIDEO["you_color"]
        self.road.vehicles.append(me)
        self.vehicle = me
        # try 10 random spots per car, skip it if theyre all too close to someone
        taken = {i: [] for i in range(self.lanes)}
        for _ in range(self.np_random.integers(TRAFFIC["min_cars"], TRAFFIC["max_cars"] + 1)):
            for _ in range(10):
                i, x = int(self.np_random.integers(self.lanes)), self.np_random.uniform(0, self.exit_x * 0.7)
                if all(abs(x - o) > TRAFFIC["spawn_spacing"] for o in taken[i]):
                    taken[i].append(x)
                    car = Driver(self.road, self.road.network.get_lane(("a", "b", i)).position(x, 0), speed=TRAFFIC["spawn_speed"])
                    self.road.vehicles.append(car.setup(self.pick(), self.np_random, self.np_random.random() < TRAFFIC["exit_chance"], self))
                    break

    # made exit, missed exit, crashed, or None if still driving
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
        result = super().step(action)
        for v in self.road.vehicles:
            if v.crashed:
                v.color = WRECK_COLOR
        ramp_end = self.split_x + ROAD["exit_ramp_length"] - 10
        self.road.vehicles = [v for v in self.road.vehicles if v is self.vehicle or not (
            v.position[0] > self.end_x or v.lane_index[:2] == ("d", "x") and v.position[0] > ramp_end)]
        return result

    # score for this step. this is what a model would learn from
    def _reward(self, action):
        lo, hi = REWARDS["speed_range"]
        r = REWARDS["speed"] * np.clip((self.vehicle.speed - lo) / (hi - lo), 0, 1)
        # actions 0 and 2 are lane left and lane right
        r += REWARDS["lane_change"] * (action in (0, 2))
        return r + {"crashed": REWARDS["crash"], "made exit": REWARDS["made_exit"], "missed exit": REWARDS["missed_exit"]}.get(self.outcome(), 0)

    # episode ends as soon as theres an outcome
    def _is_terminated(self):
        return self.outcome() is not None

    # no time limit
    def _is_truncated(self):
        return False


# what every other file calls to get the world
def make_env(render_mode=None, **config):
    env = MergeExitEnv(render_mode=render_mode)
    env.configure(config)
    return env
