# all the hand written driving rules. traffic always drives with these, and your car does too
# when YOU["driver"] is "dummydriver". none of this is deep learning, its just if this then that
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

from variables import LANES, PERSONALITIES


# faster = less sight, but never below 30% of it
def sight_at(p, speed):
    return p["sight"] * max(0.3, 1 - p["sight_speed_loss"] * speed / LANES["speed_limit"])


# every non model car is one of these. its highway-env's IDM driver with our personality knobs bolted on
class Driver(IDMVehicle):
    # turns a personality from variables.py into this car's driving settings
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

    # runs every physics frame. update how fast it wants to go for the lane its in, then drive
    def act(self, action=None):
        self.target_speed = self.lane_speed()
        super().act(action)

    # the car right ahead in my lane
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
    def go_around_side(self):
        car = self.front()
        if not car or self.lane_distance_to(car) > min(self.desired_gap(self, car) + self.LENGTH, sight_at(self.p, self.speed)):
            return None
        if car.speed < 2 or (self.p["dodge"] and car.speed < self.speed - 3):
            return next((s for s in (-1, 1) if self.lane_ok(s) and self.gap_ok(self.shift(s))), None)
        return None

    # the lane next to me. side -1 = left, 1 = right
    def shift(self, side):
        f, t, i = self.lane_index
        return f, t, i + side

    # does that lane actually exist here
    def lane_ok(self, side):
        f, t, i = self.lane_index
        return 0 <= i + side < len(self.road.network.graph[f][t])

    # decides lane changes. our rules first, highway-env's normal MOBIL if none of ours apply
    def change_lane_policy(self):
        if self.lane_index != self.target_lane_index:
            # already pulling out around a stopped car? dont let highway-env's abort rule cancel it.
            # wrecks that crashed mid lane change kept cancelling it and cars sat there forever
            car = self.front()
            if car and car.speed < 2:
                return
            return super().change_lane_policy()
        side = self.side_needed()
        car = self.front()
        # already in the exit lane line? wait like normal traffic. unless its a wreck, those never move
        if side == 0 and not (car and car.crashed):
            return
        if not side:
            side = self.go_around_side()
        if side is None:
            return super().change_lane_policy()
        if self.lane_ok(side) and self.gap_ok(self.shift(side)):
            self.target_lane_index = self.shift(side)

    # left lane fast, right lane slow, ramp and exit lane slowest. spent way too long getting the lane numbers right
    def lane_speed(self):
        env, (f, t, i) = self.env, self.lane_index
        if (f, t) == ("a", "b") and i == env.lanes or (f, t) == ("c", "d") and i == env.exit_lane or (f, t) == ("d", "x"):
            return min(self.base_speed, LANES["ramp_and_exit_speed"])
        if (f, t) == ("c", "d") and env.exit_side == "left":
            i -= 1
        # 0 = far left lane, 1 = far right. blend the bonus between them
        right = i / max(env.lanes - 1, 1)
        speed = self.base_speed + LANES["left_lane_bonus"] * (1 - right) + LANES["right_lane_bonus"] * right
        return min(speed, LANES["speed_limit"]) if self.obeys_limit else speed
