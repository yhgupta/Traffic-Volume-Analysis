# Types of vehicle, numbers denote their value in COCO DATASET
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

# 5 Vehicles are simplified into 3 classes
V_LIGHT_VEHICLE = 10
LIGHT_VEHICLE = 11
HEAVY_VEHICLE = 12

# The direction of vehicle
DIR_DOWN = 0
DIR_UP = 1
DIR_NONE = 2

# Max frames difference after which vehicle expires
MAX_FRAMES = 10

MARGINS = 10
class Vehicle:
    def __init__(self, frame, v_type, x, y, w, h):
        self.v_types = {v_type: 1}
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.first_pos = y + h / 2
        self.last_pos = self.first_pos
        self.frame = frame

    def get_type(self):
        m = 0
        v_type = None
        for t in self.v_types.keys():
            if self.v_types[t] > m:
                v_type = t
                m = self.v_types[t]
        if v_type == BICYCLE or v_type == MOTORCYCLE:
            return V_LIGHT_VEHICLE
        elif v_type == CAR:
            return LIGHT_VEHICLE
        elif v_type == TRUCK or v_type == BUS:
            return HEAVY_VEHICLE

    def match_and_update(self, frame, v_type, x, y, w, h):
        center_x = x + w / 2
        center_y = y + h / 2
        if (
            center_x >= self.x - MARGINS
            and center_x <= self.x + self.w + MARGINS
            and center_y >= self.y - MARGINS
            and center_y <= self.y + self.h + MARGINS
        ):  
            self.frame = frame
            if v_type in self.v_types:
                self.v_types[v_type] += 1
            else:
                self.v_types[v_type] = 1
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.last_pos = center_y
            return True
        return False

    def get_direction(self):
        if self.first_pos > self.last_pos:
            return DIR_UP
        elif self.first_pos < self.last_pos:
            return DIR_DOWN
        return DIR_NONE

    def expired(self, frame):
        if frame - self.frame > MAX_FRAMES:
            return True
        return False

