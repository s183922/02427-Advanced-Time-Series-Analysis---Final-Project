import numpy as np

class Compass():
    def __init__(self, target):
        self.target = target

    def get_velocity(self, position):
        return self.target - position


class Difference():
    def __init__(self, target):
        self.target = target
        self.prev_position = np.array([0, 0])
        self.velocity = np.array([1, 1])
    
    def get_velocity(self, position):
        dpos = position - self.prev_position
        alpha = np.arctan2(dpos[1], dpos[0])
        dtarget = self.target - position
        alpha_target = np.arctan2(dtarget[1], dtarget[0])
        alpha_correction = alpha_target - alpha
        self.prev_position = position
        return np.array([[np.cos(alpha_correction), np.sin(alpha_correction)], [np.sin(alpha_correction), np.cos(alpha_correction)]])@dpos