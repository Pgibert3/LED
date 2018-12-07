import numpy as np

from Animation import Animation


# TODO: Figure out leds inheritance and abstract classes in python to create an Animation super class
class RGBAnimation(Animation):

    def __init__(self, num_leds, filter_dh):
        super().__init__(num_leds)
        self._filter_dh = filter_dh
        self._wheel = [[255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0]]
        self._point = 0

    def step(self):
        onset = self._filter_dh.get_next_onset()
        if onset[2]:
            self.leds[:] = self._wheel[self._point]
            self._point = (self._point + 1) % np.shape(self._wheel)[0]
        return self.leds

