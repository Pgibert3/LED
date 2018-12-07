from abc import ABC, abstractmethod
import numpy as np


class Animation:

    def __init__(self, num_leds):
        self._num_leds = num_leds
        self.leds = np.array([[0, 0, 0, 0]] * self._num_leds)

    def step(self):
        pass
