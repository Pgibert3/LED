import numpy as np

from Animation import Animation
from RGBAnimation import RGBAnimation
from FilterDataHub import FilterDataHub


from LEDStripSimulatorDisplayController import LEDStripSimulatorDisplayController
from DisplayController import DisplayController


class MainDisplayController(DisplayController):

    def __init__(self, num_leds, animation):
        # TODO: Consider supporting multiple different frontends
        self._num_leds = num_leds
        self._animation = animation
        # TODO: Extract data hubs into brain
        self._filter_dh = FilterDataHub()

        self.start()

    def start(self):
        """
        Starts the designated frontend display controller, and runs the main animation loop
        """
        self.set_animation(RGBAnimation(self._num_leds, self._filter_dh))
        display = LEDStripSimulatorDisplayController(self._num_leds)
        while True:
            try:
                leds = self._animation.step()
                display.set_strip(leds)
            except KeyboardInterrupt:
                break

    def set_animation(self, animation):
        """
        Setter for animation attribute
        @param animation: the animation to play
        @type animation: Animation
        """
        self._animation = animation


dc = MainDisplayController(15, None)

