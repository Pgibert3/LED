import pygame

class LEDStripSimulatorDisplayController:

    def __init__(self, num_leds):
        # TODO: Extract these values
        LED_RADIUS = 10
        LED_SPACING = 20

        self._led_radius = LED_RADIUS
        self._led_spacing = LED_SPACING

    def start(self):
        """
        Initializes the LED strip UI
        @return:
        @rtype:
        """

    def check_closed(self):
        """
        Terminates the LED strip UI if the pygame has been closed
        @return:
        @rtype:
        """

    def set_strip(self, leds):
        """
        Sets all leds in the strip. Throws a value error if provided leds param is not of the correct shape
        @param leds: array of leds of the form [ [red, blue, green, brightness], ... ]
        @type leds: np.array((self._num_leds, 4), dtype=int)
        """

    def show(self):
        """
        Updates the display
        @return:
        @rtype:
        """

