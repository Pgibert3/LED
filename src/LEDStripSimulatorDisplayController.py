from graphics import GraphWin, color_rgb, Circle, Point  # Credit to John Zelle for graphics.py
import multiprocessing as mp

from DisplayController import DisplayController


class LEDStripSimulatorDisplayController:

    def __init__(self, num_leds):
        """
        Simulates an addressable LED strip using John Zelle's graphics module
        @param num_leds:
        @type num_leds:
        """
        self._led_drawing_objs = []  # contains the circle objects from graphics module representing each LED

        # TODO: Extract these values
        LED_RADIUS = 10
        LED_SPACING = 20
        self._led_radius = LED_RADIUS
        self._led_spacing = LED_SPACING
        self._num_leds = num_leds

        # Draw the initial LED strip UI
        x_dim = self._num_leds * (2 * self._led_radius + self._led_spacing) + self._led_spacing
        y_dim = 2 * (self._led_radius + self._led_spacing)

        self._win = GraphWin('LED Strip', x_dim, y_dim)  # The window object used by graphics module
        self._win.setBackground(color_rgb(255, 255, 255))

        for i in range(0, self._num_leds):
            x = i * (2 * self._led_radius + self._led_spacing) + self._led_radius + self._led_spacing
            y = self._led_radius + self._led_spacing
            # Draw led as a circle
            led = Circle(Point(x, y), self._led_radius)
            led.setOutline(color_rgb(0, 0, 0))
            led.setFill(color_rgb(255, 255, 255))
            led.draw(self._win)
            # Append led to list
            self._led_drawing_objs.append(led)

    def set_strip(self, leds):
        # TODO: Implement brightness
        """
        Sets all leds in the strip. Throws a value error if provided leds param is not of the correct shape
        @param leds: array of leds of the form [ [red, blue, green, brightness], ... ]
        @type leds: np.array((self._num_leds, 4), dtype=int)
        """
        for i in range(0, self._num_leds):
            red = leds[i][0]
            blue = leds[i][1]
            green = leds[i][2]
            color = color_rgb(red, green, blue)
            led = self._led_drawing_objs[i]
            led.setFill(color)

