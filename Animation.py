#from strip_control import StripControl
import ColorWheel
import numpy as np

class Animation:
	def __init__(self, num_leds):
		self.num_leds = num_leds
		#self.control = StripControl(self.num_leds)
		self.leds = np.array([[0, 0, 0, 0]] * self.num_leds)

	def update_settings(self, settings):
		self.settings = settings

	def show(self, wr_black=True):
		#control.set_strip(self.leds, wr_black=wr_black)
		#control.show()
		pass

	def next(self, aud_data):
		pass

	def get_settings():
		pass
