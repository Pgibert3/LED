import ColorWheel
import numpy as np

class Animation:
	def __init__(self, num_leds, settings={}):
		self.num_leds = num_leds
		self.leds = np.array([[0, 0, 0, 0]] * self.num_leds)
		self.set_settings(settings)

	def update_settings(self, settings):
		self.settings = settings

	def get_frame(self, wr_black=True):
		return self.leds
	
	def set_frame(self, frame):
		self.leds = frame

	def next(self, aud_data):
		pass

	def set_settings(self, settings):
		self.settings = settings

	def get_settings():
		return self.settings
