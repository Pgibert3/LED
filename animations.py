from strip_control import StripControl
import color_wheel
import numpy as np

class Animation:
	def __init__(default_settings, num_leds):
		self.settings = default_settings
		self.control = StripControl(num_leds)
		self.leds = [(0, 0, 0, 0)] * num_leds
		
		
	def update_settings(self, settings):
		self.settings = settings
		
	
	def apply_frame(self, wr_black=True):
		if wr_black:
			control.set_leds(np.arange(0, self.num_leds), self.leds)
		else:
			control.set_strip(self.leds)
			mask = self.leds != (0, 0, 0, 0)
			control.set_leds(mask, self.leds[mask])
	
	
	def next_frame(self, aud_data):
		pass


class ColorPulse(Animation):
	def __init__(self, default_settings, num_leds):
		super().__init__(default_settings, num_leds)
		self.delay = 0
		self.tail = 5
		
		
	def next_frame(self, aud_data):
		if aud_data["onset"]:
			if self.delay > 0:
				self.delay -= 1
			else:
				if speed < 1:
					self.delay = int(0.5 * (1 / settings["speed"]))
					step = 1
				else:
					self.delay = 0
					step = settings["speed"]
					
				if settings["origin"] == 0:
					self.leds[step:] = self.leds[:-step]
					self.leds[0] = (254, 0, 0, 15)
				elif settings["origin"] == 1:
					self.leds[:-step] = self.leds[step:]
					self.leds[self.num_leds - 1] = (254, 0, 0, 15)
		elif tail > 0:
			if settings["origin"] == 0:
				self.leds[step:] = self.leds[:-step]
				self.leds[0] = (254, 0, 0, 15)
			elif settings["origin"] == 1:
				self.leds[:-step] = self.leds[step:]
				self.leds[self.num_leds - 1] = (254, 0, 0, 15)
		elif trail == 0:
			if settings["origin"] == 0:
				self.leds[step:] = self.leds[:-step]
				self.leds[0] = (0, 0, 0, 0)
			elif settings["origin"] == 1:
				self.leds[:-step] = self.leds[step:]
				self.leds[self.num_leds - 1] = (0, 0, 0, 0)
		
		
class ColorSwitch():
	def __init__(self, num_leds):
		self.num_leds = num_leds
		
		self.colors = {
				"red" : (254, 0, 0),
				"blue" : (0, 254, 0),
				"green" : (0, 0, 254)}
		self.wheel = color_wheel.ColorWheel((self.colors["red"], self.colors["green"], self.colors["blue"]))
		
		self.control = StripControl(num_leds)
		
	
	def set_color(self, color, lv):
		r = color[0]
		g = color[1]
		b = color[2]
		self.control.set_led(np.arange(0, self.num_leds), r, g, b, lv=15)
		self.control.show()
	
	
	def switch(self, direction):
		if direction == 0:
			self.wheel.rotate(1)
		elif direction == 1:
			self.wheel.rotate(-1)
		
		color = self.wheel.get_color()
		self.set_color(color, 15)
		+
	
	

