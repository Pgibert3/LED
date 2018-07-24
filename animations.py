from strip_control import StripControl
import color_wheel
import numpy as np

class Animation:
	def __init__(self, default_settings, num_leds):
		self.settings = default_settings
		self.num_leds = num_leds
		self.control = StripControl(self.num_leds)
		self.leds = np.array([[0, 0, 0, 0]] * self.num_leds)
		
		
	def update_settings(self, settings):
		self.settings = settings
		
	
	def apply_frame(self, wr_black=True):
		control.set_strip(self.leds, wr_black=wr_black)

	
	def next_frame(self, aud_data):
		pass


class ColorPulse(Animation):
	def __init__(self, default_settings, num_leds):
		super().__init__(default_settings, num_leds)
		self.delay = 0
		self.step = 1
		self.tail = 0
		
		
	def next_frame(self, settings, aud_data):
		#TODO: FIX speed < 1
		self.update_settings(settings)
		#wait for delay to end
		if self.delay > 0:
			self.delay -= 1
		#else calc speed, new delay, and step
		else:
			if settings["speed"] < 1:
				self.delay = int(0.5 * (1 / settings["speed"]))
				self.step = 1
			else:
				self.delay = 0
				self.step = self.settings["speed"]
			#shift
			self.leds[self.step:] = self.leds[:-self.step]
			#light an led at the origin
			if self.settings["origin"] == 0:
				if aud_data["onset"]:
					self.leds[:self.step] = [254, 0, 0, 15]
					self.tail = settings["tail"]
				elif self.tail > 0:
					self.leds[:self.step] = [254, 0, 0, 15]
					self.tail -= self.step
				else:
					self.leds[:self.step] = [0, 0, 0, 15]
			elif self.settings["origin"] == 1:
				if aud_data["onset"]:
					self.leds[self.num_leds-self.step:] = [254, 0, 0, 15]
					self.tail = settings["tail"]
				elif self.tail > 0:
					self.leds[self.num_leds-self.step:] = [254, 0, 0, 15]
					self.tail -= self.step
				else:
					self.leds[self.num_leds-self.step:] = [0, 0, 0, 15]
			self.control.set_strip(self.leds, wr_black=True)
			self.control.show()
		
		
class ColorSwitch(Animation):
	def __init__(self, default_settings, num_leds):
		super().__init__(default_settings, num_leds)
		self.colors = {
				"red" : [254, 0, 0],
				"blue" : [0, 254, 0],
				"green" : [0, 0, 254]}
		self.wheel = color_wheel.ColorWheel((self.colors["red"], self.colors["green"], self.colors["blue"]))
		self.settings = {
						"direction" : 0}
						
	
	def next_frame(self, settings, aud_data):
		if aud_data["onset"]:
			self.update_settings(settings)
			direction = self.settings["direction"]
			if direction == 0:
				self.wheel.rotate(1)
			elif direction == 1:
				self.wheel.rotate(-1)
			color = self.wheel.get_color()
			self.leds[:] = np.append(color, 15)
			self.control.set_strip(self.leds, wr_black=True)
			self.control.show()
		
	
	

