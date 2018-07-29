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
		self.update_settings(settings)
		if self.delay > 0:
			self.delay -= 1
		else:
			self.shift(self.step, self.settings["origin"])
			self.set_framerate(self.settings["speed"])
			self.write_origin([254, 0, 0, 15], self.step, self.settings["origin"], aud_data["onset"]) 
		if aud_data["onset"]:
			self.tail = settings["tail"]
			self.write_origin([254, 0, 0, 15], self.step, self.settings["origin"], aud_data["onset"]) 
			self.set_framerate(self.settings["speed"])
		self.control.set_strip(self.leds, wr_black=True)
		self.control.show()
		
		
	def shift(self, step, origin):
		if origin == 0:
			self.leds[step:] = self.leds[:-step]
		elif origin == 1:
			self.leds[:-step] = self.leds[step:]
	
	
	def write_origin(self, color, step, origin, onset):
		if self.tail > 0:
			if self.tail < self.step:
				if onset:
					if origin == 0:
						self.leds[:self.tail] = color
						self.leds[self.tail:self.step] = [0, 0, 0, 15]
					elif origin == 1:
						self.leds[-self.tail:] = color
						self.leds[-self.step:-self.tail] = [0, 0, 0, 15]
				else:
					if origin == 0:
						self.leds[:self.step] = [0, 0, 0, 15]
					elif origin == 1:
						self.leds[-self.step:] = [0, 0, 0, 15]
			else:
				limit = min(self.tail, self.step)
				if origin == 0:
					self.leds[self.step-limit:self.step] = color
					self.leds[:self.step-limit] = [0, 0, 0, 15]
				elif origin == 1:
					self.leds[-self.step:-self.step+limit] = color
					self.leds[-self.step+limit:] = [0, 0, 0, 15]
				self.tail = max(0, self.tail - self.step)
		else:
			if self.tail < 0:
				self.tail = 0
			if origin == 0:
				self.leds[:self.step] = [0, 0, 0, 15]
			elif origin == 1:
				self.leds[-self.step:] = [0, 0, 0, 15]
			
	
	
	def set_framerate(self, speed):
		if speed < 1:
			self.delay = int(0.5 * (1 / speed))
			self.step = 1
		else:
			self.delay = 0
			self.step = speed
					
				
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
		
	
	

