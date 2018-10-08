from Animation import Animation
import numpy as np
import time

class Push(Animation):
	def __init__(self, num_leds, fr, length):
		super().__init__(num_leds)
		self.fr = fr
		self.length = length
		self.delay = 0
		self.step = 1
		self.q = 0
		self.d = 0
		self.set_timing()

	def next(self, aud_data):
		onset = aud_data["onset"]
		if onset:
			self.d = 0
			self.q = self.length
			self.reset()
		if self.d == 0:
			self.push()
			self.d = self.delay
		else:
			self.d -= 1

	def push(self):
		pass

	def reset():
		pass

	def shift(self, origin):
		#TODO: add better error checking
		if origin == "start":
			self.leds[self.step:self.num_leds] = self.leds[:-self.step]
		elif origin == "end":
			self.leds[:-self.step] = self.leds[self.step:self.num_leds]
		else:
			print("invalid settings")

	def write_origin(self, origin, num, colors, lvs):
		#TODO: Add better error checking
		leds = self.make_leds(colors, lvs)
		if origin == "start":
			if num <= 1:
				self.leds[:num] = leds
			else:
				self.leds[self.step-num:self.step] = leds
		elif origin == "end":
			if num <= 1:
				self.leds[-num:] = np.flip(leds, 0)
			else:
				self.leds[-self.step:-self.step+num] = np.flip(leds, 0)
		else:
			print("invalid settings")

	def set_timing(self):
		if self.fr >= 0 and self.fr < 1:
			self.delay = int(0.5 * (1 / self.fr))
			self.step = 1
		elif self.fr > 1:
			self.fr = int(self.fr)
			self.delay = 0
			self.step = self.fr


class Push0(Push):
	def __init__(self, num_leds, fr, length, origin, clr_wheel, lv_wheel):
		super().__init__(num_leds, fr, length)
		self.origin = origin
		self.clr_wheel = clr_wheel
		self.lv_wheel = lv_wheel

	def push(self):
		#TODO: add better error checking
		self.shift(self.origin)
		if self.q < 0:
			print("terrible error, q < 0! See push() of Push object")
		elif self.q == 0:
			self.write_origin(self.origin, self.step-self.q, self.clr_wheel.BLACK, [[0]])
		elif self.q < self.step:
			num = self.step - self.q
			colors = self.clr_wheel.next_value(num=num)
			lvs = self.lv_wheel.next_value(num=num)
			self.write_origin(self.origin, num, colors, lvs)
			self.q = 0
		else:
			num = self.step
			colors = self.clr_wheel.next_value(num=num)
			lvs = self.lv_wheel.next_value(num=num)
			self.write_origin(self.origin, num, colors, lvs)
			self.q -= self.step

	def reset(self):
		self.clr_wheel.reset_pos()
		self.lv_wheel.reset_pos()
