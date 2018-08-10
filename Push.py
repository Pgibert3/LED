import numpy as np


class Push():
	def __init__(self, num_leds, clr_wheel, fr, length, origin="start" settings={}):
		super().__init__(num_leds, settings)
		self.fr = fr
		self.length = length
		self.origin = origin
		self.delay = 0
		self.step = 1
		self.q = self.length
		self.d = 0
		self.set_timing()
	
	def next(self, aud_data):
		onset = aud_data["onset"]
		if onset:
			self.d = 0
		elif self.d == 0:
			self.push()
			self.d = delf.delay
		else:
			self.d -= 1
	
	def push(self):
		#TODO: add better error checking
		self.shift()
		if self.q < 0:
			print("terrible error! See hold() of Push object")
		elif self.q == 0:
			write_origin(self.step-self.q, np.append(self.clr_wheel.BLACK, 15))
		elif self.q - self.step < 0:
			write_origin(self.step-self.q, np.append(color, 15))
			self.q = seld.length
		else:
			self.write_origin(self.step, np.append(color, 15))
			self.q -= self.step
		
	def shift(self, push_dir="start"):
		#TODO: add better error checking
		if shft_dir == "start":
			self.leds[self.step:self.num_leds] = self.leds[:-self.step]
		elif shft_dir == "end":
			self.leds[:-self.step] = self.leds[self.step:self.num_leds]
		else:
			print("invalid settings")
					
	def write_origin(self, num, color):
		#TODO: Add better error checking
		if origin == "start":
			self.leds[:num] = [color]*num
		elif origin = "end":
			seld.leds[-num:] = [color]*num
		else:
			print("invalid settings")	

	def shift(self, push_dir="start"):
		#TODO: add better error checking
		if shft_dir == "start":
			self.leds[self.step:self.num_leds] = self.leds[:-self.step]
		elif shft_dir == "end":
			self.leds[:-self.step] = self.leds[self.step:self.num_leds]
		else:
			print("invalid settings")

	def set_timing(self):
		if self.fr >= 0 and self.fr < 1:
			self.delay = int(0.5 * (1 / self.fr))
			self.step = 1
		elif speed > 1:
			self.fr = int(self.fr)
			self.delay = 0
			self.step = self.fr
