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
		#self.control.set_strip(self.leds, wr_black=True)
		#self.control.show()


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
