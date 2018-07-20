import animations

class AnimationControl():
	def __init__(self, spec_conn, brain_conn, num_leds):
		self.spec_conn = spec_conn
		self.brain_conn = brain_conn

		self.clr_sw = animations.ColorSwitch(num_leds)
		self.clr_pls = animations.ColorPulse(num_leds)

		self.data = (0, False)

		self.mode = 'color_pulse'
		self.mode_data = (0,1,4)


	def start(self):
		while True:
			#if self.brain_conn.poll():
				#data = self.brain_conn.recv()
				#self.mode = data[0]
				#self.mode_data = data[1]

			if self.spec_conn.poll():
				self.data = self.spec_conn.recv()
				
				if self.mode == 'color_switch':
					self.clr_sw_ctl(self.mode_data, self.data)
				elif self.mode == 'color_pulse':
					self.clr_pls_ctl(self.mode_data, self.data)

	def clr_sw_ctl(self, settings, data):
		'''
		settings[0] : direciton - direction to rotate color wheel
				  0 : rotate wheel forward
				  1 : rotate wheel backwards
		'''

		if data[1] == True:
			self.clr_sw.switch(settings[0])
		else:
			pass


	def clr_pls_ctl(self, settings, data):
		'''
		settings[0] : origin - end of strip at which the pulse begins
				  0 : start of strip
				  1 : end of strip
				  
		settings[1] : speed - speed of the pulse
				0-# : speed in leds per cycle
				
		settings[2] : trail length - length of each pulse
			    0-# : length in leds of pulse

		'''

		self.clr_pls.pulse(settings[0], settings[1], settings[2], data[1])
