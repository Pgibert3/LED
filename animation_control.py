import animations

class AnimationControl():
	def __init__(self, spec_conn, brain_conn, num_leds):
		self.spec_conn = spec_conn
		self.brain_conn = brain_conn
		self.mode_data = {
					"onset" : False,
					"amplitude" : 0} 
		self.mode_name = "color_pulse"
		self.mode_settings = {
						"origin" : 0,
						"speed" : .5,
						"tail" : 2}
		self.clr_sw = animations.ColorSwitch(self.mode_settings, num_leds)
		self.clr_pls = animations.ColorPulse(self.mode_settings, num_leds)


	def start(self):
		while True:
			#if self.brain_conn.poll():
				#data = self.brain_conn.recv()
				#self.mode = data[0]
				#self.mode_data = data[1]
			if self.spec_conn.poll():
				self.mode_data = self.spec_conn.recv()
				if self.mode_name == 'color_switch':
					self.color_switch_ctl(self.mode_settings, self.mode_data)
				elif self.mode_name == 'color_pulse':
					self.color_pulse_ctl(self.mode_settings, self.mode_data)

	def color_switch_ctl(self, settings, data):
		if data["onset"] == True:
			self.clr_sw.next_frame(settings, data)
		else:
			pass


	def color_pulse_ctl(self, settings, data):
		self.clr_pls.next_frame(settings, data)
