import animations

class AnimationControl():
	def __init__(self, spec_conn, brain_conn, num_leds):
		self.spec_conn = spec_conn
		self.brain_conn = brain_conn
		self.Switch = animations.Switch(num_leds)
		self.PulseTrain = animations.Pulse(num_leds)


	def start(self):
		#animation = default
		while True:
			#if self.brain_conn.poll():
				#anim_data = self.brain_conn.recv()
				#AnimationSettingsControl = self.GlobalSettingsControl.get_settings_ctl(anim_data)
				#animation = AnimationSettingsControl.get_animation(anim_data)
			if self.spec_conn.poll():
				aud_data = self.spec_conn.recv()
					#animation.next_frame(aud_data)
