from Animation import Animation
from AnimationCreator import AnimationCreator

class DisplayControl():
	def __init__(self, spec_conn, brain_conn, num_leds, fname="animation_data.json"):
		self.spec_conn = spec_conn
		self.brain_conn = brain_conn
		self.num_leds = num_leds
		anim_creator = AnimationCreator(self.num_leds, fname=fname)
		self.anims = anim_creator.process()

	def start(self):
		#animation = default
		print(self.anims)
		while True:
			#if self.brain_conn.poll():
				#brain_data = self.brain_conn.recv()
			if self.spec_conn.poll():
				aud_data = self.spec_conn.recv()
					#animation.next_frame(aud_data)
