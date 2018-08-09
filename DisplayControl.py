from Animation import Animation
from AnimationCreator import AnimationCreator

class DisplayControl():
	def __init__(self, spec_conn, brain_conn, num_leds, fname="animation_data.json"):
		self.spec_conn = spec_conn
		self.brain_conn = brain_conn
		self.num_leds = num_leds

	def start(self):
		while True:
			if self.brain_conn.poll():
				anim = self.brain_conn.recv()
				print("recieved {} from brain".format(anim))
			if self.spec_conn.poll():
				aud_data = self.spec_conn.recv()
				print(aud_data["onset"])
				anim.next(aud_data)
