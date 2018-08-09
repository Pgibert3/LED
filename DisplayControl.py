from Animation import Animation
from AnimationCreator import AnimationCreator
from StripControl import StripControl

class DisplayControl():
	def __init__(self, spec_conn, brain_conn, num_leds, fname="animation_data.json"):
		self.spec_conn = spec_conn
		self.brain_conn = brain_conn
		self.num_leds = num_leds
		self.strip_ctl = StripControl(60)

	def start(self):
		while not self.brain_conn.poll():
			pass
		anim = self.brain_conn.recv()
		frame = anim.get_frame()
		while True:
			if self.brain_conn.poll():
				anim = self.brain_conn.recv()
				anim.set_frame(frame)
				print("recieved {} from brain".format(anim))
			if self.spec_conn.poll():
				aud_data = self.spec_conn.recv()
				anim.next(aud_data)
				frame = anim.get_frame()
				self.show(frame)
	
	def show(self, frame, wr_black=True):
		self.strip_ctl.set_strip(frame, wr_black=wr_black)
		self.strip_ctl.show()
