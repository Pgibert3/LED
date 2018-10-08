from Animation import Animation
from AnimationCreator import AnimationCreator
#from StripControl import StripControl
from UI.VStrip import VStrip as StripControl  # Edited for VStrip!!!!
import time

class DisplayControl():
	def __init__(self, spec_conn, brain_conn, num_leds, fname="animation_data.json"):
		self.spec_conn = spec_conn
		self.brain_conn = brain_conn
		self.num_leds = num_leds
		self.strip_ctl = StripControl(self.num_leds)

	def start(self):
		while not self.brain_conn.poll():
			pass
		anim = self.brain_conn.recv()
		frame = anim.get_frame()
		self.strip_ctl.start() #Edited for VStrip!!!!!
		while True:
			self.strip_ctl.check_closed() # Edited for VStrip!!!!
			if self.brain_conn.poll():
				anim = self.brain_conn.recv()
				anim.set_frame(frame)
			if self.spec_conn.poll():
				aud_data = self.spec_conn.recv()
				anim.next(aud_data)
				frame = anim.get_frame()
				self.show(frame)

	def show(self, frame, wr_black=True):
		self.strip_ctl.set_strip(frame, wr_black=wr_black)
		self.strip_ctl.show()
