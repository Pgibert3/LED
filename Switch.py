from Animation import Animation
from ColorWheel import ColorWheel

class Switch(Animation):
	def __init__(self, num_leds, clr_wheel):
		super().__init__(num_leds)
		self.clr_wheel = clr_wheel

	def next(self, mode, aud_data):
		onset = aud_data["onset"]
		if onset:
		    self.trigger()
		else:
		    self.hold()
		self.show()

	def trigger(self):
	    pass

	def hold(self):
	    pass


class Switch0(Switch):
	def __init__(self, num_leds, clr_wheel):
		super().__init__(num_leds, clr_wheel)

	def trigger(self):
		color = self.clr_wheel.next_color()
		self.leds[0:self.num_leds] = color

	def hold(self):
		pass
