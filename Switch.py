from Animation import Animation
import numpy as np

class Switch(Animation):
	def __init__(self, num_leds):
		super().__init__(num_leds)

	def next(self, aud_data):
		onset = aud_data["onset"]
		if onset:
			self.trigger()
		else:
			self.hold()

	def trigger(self):
		pass

	def hold(self):
		pass


class Switch0(Switch):
	#Flat color switch
	def __init__(self, num_leds, clr_wheel, lv_wheel):
		super().__init__(num_leds)
		self.clr_wheel = clr_wheel
		self.lv_wheel = lv_wheel

	def trigger(self):
		color = self.clr_wheel.next_value()
		lv = self.lv_wheel.next_value()
		led = self.make_leds(color, lv)
		self.leds[0:self.num_leds] = led[0]

	def hold(self):
		pass

# in progress....................................
class Switch1(Switch):
	#Flat color that loads from one side of the strip
	def __init__(self, num_leds):
		super().__init__(num_leds)

	def trigger(self):
		pass
		#rotate color wheel
		#set origin led and all else to black

	def hold(self):
		pass
		#check if led 0 is black
		#if not expand leds by 1
		#if so pass
