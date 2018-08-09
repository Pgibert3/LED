from Animation import Animation


class Switch(Animation):
	def __init__(self, num_leds, clr_wheel, settings={}):
		super().__init__(num_leds, settings)
		self.clr_wheel = clr_wheel
		self.settings = {}

	def next(self, aud_data):
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
	#Flat color switch
	def __init__(self, num_leds, clr_wheel, settings={}):
		super().__init__(num_leds, clr_wheel, settings)

	def trigger(self):
		color = self.clr_wheel.next_color()
		self.leds[0:self.num_leds] = color

	def hold(self):
		pass


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
