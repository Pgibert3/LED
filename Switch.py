from animations import Animation


class Switch(Animation):
	def __init__(self, num_leds):
		super().__init__(num_leds)

	def next_frame(self, mode, aud_data):
        onset = aud_data["onset"]
        if onset:
            self.trigger()
        else:
            self.hold()
        self.show_frame()

    def trigger(self):
        pass

    def hold(self):
        pass


class Switch0(Switch):
    #Flat color switch
    def __init__(self, num_leds):
        super().__init__(num_leds)

    def trigger(self):
        #rotate color wheel

    def hold(self):
        pass


class Switch1(Switch):
    #Flat color that loads from one side of the strip
    def __init__(self, num_leds):
        super().__init__(num_leds)

    def trigger(self):
        #rotate color wheel
        #set origin led and all else to black

    def hold(self):
        #check if led 0 is black
        #if not expand leds by 1
        #if so pass
