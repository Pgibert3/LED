import spidev
import numpy as np

class apa102:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        self.sof = [0x00]*4
        self.eof = [0xFF]*4
        self.clf = [0b11100000, 0, 0, 0]
        self.leds = [self.clf]*self.num_leds

        self.spi = spidev.SpiDev()
        self.spi.open(0,1)
        self.spi.max_speed_hz = int(10e6)

        #self.spi = spidev.SpiDev()
        #self.spi.open(0,1)
        #self.spi.max_speed_hz = int(10e6)

        #self.SOF = [0x00]*4
        #self.EOF = [0xFF]*4 
        #self.CLF = [0b11100000, 0, 0, 0]
        #self.LEDS = [list(self.CLF) for i in range(num_leds)]


    def update_leds(self):
        try:
            self.spi.writebytes(self.sof)
            for l in self.leds:
                self.spi.writebytes(l)
            self.spi.writebytes(self.eof)
        except TypeError:
            print("Invalid parameter provided to spidev.writebytes()")


    def get_led_frame(self, r, g, b, lv):
        if r > 254 or r < 0:
            raise ValueError
        if g > 254 or g < 0:
            raise ValueError
        if b > 254 or b < 0:
            raise ValueError
        if lv > 31 or lv < 0:
            raise ValueError

        b1 = 0b11100000 | lv
        b2 = int(b)
        b3 = int(g)
        b4 = int(r)

        return [b1, b2, b3, b4]


    def set_led(self, i, r, g, b, lv):
        try:
            led_frame = self.get_led_frame(r, g, b, lv)
            self.leds[i] = led_frame
        except ValueError:
            print("Invalid value for r ({}), g ({}), b ({}), or lv ({})"\
                    .format(r, g, b, lv))
            return False
        
        self.update_leds()
        return True


    def clear(self):
        for i in range(0, self.num_leds):
            self.set_led(i, 0, 0, 0, 0)


    def fill(self, r, g, b, lv):
        for i in range(0, self.num_leds):
            self.set_led(i, r, g, b, lv)
