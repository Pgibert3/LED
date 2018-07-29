import strip_control
import time
import numpy as np


NUM_LEDS = 60
control = strip_control.StripControl(NUM_LEDS)


def fill(color):
	control.set_leds(np.arange(0, NUM_LEDS), color)
	control.show()


def clear():
	fill([0, 0, 0, 0])


def cycle_rgb(delay_secs, lv=15):
	while True:
		try:
			fill([254, 0, 0, lv])
			time.sleep(delay_secs)
			
			fill([0, 254, 0, lv])
			time.sleep(delay_secs)
			
			fill([0, 0, 254, lv])
			time.sleep(delay_secs)
			
		except KeyboardInterrupt:
			clear()
			return
	
def strobe(delay_secs, i=np.arange(0, NUM_LEDS), color=[254,254,254,15]):
	while True:
		try:
			control.set_leds(i, color)
			control.show()
			time.sleep(delay_secs)
			
			clear()
			time.sleep(delay_secs)
			
		except KeyboardInterrupt:
			clear()
			return


clear()
