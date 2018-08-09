import spidev
import numpy as np

class StripControl:
	def __init__(self, num_leds, speed=2000000):
		self.num_leds = num_leds
		sof = [0x00]*4
		ledf_lv = [0xFF]
		ledf_clrs = [0x00]*3
		ledf = ledf_lv + ledf_clrs
		self.num_byts_eof = int(np.ceil(self.num_leds / 4))
		eof = [0xFF]*self.num_byts_eof
		self.data = np.array(sof + ledf*self.num_leds + eof)
		self.spi = spidev.SpiDev()
		self.spi.open(0, 1)
		self.spi.max_speed_hz = speed
	
	
	def set_leds(self, i, color):
		r_index = 4 + 4*i + 3
		g_index = 4 + 4*i + 2
		b_index = 4 + 4*i + 1
		lv_index = 4 + 4*i
		r = color[0]
		g = color[1]
		b = color[2]
		lv = color[3]
		self.data[r_index] = r
		self.data[g_index] = g
		self.data[b_index] = b
		self.data[lv_index] = 0b11100000 | lv
	
	
	def get_led(self, i):
		r_index = 4 + 4*i + 3
		g_index = 4 + 4*i + 2
		b_index = 4 + 4*i + 1
		lv_index = 4 + 4*i
		r = int(self.data[r_index])
		g = int(self.data[g_index])
		b = int(self.data[b_index])
		lv = int(self.data[lv_index])
		return [r, g, b, lv]
	
	
	def set_strip(self, leds, wr_black=True):
		data = np.flip(leds, axis=-1)
		data[:,0] = data[:,0] | 0b11100000
		data = data.ravel()
		if wr_black:
			self.data[4:-self.num_byts_eof] = data
		else:
			self.data[4:-self.numbyts_eof][data>0] = data[data>0]
	
	
	def get_strip(self):
		return self.data[4:-self.num_byts_eof]
		
				
	def show(self):
		self.spi.writebytes(self.data.tolist())
		
		
	def set_spi_speed(self, speed):
		spi.max_speed_hz(speed)
		
