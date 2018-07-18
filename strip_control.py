import spidev

class StripControl:
	def __init__(self, num_leds, speed=3000000):
		self.num_leds = num_leds
		
		sof = [0x00]*4
		
		ledf_lv = [0xFF]
		ledf_clrs = [0x00]*3
		ledf = ledf_lv + ledf_clrs
		
		num_byts_eof = int(np.ceil(self.num_leds / 4)) 
		eof = [0xFF]*num_byts_eof
		
		self.data = np.array(sof + ledf*num_leds + eof)
		
		self.spi = spidev.SpiDev()
		self.spi.open(0, 1)
		self.spi.max_speed_hz = speed
	
	
	def set_led(self, i, r, g, b, lv):
		r_index = 4 + 4*i + 3
		g_index = 4 + 4*i + 2
		b_index = 4 + 4*i + 1
		lv_index = 4 + 4*i
		
		self.data[r_index] = r
		self.data[g_index] = g
		self.data[b_index] = b
		self.data[lv_index] = 0b11100000 | lv
	
	
	def show(self):
		self.spi.writebytes(self.data.tolist())
		
		
	def set_spi_speed(speed):
		spi.max_speed_hz(speed)
		
