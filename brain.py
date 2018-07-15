import Spectrum
import color_switch
from multiprocessing import Process, Pipe

class brain:
	__init__(num_leds):
		self.num_leds = num_leds
		self.parent_data_conn, child_data_conn = Pipe()
		self.parent_mode_conn, child_mode_conn = Pipe()
		
		self.Spectrum = Spectrum.Spectrum(child_data_conn)
		color_switch = color_switch.color_switch(child_mode_conn)
		
		self.onset_p = Process(target=self.color_switch.run)
		self.switch_p = Process(target=self.Spectrum.start)
		
		self.trigger = False
		
		mode = 0
		
		
	def start():
		self.onset_p.start(self.child_data_conn)
		self.switch_p.start(self.child_mode_conn)
	
	def run():
		if self.parent_data_conn.poll():
			data = self.parent_data_conn.recv()
			if data:
				self.parent_data_conn.send(mode)
			
		
		
