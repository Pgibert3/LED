import Spectrum
import DisplayControl
from multiprocessing import Process, Pipe

class Brain:
	def __init__(self, num_leds):
		self.num_leds = num_leds
		#Pipes
		slv_spec2disp_conn, mst_spec2disp_conn = Pipe(False) #F
		slv_spec2brain_conn, mst_spec2brain_conn = Pipe(False) #F
		brain2disp_conn, disp2brain_conn = Pipe(True)
		#Classes
		disp = DisplayControl.DisplayControl(slv_spec2disp_conn, disp2brain_conn, self.num_leds)
		spec = Spectrum.Spectrum(mst_spec2disp_conn, mst_spec2brain_conn)
		#Processes
		self.spec_pr = Process(target=spec.start)
		self.disp_pr = Process(target=disp.start)

	def start(self):
		#self.spec_pr.start()
		self.disp_pr.start()
		while True:
			#TODO: Recieve and process onset data
			try:
				pass
			except KeyboardInterrupt:
				#self.spec_pr.join()
				self.disp_pr.join()
				break
