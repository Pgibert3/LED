import Spectrum
import animation_control
from multiprocessing import Process, Pipe

class Brain:
	def __init__(self, num_leds):
		self.num_leds = num_leds

		#Pipes
		slv_spec2anim_conn, mst_spec2anim_conn = Pipe(False) #F
		slv_spec2brain_conn, mst_spec2brain_conn = Pipe(False) #F
		brain2anim_conn, anim2brain_conn = Pipe(True)

		#Classes
		anim = animation_control.AnimationControl(
								slv_spec2anim_conn,
								anim2brain_conn,
								self.num_leds)
		spec = Spectrum.Spectrum(mst_spec2anim_conn,
									  mst_spec2brain_conn)

		#Processes
		self.spec_pr = Process(target=spec.start)
		self.anim_pr = Process(target=anim.start)

		
	def start(self):
		self.spec_pr.start()
		self.anim_pr.start()

	def run(self):
		self.start()
		while True:
			#TODO: Recieve and process onset data
			try:
				pass
			except KeyboardInterrupt:
				self.spec_pr.join()
				self.anim_pr.join()

				break
