from multiprocessing import Process, Pipe
import numpy as np
import time
from DisplayControl import DisplayControl
from Spectrum import Spectrum
from AnimationCreator import AnimationCreator


class Brain:
	def __init__(self, num_leds, fname="animation_data.json"):
		self.num_leds = num_leds
		#Pipes
		slv_spec2disp_conn, mst_spec2disp_conn = Pipe(False) #F
		slv_spec2brain_conn, mst_spec2brain_conn = Pipe(False) #F
		slv_brain2disp_conn, self.mst_brain2disp_conn = Pipe(False)
		#Classes
		disp = DisplayControl(
								slv_spec2disp_conn,
								slv_brain2disp_conn,
								self.num_leds)
		spec = Spectrum(mst_spec2disp_conn,
									  mst_spec2brain_conn)
		anim_creator = AnimationCreator(self.num_leds, fname=fname)
		self.anim_ptrs = anim_creator.process()
		#Processes
		self.spec_pr = Process(target=spec.start)
		self.disp_pr = Process(target=disp.start)

	def start(self):
		if __name__ == "__main__":
			self.spec_pr.start()
			self.disp_pr.start()
			while True:
				try:
					ptr = self.get_random(self.anim_ptrs)
					self.mst_brain2disp_conn.send(ptr)
					time.sleep(4)
				except KeyboardInterrupt:
					self.spec_pr.join()
					self.disp_pr.join()
					break

	def get_random(self, ptrs):
		i = np.random.randint(0, len(ptrs))
		print(i)
		return ptrs[i]
