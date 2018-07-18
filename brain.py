import Spectrum
import color_switch
from multiprocessing import Process, Pipe

class Brain:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        
        #Pipes
        slv_spec2anim_conn, mst_spec2anim_conn = Pipe(False)
        slv_spec2brain_conn, mst_spec2brain_conn = Pipe(False)
        brain2anim_conn, anim2brain_conn = Pipe(True)
        
        #Classes
        self.anim = animation_control.animation_control(
                                slv_spec2anim_conn,
                                anim2brain_conn,
                                self.num_leds)
        self.spec = Spectrum.Spectrum(mst_spec2anim_conn,
                                      mst_spec2brain_conn)
        
        #Processes
        self.spec_pr = Process(target=spec.start)
        self.anim_pr = Process(target=anim.start)
        
        
    def start(self):
        self.spec_pr.start()
        self.anim_pr.start()


    def run(self):
        while True:
            #TODO: Recieve and process onset data
            pass

        
                    
            
            


