import Spectrum
import color_switch
from multiprocessing import Process, Pipe

class brain:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        self.parent_data_conn, child_data_conn = Pipe(True)
        self.parent_mode_conn, child_mode_conn = Pipe(True)

        self.Spectrum = Spectrum.Spectrum(child_data_conn)
        self.color_switch = color_switch.color_switch(child_mode_conn, self.num_leds)

        self.onset_p = Process(target=self.color_switch.start)
        self.switch_p = Process(target=self.Spectrum.start)

        self.trigger = False

        self.mode = 0
        
        
    def start(self):
        self.onset_p.start()
        self.switch_p.start()


    def run(self):
        while True:
            if self.parent_data_conn.poll():
                data = self.parent_data_conn.recv()
                if data:
                    self.parent_mode_conn.send(self.mode)
                    
            
            


