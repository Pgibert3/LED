import strip_control
import color_wheel
import time
import numpy as np

class color_switch():
    def __init__(self, num_leds):
        self.num_leds = num_leds
        
        self.colors = {
                "red" : (254, 0, 0),
                "blue" : (0, 254, 0),
                "green" : (0, 0, 254)}
        self.wheel = color_wheel.color_wheel((self.colors["red"], self.colors["green"], self.colors["blue"]))
        
        self.strip_control = strip_control.strip_control(num_leds)
        
        #self.conn = conn
        self.conn_key = {
                0 : self.fwd_rot,
                1 : self.bwd_rot}
                        
    
    def set_color(self, color, lv):
        r = color[0]
        g = color[1]
        b = color[2]
        self.strip_control.set_led(np.arange(0, self.num_leds), r, g, b, lv=15)
        self.strip_control.show()
    
    
    def fwd_rot(self):
        self.wheel.rotate(1)
        color = self.wheel.get_color()
        self.set_color(color, 15)
            
    
    def bwd_rot(self):
        self.wheel.rotate(-1)
        color = self.wheel.get_color()
        self.set_color(color, 15)
    
    
    def start(self, delay=.001):
        while True:
            if self.conn.poll():
                data = self.conn.recv()
                self.fwd_rot()
            time.sleep(delay)




