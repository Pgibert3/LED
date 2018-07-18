import strip_control
import color_wheel

class ColorSwitch():
    def __init__(self, conn, num_leds):
        self.num_leds = num_leds
        
        self.colors = {
                "red" : (254, 0, 0),
                "blue" : (0, 254, 0),
                "green" : (0, 0, 254)}
        self.wheel = color_wheel.ColorWheel((self.colors["red"], self.colors["green"], self.colors["blue"]))
        
        self.strip_control = strip_control.StripControl(num_leds)
        
    
    def set_color(self, color, lv):
        r = color[0]
        g = color[1]
        b = color[2]
        self.strip_control.set_led(np.arange(0, self.num_leds), r, g, b, lv=15)
        self.strip_control.show()
    
    
    def rotate(self, step):
        self.wheel.rotate(step)
        color = self.wheel.get_color()
        self.set_color(color, 15)
        
    
    def start(self):
        while True:
            if self.conn.poll():
                data = self.conn.recv()
                self.fwd_rot()




