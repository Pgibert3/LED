class SettingsControl:
    def __init__(self, animation_conn, attribute_max):
        self.conn = animation_conn
        self.attmax = attribute_max

    def cast_complexity(self, complexity):
        pass

    def cast_gradient(self, gradient):
        pass

    def cast_depth(self, depth):
        pass

    def send_settings(self, settings):
        pass


class SwitchSettingsControl(SettingsControl):
    def __init__(self, animaiton_conn):
         super().__init__(animation_conn)

    def cast_complexity(self, complexity):
        #complexity correlates somewhat to num of colors and the most to lv_filter


    def cast_gradient(self, gradient):
        #gradient correlates to color distribution and somewhat to lv_filter

    def cast_depth(self, depth):
        #depth correlates the most to num of colors


class PulseSettingsControl(SettingsControl):
    def __init__(self, animaiton_conn):
         super().__init__(animation_conn)

    def cast_complexity(self, complexity):
        #complexity correlates somewhat to num of colors, the most to lv_filter, some to tail length

    def cast_gradient(self, gradient):
        #gradient correlates to color distribution and somewhat to lv_filter, the most to tail length

    def cast_depth(self, depth):
        #depth correlates the most to num of colors
