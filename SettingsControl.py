from ColorsUtil import ColorsUtil


class GlobalSettingsControl():
    def __init__(self):
        self.c = ColorsUtil()
        self.colors = []

    def get_settings_ctl(self, anim_data):
        num_clrs = anim_data(["num_clrs"])
        clr_dist = anim_data(["clr_dist"])
        self.colors = self.c.get_colors(num_clrs_rng=num_clrs, dist_rng=clr_dist)
        #create an animations json to store animation attributes


class AnimationSettingsControl():
    def __init__(self):
        pass

    def get_animation(anim_dat):
        pass


class SwitchSettingsControl(AnimationSettingsControl):
    def __init__(self):
        super().__init__()

    def get_animation(self):
        pass
