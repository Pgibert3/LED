from ColorsUtil import ColorsUtil


class GlobalSettingsControl():
    def __init__(self):
        self.c = ColorsUtil()
        self.colors = []

    def get_settings_ctl(self, anim_data):
        num_clrs = anim_data(["num_clrs"])
        clr_dist = anim_data(["clr_dist"])
        area = anim_data(["area"])
        clr_comp = anim_data(["color_complexity"])
        dyn_pr = anim_data(["dynamic_progression"])
        dyn_area = anim_data(["dynamic_area"])
        self.colors = self.c.get_colors(num_clrs_rng=num_clrs, dist_rng=clr_dist)
        data = {
                "area" : area,
                "color_complexity" : clr_comp,
                "dynamic_progression" : dyn_pr,
                "dynamic_area" : "dyn_area"
        }


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
