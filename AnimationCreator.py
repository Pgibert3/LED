import json
import Switch
import Push
from Wheel import ColorWheel, LVWheel

class AnimationCreator:
    def __init__(self, num_leds, fname="animation_data.json"):
        self.num_leds = num_leds
        self.fname = fname
        self.anim_data = []
        self.load()

    def load(self):
        with open(self.fname, "r") as fh:
            self.anim_data = json.loads(fh.read())
            fh.close()

    def save(self, indent=2):
        with open(self.fname, "w") as fh:
            json.dump(self.anim_data, fh, indent=indent)
            fh.close()

    def list(self):
        for i in range(0, len(self.anim_data["animations"])):
            print(self.anim_data["animations"][i])

    def add(self, data):
        self.anim_data["animations"].append(data)

    def remove(self, name):
        for i in range(0, len(self.anim_data["animations"])):
            if self.anim_data["animations"][i]["name"] == name:
                del self.anim_data["animations"][i]
                break

    def get_anim_settings(self):
        #TODO: add error checking
        name = input("name: ")
        type = input("type: ")
        if type == "Switch":
            effect = input("effect: ")
            if effect == "0":
                colors = self.get_clr_wheel_settings()
                lvs = self.get_lv_wheel_settings()
                settings = {
                    "name" : name,
                    "type" : type,
                    "effect" : effect,
                    "colors" : colors,
                    "lvs" : lvs
                }
            else:
                print("Invalid settings")
                return
        elif type == "Push":
            effect = input("effect: ")
            if effect == "0":
                fr = input("frames per clock: ")
                length = input("length: ")
                origin = input("origin: ")
                colors = self.get_clr_wheel_settings()
                lvs = self.get_lv_wheel_settings()
                settings = {
                    "name" : name,
                    "type" : type,
                    "effect" : effect,
                    "fr" : fr,
                    "origin": origin,
                    "length": length,
                    "colors" : colors,
                    "lvs" : lvs
                }
            else:
                print("Invalid settings")
                return
        else:
            print("Invalid settings")
            return
        return settings

    def get_clr_wheel_settings(self):
        colors = []
        rsp = input("Add Color? (y/n): ")
        while rsp == "y":
            r = int(input("RGB values, R?: "))
            g = int(input("RGB values, G?: "))
            b = int(input("RGB values, B?: "))
            if r < 0 or r > 255 or g < 0 or g > 255 or b < 0 or b > 255:
                print("invalid RGB value(s)")
            else:
                colors.append([r,g,b])
            rsp = input("Add another Color? (y/n): ")
        return colors

    def get_lv_wheel_settings(self):
        lvs = []
        rsp = input("Add lv? (y/n): ")
        while rsp == "y":
            lv = int(input("lv value: "))
            if lv < 0 or lv > 32:
                print("invalid lv value")
            else:
                lvs.append([lv])
            rsp = input("Add another lv? (y/n): ")
        return lvs

    def create_switch(self, settings):
        #TODO: add better error checking
        effect = settings["effect"]
        clr_wheel = ColorWheel(settings["colors"])
        if effect == "0":
            return Switch.Switch0(self.num_leds, clr_wheel)
        else:
            print("Invalid animation settings")

    def create_push(self, settings):
        #TODO: add better error checking
        effect = settings["effect"]
        fr = settings["fr"]
        length = settings["length"]
        origin = settings["origin"]
        clr_wheel = ColorWheel(settings["colors"])
        lv_wheel = LVWheel(settings["lvs"])
        if effect == "0":
            return Push.Push0(self.num_leds, fr, length, origin, clr_wheel, lv_wheel)
        else:
            print("Invalid animation settings")

    def process(self):
        #TODO add better error checking
        anims = []
        for i in range(0, len(self.anim_data["animations"])):
            if self.anim_data["animations"][i]["type"] == "Switch":
                settings = self.anim_data["animations"][i]
                anim = self.create_switch(settings)
                anims.append(anim)
            elif self.anim_data["animations"][i]["type"] == "Push":
                settings = self.anim_data["animations"][i]
                anim = self.create_push(settings)
                anims.append(anim)
            else:
                print("Did not find any valid animations")
        return anims
