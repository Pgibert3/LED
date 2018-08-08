import json


class ColorsUtil:
    def __init__(self):
        self.animations = {}
        self.load_json()

    def load_json(self):
        with open("color_vectors.json", "r") as fh:
            self.animations = json.loads(fh.read())
        close(fh)

    def update_json(self, compact=True):
        with open("color_vectors.json", "w") as fh:
            json.dump(self.color_vectors, fh, indent = None if compact else 4)
        close(fh)

    def get_animations(self, atts):
        anim = self.animations["animations"]
        anims = []
        for item in anim.items():
            if (
                item["area"] >= atts["area"][0]
                and item["area"] <= atts["area"][1]
                and item["color_complexity"] >= atts["color_complexity"][0]
                and item["color_complexity"] <= atts["color_complexity"][1]
                and item["dynamic_pregression"] == atts["dynamic_progession"][0]
                and item["dynamic_area"] == atts["dynamic_area"][1]):
            anims.append(item["name"])
        return anims
