import json
import numpy as np


class Colors:
    def __init__(self):
        self.color_vectors = {}
        self.load_json()

    def load_json(self):
        with open("color_vectors.json", "r") as fh:
            self.color_vectors = json.loads(fh.read())

    def calc_attributes(self):
        vecs = self.color_vectors["color_vectors"]
        for key in vecs:
            num_clrs = len(vecs[key]["values"])
            r = []
            g = []
            b = []
            for color in vecs[key]["values"]:
                r.append(color[0])
                g.append(color[1])
                b.append(color[2])
            dist = np.mean([np.std(r), np.std(g), np.std(b)])
            vecs[key]["attributes"]["num_colors"] = num_clrs
            vecs[key]["attributes"]["dist"] = dist

    def update_json(self, compact=True):
        with open("color_vectors.json", "w") as fh:
            json.dump(self.color_vectors, fh, indent = None if compact else 4)
