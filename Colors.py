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
            colors = []
            for color in vecs[key]["values"]:
                colors.append(int(color, base=16))
            dist = np.mean(np.abs(np.diff(colors)))
            vecs[key]["attributes"]["num_colors"] = int(num_clrs)
            vecs[key]["attributes"]["dist"] = int(dist)

    def update_json(self, compact=True):
        with open("color_vectors.json", "w") as fh:
            json.dump(self.color_vectors, fh, indent = None if compact else 4)

    def get_vec_keys(self, num_clrs_rng=None, dist_rng=None):
        vecs = self.color_vectors["color_vectors"]
        output = []
        if num_clrs_rng:
            for key in vecs:
                val = vecs[key]["attributes"]["num_colors"]
                if val >= num_clrs_rng[0] and val <= num_clrs_rng[1]:
                    output.append(key)
        final = []
        if dist_rng:
            if output:
                keys = output
            else:
                keys = vecs
            for key in keys:
                val = vecs[key]["attributes"]["dist"]
                if val >= dist_rng[0] and val <= dist_rng[1]:
                    final.append(key)
        else:
            final = output
        if num_clrs_rng is None and dist_rng is None:
            return vecs.keys()
        else:
            return final

    def get_colors(self, num_clrs_rng=None, dist_rng=None):
        keys = self.get_vec_keys(num_clrs_rng=num_clrs_rng, dist_rng=None)
        vecs = self.color_vectors["color_vectors"]
        colors = []
        for key in keys:
            colors.append(vecs[key]["values"])
        return colors
