import numpy as np
import Color from Color

class ColorWheel:
	def __init__(self, colors):
		self.pos = 0
		self.colors = colors

	def get_color(self):
		return self.colors[self.pos]

	def rotate(self, step):
		self.pos = (self.pos + (step % len(self.colors))) % len(self.colors)
