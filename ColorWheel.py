import numpy as np


class ColorWheel:
	def __init__(self, colors):
		self.pos = 0
		self.colors = colors

	def get_color(self):
		return self.colors[self.pos]

	def get_colors(self):
		return self.colors

	def rotate(self, step):
		self.pos = (self.pos + (step % len(self.colors))) % len(self.colors)

	def next_color(self):
		self.rotate(1)
		return self.get_color()
